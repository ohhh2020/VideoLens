"""知影 · GPU 推理环境一键配置（软件工程交付件：新机器一条命令跑起来）。

    python scripts/setup_gpu.py            # 检测 + 安装 + 验证
    python scripts/setup_gpu.py --check    # 只检测不安装（CI/诊断用）

它解决什么
----------
PyPI 的默认轮子全是 **CPU 构建**：``torch`` 默认 +cpu、``llama-cpp-python``
默认无 CUDA。用户机器明明有 N 卡（本项目实测：RTX 5060 8GB + CPU 轮子，
转写/视觉/大模型全在 CPU 上爬）。本脚本把"换 GPU 栈"收敛成一条命令：

1. **探测**：nvidia-smi（显卡型号 / 驱动 / 支持的 CUDA 版本 / 算力）。
2. **torch**：CPU 构建且存在 N 卡时，按驱动能力装对应 CUDA 轮子
   （Blackwell 及 12.8+ 驱动 → cu128；否则 cu126/cu124）。
3. **CUDA 运行库**：装 pip 版 ``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12``
   （faster-whisper 的 CTranslate2 后端需要；``core/models/_runtime.py`` 会在
   运行时自动把它们挂进 DLL 搜索路径，用户无需装 CUDA Toolkit、无需改 PATH）。
4. **llama-cpp-python**：GPU 卸载不可用时尝试预编译 CUDA 轮子
   （abetlen 索引）；**Blackwell（sm_100+）没有预编译轮子**，给出精确的
   源码编译命令（需要 VS2022 + CUDA Toolkit 12.8+），失败不阻断其余步骤。
5. **验证**：torch 真实张量计算 / llama GPU 卸载探测 / CTranslate2 设备数，
   打一张"当前推理栈状态表"，与 preflight 的 GPU 提示互相印证。

**隔离原则（重要）**：CUDA 版 torch（约 5-7GB）与 CUDA 运行库（约 1.5GB）
默认装到 **项目内** ``<项目根>/runtime/``（``--target`` + TEMP 也指到
``runtime/.tmp``），不污染系统盘——C 盘吃紧的机器尤其关键。
``core/models/_runtime.py`` 负责在运行时把这些目录挂进 ``sys.path`` 与
DLL 搜索路径；删掉 ``runtime/`` 即完全卸载。要用 conda 环境的用户可传
``--into-site`` 改为装进当前环境。

退出码：0 = 全部就绪（或无 N 卡、CPU 模式成立）；2 = 部分完成（有明确下一步）。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# --------------------------------------------------------------------------- #
# 探测
# --------------------------------------------------------------------------- #
@dataclass
class GpuInfo:
    name: str
    driver: str
    cuda_version: str  # 驱动支持的 CUDA 上限，如 "12.8"
    compute_cap: str   # 如 "12.0"（Blackwell sm_120）


def _run(argv: list[str], timeout: float = 60) -> tuple[int, str]:
    try:
        proc = subprocess.run(  # noqa: S603 - 参数固定
            argv, capture_output=True, text=True, timeout=timeout, check=False,
            encoding="utf-8", errors="replace",
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def detect_gpu() -> GpuInfo | None:
    """nvidia-smi 探测。没有 N 卡 / 没有 nvidia-smi 返回 None。"""
    exe = shutil.which("nvidia-smi") or shutil.which(
        "nvidia-smi", path=r"C:\Windows\System32"
    )
    if exe is None:
        return None
    code, out = _run([
        exe, "--query-gpu=name,driver_version,compute_cap",
        "--format=csv,noheader",
    ])
    if code != 0 or not out.strip():
        return None
    first = out.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 3:
        return None
    cuda = ""
    code2, out2 = _run([exe], timeout=20)
    if code2 == 0:
        match = re.search(r"CUDA Version[:\s]+([0-9][0-9.]*)", out2)
        if match:
            cuda = match.group(1)
    return GpuInfo(name=parts[0], driver=parts[1], compute_cap=parts[2], cuda_version=cuda)


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_flavor() -> tuple[str | None, bool]:
    """返回 (版本串, 是否 CUDA 构建)。

    **必须真实 import 判断**：Windows 上 PyPI 默认 torch 的元数据就是
    "2.14.0"（不带 +cpu 本地标签），靠版本串判断会把 CPU 构建误判成 CUDA。
    检查顺序：项目 runtime（隔离安装）→ 当前环境。import 约两三秒，
    只在有 N 卡的机器上付出。
    """
    version = _dist_version("torch")
    try:
        sys.path.insert(0, str(_ROOT))
        from core.models._runtime import ensure_torch_path  # noqa: PLC0415

        ensure_torch_path()
        import torch  # noqa: PLC0415

        return version or getattr(torch, "__version__", None), bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - runtime 与环境里都没有可用 torch
        return version, False


def llama_gpu_ok() -> bool | None:
    """True/False；None = 探测不可用（老版本/未安装）。"""
    try:
        spec = importlib.util.find_spec("llama_cpp")
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return None
    try:
        from llama_cpp import llama_supports_gpu_offload  # noqa: PLC0415

        return bool(llama_supports_gpu_offload())
    except Exception:  # noqa: BLE001
        return None


def ct2_device_count() -> int:
    try:
        import ctranslate2  # noqa: PLC0415

        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001
        return 0


def _pip_install(args: list[str]) -> tuple[bool, str]:
    code, out = _run([sys.executable, "-m", "pip", "install", *args], timeout=3600)
    return code == 0, out.strip().splitlines()[-6:] if out else out


# --------------------------------------------------------------------------- #
# 安装决策
# --------------------------------------------------------------------------- #
def pick_torch_index(gpu: GpuInfo) -> str | None:
    """按驱动能力挑 torch CUDA 轮子索引；None = 建议保持 CPU。"""
    try:
        cap_major = int(float(gpu.compute_cap.split(".")[0]))
    except (ValueError, IndexError):
        cap_major = 0
    try:
        cuda_major_minor = tuple(int(x) for x in gpu.cuda_version.split(".")[:2])
    except (ValueError, IndexError):
        cuda_major_minor = (0, 0)
    # Blackwell（sm_100/120）只认 cu128 起步的轮子
    if cap_major >= 10 or cuda_major_minor >= (12, 8):
        return "https://download.pytorch.org/whl/cu128"
    if cuda_major_minor >= (12, 6):
        return "https://download.pytorch.org/whl/cu126"
    if cuda_major_minor >= (12, 4):
        return "https://download.pytorch.org/whl/cu124"
    if cuda_major_minor != (0, 0):
        return "https://download.pytorch.org/whl/cu121"
    return None


#: llama-cpp-python 预编译 CUDA 轮子索引（abetlen 官方维护）。
_LLAMA_WHL = "https://abetlen.github.io/llama-cpp-python/whl/{tag}"

#: 项目内运行时布局（--into-site 未指定时启用）。
_RUNTIME_TORCH = Path(__file__).resolve().parents[1] / "runtime" / "torch_cu128"
_RUNTIME_CUDA = Path(__file__).resolve().parents[1] / "runtime" / "cuda12"
_RUNTIME_TMP = Path(__file__).resolve().parents[1] / "runtime" / ".tmp"


def _pip_install_isolated(args: list[str], target: Path | None) -> tuple[bool, list[str]]:
    """pip 安装：默认隔离到项目 runtime（--target + TEMP 同迁）。

    TEMP 必须一起指过去：pip 的下载/解压临时文件默认落在系统盘
    （C:\\...\\Temp），3GB 的 torch 轮子会把隔离意义抵消掉。
    """
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
    if target is not None:
        target.mkdir(parents=True, exist_ok=True)
        cmd += ["--target", str(target)]
    cmd += args
    env = dict(os.environ)
    if target is not None:
        _RUNTIME_TMP.mkdir(parents=True, exist_ok=True)
        tmp = str(_RUNTIME_TMP)
        env["TMP"] = env["TEMP"] = env["TMPDIR"] = tmp
    try:
        proc = subprocess.run(  # noqa: S603 - 参数由脚本拼装
            cmd, capture_output=True, text=True, timeout=3600, check=False,
            encoding="utf-8", errors="replace", env=env,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, out.strip().splitlines()[-6:]
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, [str(exc)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="知影 GPU 推理环境一键配置")
    parser.add_argument("--check", action="store_true", help="只检测不安装")
    parser.add_argument("--skip-torch", action="store_true", help="跳过 torch 步骤")
    parser.add_argument("--skip-llama", action="store_true", help="跳过 llama 步骤")
    parser.add_argument(
        "--into-site", action="store_true",
        help="装进当前 Python 环境（默认装到项目 runtime/，不污染系统盘）",
    )
    args = parser.parse_args(argv)

    print("=" * 64)
    print("  知影 · GPU 环境配置")
    print("=" * 64)

    gpu = detect_gpu()
    if gpu is None:
        print("[i] 未检测到 NVIDIA GPU（无 nvidia-smi 或无卡）。")
        print("    保持 CPU 模式即可 —— 项目全链路支持纯 CPU，只是更慢。")
        return 0
    print(f"[i] GPU: {gpu.name}  驱动 {gpu.driver}  算力 sm_{gpu.compute_cap.replace('.', '')}"
          f"  驱动支持 CUDA ≤ {gpu.cuda_version or '?'}")
    if args.into_site:
        print("[i] 安装目标：当前 Python 环境（--into-site）")
    else:
        print(f"[i] 安装目标：项目内隔离 runtime/（系统盘零占用；卸载=删目录）")

    partial = False

    # -- 1) torch -----------------------------------------------------------
    version, cuda_build = torch_flavor()
    if version is None:
        print("[!] 未安装 torch（BGE-M3 / SigLIP2 需要）")
    elif cuda_build:
        print(f"[ok] torch {version}（CUDA 构建）")
    else:
        print(f"[!] torch {version} 是 CPU 构建 —— 向量编码全程 CPU")
    index = pick_torch_index(gpu)
    torch_target = None if args.into_site else _RUNTIME_TORCH
    if version is not None and not cuda_build and index and not args.skip_torch:
        if args.check:
            print(f"[check] 将安装 torch（{index.rsplit('/', 1)[-1]}）→ {torch_target or '当前环境'}")
        else:
            print(f"[*] 安装 CUDA 版 torch（{index.rsplit('/', 1)[-1]}，下载约 2-3 GB）…")
            ok, tail = _pip_install_isolated(["torch", "--index-url", index], torch_target)
            print(("[ok] torch CUDA 已装到 " + (str(torch_target) if torch_target else "当前环境")
                   if ok else "[!] torch 安装失败，可稍后手动重试")
                  + ("\n    " + "\n    ".join(tail) if tail else ""))
            if not ok:
                partial = True
    elif index is None and not cuda_build:
        print("[!] 驱动支持的 CUDA 版本过低，无法匹配 torch CUDA 轮子；请升级显卡驱动")

    # -- 2) CUDA 运行库（CTranslate2 / faster-whisper 需要）------------------
    cuda_target = None if args.into_site else _RUNTIME_CUDA
    libs_present = (_RUNTIME_CUDA / "nvidia").is_dir() if not args.into_site else False
    if libs_present:
        print(f"[ok] CUDA 运行库已在 {cuda_target}")
    else:
        for pkg in ("nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cuda-nvrtc-cu12"):
            if _dist_version(pkg) is not None and args.into_site:
                print(f"[ok] {pkg} 已安装（当前环境）")
                continue
            if args.check:
                print(f"[check] 将安装: {pkg} → {cuda_target or '当前环境'}")
            else:
                print(f"[*] 安装 {pkg}（faster-whisper GPU 推理的运行库）…")
                ok, _tail = _pip_install_isolated([pkg], cuda_target)
                print(f"[{'ok' if ok else '!'}] {pkg}" + ("" if ok else " 安装失败"))
                if not ok:
                    partial = True

    # -- 3) llama-cpp-python -------------------------------------------------
    lgpu = llama_gpu_ok()
    if lgpu is None:
        print("[i] 未安装 llama-cpp-python（本地 LLM/视觉链路会在需要时装）")
    elif lgpu:
        print("[ok] llama-cpp-python 支持 GPU 卸载")
    else:
        print("[!] llama-cpp-python 是 CPU 构建 —— Qwen2.5/Qwen2-VL 全程 CPU")
        try:
            cap_major = int(float(gpu.compute_cap.split(".")[0]))
        except (ValueError, IndexError):
            cap_major = 0
        if cap_major >= 10:
            # Blackwell 没有预编译 CUDA 轮子（需要 CUDA 12.8+ 工具链）
            print("[i] Blackwell（RTX 50 系）暂无预编译轮子，需源码编译（一次性）：")
            print("      1) 安装 Visual Studio 2022（含 C++ 桌面开发）与 CUDA Toolkit 12.8+")
            print('      2) set CMAKE_ARGS=-DGGML_CUDA=on')
            print("      3) pip install llama-cpp-python --force-reinstall --no-cache-dir \\")
            print("            --no-binary llama-cpp-python")
            partial = True
        elif not args.skip_llama:
            tag = "cu125" if (gpu.cuda_version and gpu.cuda_version >= "12.5") else "cu124"
            url = _LLAMA_WHL.format(tag=tag)
            if args.check:
                print(f"[check] 将尝试预编译轮子: {url}")
            else:
                print(f"[*] 尝试预编译 CUDA 轮子（{tag}）…")
                ok, _tail = _pip_install_isolated([
                    "llama-cpp-python", "--force-reinstall",
                    "--extra-index-url", url,
                ], None)  # llama 是编译产物，体积小，装当前环境
                if ok and llama_gpu_ok():
                    print("[ok] llama-cpp-python 已切换 CUDA 构建")
                else:
                    print("[!] 预编译轮子不可用或不含本机算力；可源码编译（见上式命令）")
                    partial = True

    # -- 4) 验证汇总 ---------------------------------------------------------
    print("-" * 64)
    ct2 = ct2_device_count()
    print(f"  CTranslate2 CUDA 设备数: {ct2}" + ("（faster-whisper 将走 GPU）" if ct2 else ""))
    try:
        sys.path.insert(0, str(_ROOT))
        from core.models._runtime import ensure_cuda_dlls, ensure_torch_path  # noqa: PLC0415

        ensure_torch_path()
        ensure_cuda_dlls()
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            x = torch.zeros(8, device="cuda") + 1
            print(f"  torch CUDA 实测: ok（{torch.cuda.get_device_name(0)}，{x.sum().item():.0f}）")
        else:
            print("  torch CUDA: 不可用（CPU 构建）")
    except Exception as exc:  # noqa: BLE001
        print(f"  torch 探测失败: {exc}")
    lgpu2 = llama_gpu_ok()
    if lgpu2 is not None:
        print(f"  llama GPU 卸载: {'ok' if lgpu2 else '不可用（CPU 构建）'}")
    if not args.into_site and (_RUNTIME_TORCH.is_dir() or _RUNTIME_CUDA.is_dir()):
        print(f"  隔离目录: {_RUNTIME_TORCH.parent}（卸载 = 删除该目录；已在 .gitignore）")
    print("=" * 64)
    if partial:
        print("  部分步骤未完成 —— 按上面 [!] 的提示处理；python scripts/preflight.py 可复查。")
        return 2
    print("  完成。重启知影服务后生效；python scripts/preflight.py 可复查。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
