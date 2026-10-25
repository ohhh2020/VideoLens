"""项目内 GPU 运行时（``runtime/``）的路径注入。

为什么需要这个模块
------------------
PyPI 默认轮子是 CPU 构建；GPU 化的三个大件（CUDA 版 torch 约 5-7GB、
cublas/cudnn 运行库约 1.5GB）如果按默认方式 ``pip install``，会全部装进
用户目录（C 盘）——项目明明在 D 盘，C 盘却被吃掉近 10GB。

约定：``scripts/setup_gpu.py --runtime`` 把这些大件装到
``<项目根>/runtime/``（gitignore，随项目走、随项目删）::

    runtime/
      torch_cu128/     # pip --target：torch + bundled CUDA DLL
      cuda12/          # pip --target：nvidia-cublas/cudnn/nvrtc（CT2 用）

本模块只做两件事，全部幂等、失败静默（退回系统安装/纯 CPU）：

1. :func:`ensure_torch_path` —— 把 ``runtime/torch_cu128`` 插到
   ``sys.path`` 最前（若系统里已有可用的 torch 则不动）；
2. :func:`ensure_cuda_dlls` —— 把 ``runtime/cuda12/nvidia/*/bin`` 挂进
   DLL 搜索路径（Windows），顺带覆盖 site-packages 里 pip 版 nvidia 包
   的场景（用户级安装不在 PATH 上）。

调用点：所有 ``import torch`` / 加载 CT2 模型之前（asr / embed / manager）。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("zhiying.models.runtime")

__all__ = ["ensure_cuda_dlls", "ensure_torch_path", "runtime_root"]

_ROOT = Path(__file__).resolve().parents[2]
_DLL_HOOKED = False
_TORCH_CHECKED = False


def runtime_root() -> Path:
    """``<项目根>/runtime/``（core/models/_runtime.py → 上溯两级）。"""
    return _ROOT / "runtime"


def ensure_torch_path() -> None:
    """让 ``import torch`` 优先命中项目 runtime 里的 CUDA 版（若存在）。"""
    global _TORCH_CHECKED
    if _TORCH_CHECKED:
        return
    _TORCH_CHECKED = True
    target = runtime_root() / "torch_cu128"
    if not target.is_dir():
        return
    try:
        import torch  # noqa: PLC0415,F401

        return  # 系统已有 torch，尊重现有安装
    except ImportError:
        pass
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))
        logger.info("已挂载项目内 torch: %s", target)


def ensure_cuda_dlls() -> None:
    """把 CUDA 运行库目录挂进 Windows DLL 搜索路径（幂等）。

    覆盖两种布局：项目 runtime（``runtime/cuda12/nvidia/*/bin``）与
    pip 用户级安装（``site-packages/nvidia/*/bin``）。CTranslate2 在
    加载 CUDA 模型时按名字找 cublas64_12.dll / cudnn64_9.dll，
    不做这一步就只能靠系统 PATH —— "有卡缺运行库"的机器会在
    CUDA 档反复失败。
    """
    global _DLL_HOOKED
    if _DLL_HOOKED or sys.platform != "win32":
        return
    _DLL_HOOKED = True
    roots: list[Path] = [runtime_root() / "cuda12"]

    try:
        import site

        for getter in (getattr(site, "getsitepackages", None), site.getusersitepackages):
            try:
                value = getter()
            except Exception:  # noqa: BLE001
                continue
            for item in (value if isinstance(value, list) else [value]):
                if item:
                    roots.append(Path(str(item)))
    except Exception:  # noqa: BLE001
        pass

    added = 0
    for root in roots:
        nvidia = root / "nvidia"
        if not nvidia.is_dir():
            # --target 布局：包直接铺在根下（runtime/cuda12/nvidia/...）
            # 与 site-packages 相同；torch_cu128 的 bundled CUDA 也在
            # torch_cu128/lib 等目录，一并挂上。
            pass
        candidates: list[Path] = []
        if nvidia.is_dir():
            candidates.extend(p for p in nvidia.glob("*/bin") if p.is_dir())
        for sub in ("lib", "library", "bin"):
            alt = root / sub
            if alt.is_dir():
                candidates.append(alt)
        for bin_dir in candidates:
            try:
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                added += 1
            except OSError:
                continue
    if added:
        logger.debug("已挂载 %d 个 CUDA 运行库目录", added)
