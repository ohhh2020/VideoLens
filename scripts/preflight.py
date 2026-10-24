"""知影 · 启动前预检（供 start.bat 调用，也可单独跑）。

设计目标：**3 秒内跑完，不加载任何模型权重**。只做一件事——
在用户点"启动"之前，把所有会让服务跑不起来的原因一次性列出来，
并按严重程度给结论：

    READY      可以正常启动（关键项全绿）
    DEGRADED   能启动，但有降级（缺可选模型/依赖，功能不全但不会崩）
    BROKEN     启动前必须先修（缺 Python 包/配置加载失败/目录不可写）

输出既有人读的行，也有机器可解析的 ``KEY=STATE|detail`` 行，
bat 用 ``for /f "tokens=1,* delims=="`` 直接吃。

单独使用::

    python scripts/preflight.py            # 人读 + 机器行
    python scripts/preflight.py --json     # 只输出 JSON（CI 用）
    python scripts/preflight.py --quiet    # 只输出机器行

退出码：0 = READY，1 = DEGRADED，2 = BROKEN。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import socket
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# 允许从任意目录运行（bat 里 cd 过一次，但单独调用时也要稳）
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# --------------------------------------------------------------------------- #
# 结果模型
# --------------------------------------------------------------------------- #
OK = "OK"
WARN = "WARN"
MISS = "MISS"
FAIL = "FAIL"
SKIP = "SKIP"

# 严重度：决定最终结论
_SEVERITY = {OK: 0, SKIP: 0, WARN: 1, MISS: 1, FAIL: 2}

READY = "READY"
DEGRADED = "DEGRADED"
BROKEN = "BROKEN"


@dataclass
class Check:
    """一条检查结果。"""

    key: str
    state: str
    detail: str = ""
    note: str = ""  # 人读的补充说明（缺失时怎么办）

    @property
    def severity(self) -> int:
        return _SEVERITY.get(self.state, 0)

    def line(self) -> str:
        return f"{self.key}={self.state}|{self.detail}"


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)
    hard_errors: list[str] = field(default_factory=list)

    def add(
        self,
        key: str,
        state: str,
        detail: str = "",
        note: str = "",
    ) -> Check:
        chk = Check(key=key, state=state, detail=str(detail), note=str(note))
        self.checks.append(chk)
        return chk

    @property
    def verdict(self) -> str:
        worst = max((c.severity for c in self.checks), default=0)
        if worst >= 2:
            return BROKEN
        if worst == 1:
            return DEGRADED
        return READY

    @property
    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.started) * 1000)

    def get(self, key: str) -> Check | None:
        for c in self.checks:
            if c.key == key:
                return c
        return None


# --------------------------------------------------------------------------- #
# 各检查项
# --------------------------------------------------------------------------- #
_GLYPH = {OK: "[OK]", WARN: "[!]", MISS: "[ ]", FAIL: "[X]", SKIP: "[-]"}

# 依赖清单：(模块名, pip 包名, 用途, 是否关键)
# 关键 = 缺了服务起不来或主链路断掉
_DEPS: tuple[tuple[str, str, str, bool], ...] = (
    ("yaml", "pyyaml", "读取 YAML 配置", True),
    ("httpx", "httpx", "HTTP 客户端 / 远程模式", True),
    ("numpy", "numpy", "数值计算", True),
    ("fastapi", "fastapi", "HTTP 服务", True),
    ("uvicorn", "uvicorn", "ASGI 服务器", True),
    ("pydantic", "pydantic", "请求/响应模型", True),
    ("cv2", "opencv-python", "清晰度评估（Laplacian）", False),
    ("scenedetect", "scenedetect", "镜头切分", False),
    ("PIL", "pillow", "图片处理 / 关键帧缩放", False),
    ("faster_whisper", "faster-whisper", "语音转写", False),
    ("chromadb", "chromadb", "向量库", False),
    ("jieba", "jieba", "中文分词（BM25）", False),
    ("yt_dlp", "yt-dlp", "在线视频下载", False),
    ("torch", "torch", "深度模型底座", False),
    ("transformers", "transformers", "BGE-M3 / SigLIP2 编码", False),
    ("gradio", "gradio", "Gradio 备用界面", False),
    ("llama_cpp", "llama-cpp-python", "本地 GGUF 推理", False),
)


def check_python(rep: Report) -> None:
    ver = sys.version.split()[0]
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 11:
        rep.add("PYTHON", OK, f"{ver} @ {sys.executable}")
    else:
        rep.add(
            "PYTHON",
            FAIL,
            f"{ver}（需要 3.11+）",
            note="装 Python 3.11 或 3.12，别用 3.13（部分轮子还没出）",
        )


def check_layout(rep: Report) -> None:
    need = ["core", "api", "clients", "configs", "scripts"]
    missing = [d for d in need if not (_ROOT / d).is_dir()]
    if missing:
        rep.add(
            "LAYOUT",
            FAIL,
            f"缺少目录：{', '.join(missing)}（当前根 {_ROOT}）",
            note="bat 必须放在项目根目录，或从根目录运行",
        )
    else:
        rep.add("LAYOUT", OK, str(_ROOT))


def check_config(rep: Report, config_path: str | None = None) -> Any:
    try:
        from core.config import load_app_config  # noqa: PLC0415

        cfg = load_app_config(config_path=config_path)
    except Exception as exc:  # noqa: BLE001
        rep.add(
            "CONFIG",
            FAIL,
            f"加载失败：{type(exc).__name__}: {exc}",
            note="检查 configs/app_config.yaml 的 YAML 语法与字段名",
        )
        rep.hard_errors.append(f"配置加载失败：{exc}")
        return None

    src = cfg.source_file or "（内置默认值）"
    rep.add("CONFIG", OK, f"{cfg.root_dir} | {src}")

    # 目录可写（模型目录只查存在，不强制可写——只读挂载也能用）
    bad: list[str] = []
    for name in ("data", "frames", "audio", "temp", "logs"):
        try:
            p = Path(cfg.path(name))
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".zhiying-preflight"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            bad.append(f"{name}({exc.__class__.__name__})")
    if bad:
        rep.add(
            "DIRS",
            FAIL,
            f"不可写：{', '.join(bad)}",
            note="把项目移到用户有写权限的路径，或以管理员身份运行",
        )
        rep.hard_errors.append("数据目录不可写")
    else:
        rep.add("DIRS", OK, cfg.path("data"))

    # 磁盘剩余
    try:
        usage = shutil.disk_usage(str(cfg.root_dir))
        free_gb = usage.free / (1024 ** 3)
        need_gb = 2.0
        if free_gb < need_gb:
            rep.add(
                "DISK",
                WARN,
                f"剩余 {free_gb:.1f} GB（建议 ≥{need_gb:.0f} GB）",
                note="分析视频会产生关键帧与向量库，空间不足会中途失败",
            )
        else:
            rep.add("DISK", OK, f"剩余 {free_gb:.1f} GB")
    except OSError:
        rep.add("DISK", SKIP, "无法读取磁盘信息")

    return cfg


def check_ffmpeg(rep: Report, cfg: Any) -> None:
    try:
        from core.media.ffmpeg_utils import resolve_binary  # noqa: PLC0415

        try:
            ff = Path(resolve_binary("ffmpeg"))
        except Exception:  # noqa: BLE001
            ff = None
        try:
            fp = Path(resolve_binary("ffprobe"))
        except Exception:  # noqa: BLE001
            fp = None
        if ff is None:
            rep.add(
                "FFMPEG",
                FAIL,
                "未找到 ffmpeg",
                note="装了 ffmpeg 但不在 PATH？设环境变量 ZHIYING_FFMPEG=D:\\...\\ffmpeg.exe",
            )
        else:
            import subprocess  # noqa: PLC0415

            try:
                out = subprocess.run(
                    [str(ff), "-version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    encoding="utf-8",
                    errors="replace",
                ).stdout
                first = (out.splitlines() or ["?"])[0]
                ver = first.split("version")[-1].strip().split()[0] if "version" in first else "?"
                rep.add("FFMPEG", OK, f"{ver} @ {ff}")
            except Exception:  # noqa: BLE001
                rep.add("FFMPEG", WARN, f"存在但执行失败：{ff}", note="可能是损坏的构建或权限问题")
        if fp is None:
            rep.add(
                "FFPROBE",
                FAIL,
                "未找到 ffprobe",
                note="ffprobe 与 ffmpeg 同目录；设 ZHIYING_FFPROBE 指定路径",
            )
        else:
            rep.add("FFPROBE", OK, str(fp))
    except Exception as exc:  # noqa: BLE001
        rep.add("FFMPEG", FAIL, f"探测异常：{type(exc).__name__}: {exc}")


def check_deps(rep: Report) -> None:
    seen: set[str] = set()
    missing_critical: list[str] = []
    missing_optional: list[str] = []
    present = 0
    total = 0
    details: list[str] = []

    for mod, pip_name, purpose, critical in _DEPS:
        if mod in seen:
            continue
        seen.add(mod)
        total += 1
        try:
            spec = importlib.util.find_spec(mod)
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            present += 1
            continue
        if critical:
            missing_critical.append(pip_name)
        else:
            missing_optional.append(pip_name)
            details.append(f"{pip_name}({purpose})")

    if missing_critical:
        rep.add(
            "DEPS",
            FAIL,
            f"缺关键依赖：{', '.join(missing_critical)}",
            note=f"pip install -r requirements.txt（已装 {present}/{total}）",
        )
        rep.hard_errors.append("缺关键依赖：" + ", ".join(missing_critical))
    elif missing_optional:
        rep.add(
            "DEPS",
            WARN,
            f"{present}/{total}；缺 {', '.join(missing_optional)}",
            note="缺的是可选依赖，对应功能会降级：" + "; ".join(details[:4]),
        )
    else:
        rep.add("DEPS", OK, f"{present}/{total}")


def check_models(rep: Report, cfg: Any) -> None:
    """检查五类角色各自配置的模型是否已落盘。"""
    if cfg is None:
        rep.add("MODELS", SKIP, "配置不可用")
        return

    roles: list[tuple[str, str, bool]] = []
    try:
        roles.append(("ASR_MODEL", str(cfg.asr.model), True))
    except AttributeError:
        pass
    try:
        roles.append(("EMB_MODEL", str(cfg.embedding.text_model), True))
        roles.append(("IMG_MODEL", str(cfg.embedding.image_model), False))
    except AttributeError:
        pass
    try:
        roles.append(("VL_MODEL", str(cfg.vl.model), False))
    except AttributeError:
        pass
    # llm.mode（auto/cloud/local）：local 与 auto 都依赖本地 GGUF 兜底
    try:
        llm_mode = str(getattr(cfg.llm, "mode", "auto") or "auto")
        if llm_mode in {"local", "auto"}:
            roles.append(("LLM_MODEL", str(cfg.llm.local.model), False))
    except Exception:  # noqa: BLE001
        llm_mode = "auto"

    try:
        from core.models.manager import get_model_manager  # noqa: PLC0415

        mgr = get_model_manager()
        report = mgr.status_report()
    except Exception as exc:  # noqa: BLE001
        for key, name, critical in roles:
            rep.add(key, SKIP, f"{name}（状态不可读：{type(exc).__name__}）")
        rep.add("MODELS", WARN, f"无法读取模型状态：{exc}")
        return

    present = {getattr(s, "name", "?"): str(getattr(getattr(s, "state", None), "value", "")) for s in report}
    missing_critical: list[str] = []
    missing_optional: list[str] = []

    for key, name, critical in roles:
        state = present.get(name)
        if state == "present":
            rep.add(key, OK, name)
        elif state is None:
            # 配置里写了但 models.yaml 没这个模型
            rep.add(
                key,
                WARN if critical else MISS,
                f"{name}（models.yaml 无此条目）",
                note="模型名写错，或 models.yaml 里没有该规格",
            )
            (missing_critical if critical else missing_optional).append(name)
        else:
            rep.add(
                key,
                MISS,
                f"{name}（{state}）",
                note=f"下载：python scripts/download_models.py --models {name}",
            )
            (missing_critical if critical else missing_optional).append(name)

    if missing_critical:
        rep.add(
            "MODELS",
            MISS,
            f"缺核心权重：{', '.join(missing_critical)}",
            note="现在能启动，但转写/检索会走降级路径，结果不可信",
        )
    elif missing_optional:
        rep.add(
            "MODELS",
            WARN,
            f"核心齐；缺可选：{', '.join(missing_optional)}",
            note="对应功能降级：图像检索变纯文本、画面描述缺失",
        )
    else:
        rep.add("MODELS", OK, "全部就绪")

    # LLM 后端可用性（按 llm.mode 与实际密钥/权重/Ollama 探测综合判断）
    local_name = str(getattr(getattr(cfg.llm, "local", None), "model", "?"))
    local_ready = present.get(local_name) == "present"
    try:
        import importlib.util as _u  # noqa: PLC0415

        has_llama_cpp = _u.find_spec("llama_cpp") is not None
    except (ImportError, ValueError):
        has_llama_cpp = False

    if llm_mode == "cloud":
        key = ""
        try:
            key = cfg.llm.cloud.resolved_key()
        except Exception:  # noqa: BLE001
            pass
        base = str(getattr(cfg.llm.cloud, "base_url", "") or "")
        if key and base:
            rep.add("LLM", OK, f"cloud:{base}")
        else:
            rep.add(
                "LLM",
                WARN,
                f"cloud:{base or '（未配 base_url）'}（缺密钥/地址）",
                note="在网页设置页填 base_url 与 API key，或改 llm.mode=local 走本地",
            )
    elif llm_mode == "local":
        if local_ready and has_llama_cpp:
            rep.add("LLM", OK, f"local:{local_name}")
        elif _ollama_alive():
            rep.add("LLM", OK, f"local 兜底 → 本地 OpenAI 兼容服务（{local_name} 权重缺失）")
        else:
            rep.add(
                "LLM",
                MISS,
                f"local:{local_name}（权重或 llama-cpp-python 缺失，且未探测到 Ollama）",
                note="下载权重 / pip install llama-cpp-python / 启动 Ollama，"
                     "三者其一即可；全缺时问答退化为关键词抽取",
            )
    else:  # auto
        key = ""
        try:
            key = cfg.llm.cloud.resolved_key()
        except Exception:  # noqa: BLE001
            pass
        if key and str(getattr(cfg.llm.cloud, "base_url", "") or ""):
            detail = f"cloud 优先"
            detail += "；local 兜底就绪" if (local_ready and has_llama_cpp) else "；local 兜底缺失"
            rep.add("LLM", OK, detail)
        elif local_ready and has_llama_cpp:
            rep.add("LLM", OK, f"local:{local_name}（云端未配密钥）")
        elif _ollama_alive():
            rep.add("LLM", OK, "本地 OpenAI 兼容服务（Ollama）")
        else:
            rep.add(
                "LLM",
                MISS,
                "云端无密钥、本地权重缺失且未探测到 Ollama → 兜底关键词抽取",
                note="配云端密钥 / 下载本地权重 / 启动 Ollama，任选其一",
            )


def _ollama_alive(timeout: float = 0.4) -> bool:
    """探测本机 OpenAI 兼容服务（Ollama/LM Studio）是否在线。"""
    try:
        from core.models.llm import detect_local_server  # noqa: PLC0415

        return detect_local_server(timeout=timeout) is not None
    except Exception:  # noqa: BLE001
        return False


def check_storage(rep: Report, cfg: Any) -> None:
    if cfg is None:
        rep.add("DB", SKIP, "配置不可用")
        return
    try:
        from core.storage import SqliteRepository  # noqa: PLC0415

        repo = SqliteRepository.from_config(cfg)
        try:
            stats = repo.stats()
            total = stats.videos
            clips = stats.clips
            rep.add("DB", OK, f"videos={total} clips={clips}")
        finally:
            repo.close()
    except Exception as exc:  # noqa: BLE001
        rep.add(
            "DB",
            FAIL,
            f"{type(exc).__name__}: {exc}",
            note="python scripts/init_db.py 重建库；或删掉 data/zhiying.db 重来",
        )
        rep.hard_errors.append(f"数据库不可用：{exc}")


def _port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def check_ports(rep: Report, cfg: Any) -> None:
    if cfg is None:
        rep.add("PORT", SKIP, "配置不可用")
        return
    try:
        api_port = int(cfg.api.port)
    except Exception:  # noqa: BLE001
        api_port = 8000
    free = _port_free(api_port)
    if free:
        rep.add("PORT", OK, str(api_port))
    else:
        # 往后找 10 个
        alt = next((p for p in range(api_port + 1, api_port + 11) if _port_free(p)), None)
        if alt:
            rep.add(
                "PORT",
                WARN,
                f"{api_port} 被占用 → 建议用 {alt}",
                note=f"start.bat 会自动改用 {alt}",
            )
        else:
            rep.add(
                "PORT",
                WARN,
                f"{api_port}~{api_port + 10} 全被占用",
                note="可能有多个知影实例在跑，或别的程序占了端口",
            )

    try:
        gport = int(cfg.gradio.port)
    except Exception:  # noqa: BLE001
        gport = 7860
    if gport != api_port and not _port_free(gport):
        rep.add("PORT_UI", WARN, f"Gradio 端口 {gport} 被占用", note="Gradio 界面可能起不来")


def check_vectors(rep: Report, cfg: Any) -> None:
    """只查 chromadb 能否导入 + 向量目录存在，不初始化客户端（慢）。"""
    try:
        import importlib.util as _u  # noqa: PLC0415

        if _u.find_spec("chromadb") is None:
            rep.add("VECTOR", WARN, "未安装 chromadb → 检索退化为 BM25", note="pip install chromadb")
            return
    except (ImportError, ValueError):
        rep.add("VECTOR", WARN, "chromadb 探测失败")
        return
    if cfg is None:
        rep.add("VECTOR", SKIP, "配置不可用")
        return
    try:
        p = Path(cfg.path("chroma"))
        if p.exists():
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            rep.add("VECTOR", OK, f"{p} ({size / 1024 / 1024:.1f} MB)")
        else:
            rep.add("VECTOR", OK, f"{p}（首次分析时创建）")
    except OSError as exc:
        rep.add("VECTOR", WARN, f"{exc}")


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
_TITLES = {
    "PYTHON": "Python",
    "LAYOUT": "项目结构",
    "CONFIG": "配置",
    "DIRS": "数据目录",
    "DISK": "磁盘空间",
    "FFMPEG": "FFmpeg",
    "FFPROBE": "FFprobe",
    "DEPS": "依赖包",
    "ASR_MODEL": "语音转写模型",
    "EMB_MODEL": "文本向量模型",
    "IMG_MODEL": "图像向量模型",
    "VL_MODEL": "视觉理解模型",
    "LLM_MODEL": "本地大模型",
    "MODELS": "权重总览",
    "LLM": "问答后端",
    "DB": "数据库",
    "VECTOR": "向量库",
    "PORT": "API 端口",
    "PORT_UI": "Gradio 端口",
}

_VERDICT_TEXT = {
    READY: ("环境就绪，可以启动。", ""),
    DEGRADED: (
        "可以启动，但存在降级项。",
        "降级意味着部分功能不可用或结果质量下降，不是崩溃。",
    ),
    BROKEN: ("环境有问题，启动大概率失败。", "先修复上面标 [X] 的项。"),
}


def render(rep: Report, *, quiet: bool = False) -> str:
    out: list[str] = []
    if not quiet:
        out.append("")
        out.append("=" * 64)
        out.append(f"  知影 · 启动预检    ({rep.elapsed_ms} ms)")
        out.append("=" * 64)
        for chk in rep.checks:
            title = _TITLES.get(chk.key, chk.key)
            glyph = _GLYPH.get(chk.state, "?")
            pad = "　" * max(0, 8 - len(title))
            out.append(f"  {glyph} {title}{pad}{chk.detail}")
            if chk.note and chk.state in (WARN, MISS, FAIL):
                out.append(f"      → {chk.note}")
        out.append("-" * 64)
        verdict = rep.verdict
        head, tail = _VERDICT_TEXT[verdict]
        out.append(f"  结论：{verdict} — {head}")
        if tail:
            out.append(f"        {tail}")
        if rep.hard_errors:
            for err in rep.hard_errors:
                out.append(f"  · {err}")
        out.append("=" * 64)
        out.append("")

    # 机器可读段（bat 解析用）
    out.append("###PRECHECK###")
    for chk in rep.checks:
        out.append(chk.line())
    out.append(f"VERDICT={rep.verdict}")
    out.append(f"ELAPSED={rep.elapsed_ms}")
    out.append("###END###")
    return "\n".join(out)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="知影启动前环境预检（3 秒内完成，不加载模型）",
    )
    parser.add_argument("--json", action="store_true", help="只输出 JSON")
    parser.add_argument("--quiet", action="store_true", help="只输出机器可读行")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--skip-ports", action="store_true", help="跳过端口探测")
    args = parser.parse_args(list(argv) if argv is not None else None)

    rep = Report()
    try:
        check_python(rep)
        check_layout(rep)
        # --config 直接传给 load_app_config。注意不能写 ZHIYING_CONFIG 环境变量 ——
        # 那是 models.yaml 的路径变量（ModelManager 读），拿 app_config.yaml 的
        # 路径去污染它，模型状态检查会去找错文件。
        cfg = check_config(rep, config_path=args.config)
        check_ffmpeg(rep, cfg)
        check_deps(rep)
        check_models(rep, cfg)
        check_storage(rep, cfg)
        check_vectors(rep, cfg)
        if not args.skip_ports:
            check_ports(rep, cfg)
    except Exception:  # noqa: BLE001
        # 预检自己不能崩——崩了就当 BROKEN 报出来
        rep.add("PRECHECK", FAIL, "预检脚本异常")
        rep.hard_errors.append(traceback.format_exc(limit=3))

    verdict = rep.verdict

    if args.json:
        payload = {
            "verdict": verdict,
            "elapsed_ms": rep.elapsed_ms,
            "checks": [
                {"key": c.key, "state": c.state, "detail": c.detail, "note": c.note}
                for c in rep.checks
            ],
            "errors": rep.hard_errors,
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(render(rep, quiet=args.quiet) + "\n")

    return {READY: 0, DEGRADED: 1, BROKEN: 2}[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
