"""代码与运行目录的唯一边界；不依赖当前工作目录或系统 PATH。"""

from __future__ import annotations
import logging
import os
from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[1]


def runtime_root(source_root: str | Path | None = None) -> Path:
    root = (Path(source_root) if source_root is not None else SOURCE_ROOT).resolve()
    # 独立配置目录（如临时测试配置）不继承主项目的运行目录。
    override = os.environ.get("VIDEOLENS_RUNTIME_ROOT") if root == SOURCE_ROOT else None
    if override:
        return Path(override).expanduser().resolve()
    path = (
        root.parent / "VideoLens-runtime" if root == SOURCE_ROOT else root / "runtime"
    )
    return path.resolve()


def runtime_path(name: str, source_root: str | Path | None = None) -> Path:
    return runtime_root(source_root) / name


def repair_conda_ca_path() -> Path | None:
    """修正 Windows Conda 的 Bash 激活脚本写入的失效默认 CA 路径。

    只处理当前 Conda 环境的 ssl/cacert.pem，不覆盖用户的自定义证书设置，
    不禁用证书校验，也不修改 Conda、系统或父终端的环境配置。
    """
    value = os.environ.get("SSL_CERT_FILE", "")
    if sys.platform != "win32" or not value:
        return None
    configured = Path(value).expanduser()
    if configured.is_file():
        return None
    prefix = Path(sys.prefix)
    if not (prefix / "conda-meta").is_dir():
        return None
    if configured.resolve() != (prefix / "ssl/cacert.pem").resolve():
        return None

    candidate = prefix / "Library/ssl/cacert.pem"
    if not candidate.is_file():
        try:
            import certifi
        except ImportError:
            return None
        candidate = Path(certifi.where())
    if not candidate.is_file():
        return None

    import ssl
    # 先让 OpenSSL 实际加载证书包，绝不使用 verify=False 绕过校验。
    ssl.create_default_context(cafile=str(candidate))
    os.environ["SSL_CERT_FILE"] = str(candidate.resolve())
    logging.getLogger("zhiying.runtime").warning(
        "Conda 默认 CA 路径不存在，本进程改用有效证书包：%s（HTTPS 校验保持开启）",
        candidate,
    )
    return candidate


def configure_runtime() -> None:
    """只配置本进程的缓存和临时目录，不改动 PATH 或系统设置。"""
    import tempfile

    runtime = runtime_root()
    for name in ("cache/temp", "cache/huggingface", "cache/conda/pkgs"):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(runtime / "cache/huggingface"))
    os.environ.setdefault("CONDA_PKGS_DIRS", str(runtime / "cache/conda/pkgs"))
    os.environ["TEMP"] = os.environ["TMP"] = str(runtime / "cache/temp")
    tempfile.tempdir = str(runtime / "cache/temp")
    repair_conda_ca_path()
