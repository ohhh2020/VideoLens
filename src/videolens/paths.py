"""在导入第三方库之前集中设置缓存，避免默认写入用户目录。"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def configure_paths() -> Path:
    """项目所有可控缓存都在 cache；环境变量覆盖外部库的用户级默认目录。"""
    root = Path(os.environ.get("VIDEOLENS_ROOT", Path(__file__).resolve().parents[2])).resolve()
    locations = {
        "TEMP": "cache/tmp", "TMP": "cache/tmp", "TMPDIR": "cache/tmp",
        "XDG_CACHE_HOME": "cache", "HF_HOME": "cache/huggingface",
        "HF_HUB_CACHE": "models/huggingface", "HF_XET_CACHE": "cache/huggingface/xet",
        "TORCH_HOME": "cache/torch", "SENTENCE_TRANSFORMERS_HOME": "models/sentence-transformers",
        "GRADIO_TEMP_DIR": "cache/gradio", "PIP_CACHE_DIR": "cache/pip",
        "MPLCONFIGDIR": "cache/matplotlib", "NUMBA_CACHE_DIR": "cache/numba",
        "CUDA_CACHE_PATH": "cache/cuda", "TRITON_CACHE_DIR": "cache/triton",
        "HOME": "cache/home", "USERPROFILE": "cache/home",
        "APPDATA": "cache/appdata", "LOCALAPPDATA": "cache/localappdata",
    }
    for name, relative in locations.items():
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    # tempfile 可能已被服务器导入，清除其已缓存的系统临时目录。
    tempfile.tempdir = str(root / "cache/tmp")
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    return root


def open_project_browser(url: str) -> None:
    """使用独立浏览器资料目录；报告下载也默认落在项目内。"""
    import json
    import subprocess
    candidates = [Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
                  Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe"]
    executable = next((path for path in candidates if path.is_file()), None)
    if executable is None:
        raise RuntimeError("未找到 Edge 或 Chrome，请手动访问网页并将下载目录设为项目 data/exports")
    root = configure_paths()
    profile = root / "cache/browser"
    preferences = profile / "Default/Preferences"
    preferences.parent.mkdir(parents=True, exist_ok=True)
    # 仅首次创建我们的专用资料目录，不覆盖用户正在运行的浏览器配置。
    if not preferences.exists():
        preferences.write_text(json.dumps({"download": {"default_directory": str(root / "data/exports"),
                                                        "prompt_for_download": False}}, ensure_ascii=False), encoding="utf-8")
    subprocess.Popen([str(executable), "--user-data-dir=" + str(profile),
                      "--disk-cache-dir=" + str(root / "cache/browser-disk"), "--no-first-run", url],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
