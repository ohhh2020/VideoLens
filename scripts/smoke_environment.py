"""环境冒烟检查：实际执行 FFmpeg，不把“存在可执行文件”等同于可用。"""
from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from videolens.config import load_settings


def main() -> int:
    """检查基础依赖、UTF-8 中文文件名和 16kHz 单声道音频输出。"""
    for module in ("pydantic", "fastapi", "httpx", "gradio", "PIL", "pysubs2", "keyring"):
        importlib.import_module(module)
    settings = load_settings()
    ffmpeg, ffprobe = shutil.which(settings.ffmpeg), shutil.which(settings.ffprobe)
    if not ffmpeg or not ffprobe:
        raise RuntimeError("缺少 FFmpeg/FFprobe，请在 VideoLens 环境安装 conda-forge::ffmpeg。")
    temporary_root = Path(settings.data_dir) / "tmp" / "environment-check"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="知影环境验证 ", dir=temporary_root) as temporary:
        target = Path(temporary) / "中文音频 空格.wav"
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=0.2", "-ar", "16000", "-ac", "1", str(target)], check=True)
        result = subprocess.run([ffprobe, "-v", "error", "-show_streams", "-of", "json", str(target)],
                                check=True, capture_output=True, text=True, encoding="utf-8")
        stream = json.loads(result.stdout)["streams"][0]
        assert stream["sample_rate"] == "16000" and stream["channels"] == 1
    print(json.dumps({"状态": "通过", "Python": sys.version.split()[0], "FFmpeg": ffmpeg,
                      "验证": "依赖导入、中文路径、实际音频生成与解析"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
