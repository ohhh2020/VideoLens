from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    value = _env_str(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class Settings:
    openai_api_key: str | None
    openai_base_url: str
    vlm_model: str
    llm_model: str
    asr_model: str
    asr_device: str
    asr_compute_type: str
    frame_interval: int
    max_frames: int
    video_dir: Path
    output_dir: Path
    cookies_file: Path | None

    @classmethod
    def from_env(cls) -> "Settings":
        video_dir = Path(_env_str("VIDEO_DIR") or "data/videos")
        output_dir = Path(_env_str("OUTPUT_DIR") or "data/output")
        if not video_dir.is_absolute():
            video_dir = ROOT_DIR / video_dir
        if not output_dir.is_absolute():
            output_dir = ROOT_DIR / output_dir

        cookies_file = _env_str("COOKIES_FILE")
        return cls(
            openai_api_key=_env_str("OPENAI_API_KEY"),
            openai_base_url=_env_str("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            vlm_model=_env_str("VLM_MODEL") or "gpt-4o-mini",
            llm_model=_env_str("LLM_MODEL") or "gpt-4o-mini",
            asr_model=_env_str("ASR_MODEL") or "base",
            asr_device=_env_str("ASR_DEVICE") or "cpu",
            asr_compute_type=_env_str("ASR_COMPUTE_TYPE") or "int8",
            frame_interval=_env_int("FRAME_INTERVAL", 8),
            max_frames=_env_int("MAX_FRAMES", 20),
            video_dir=video_dir,
            output_dir=output_dir,
            cookies_file=Path(cookies_file) if cookies_file else None,
        )

    def ensure_dirs(self) -> None:
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


settings = Settings.from_env()

