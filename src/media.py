from __future__ import annotations

import subprocess
from pathlib import Path


class MediaError(RuntimeError):
    pass


def _run_ffprobe(path: Path) -> str:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise MediaError("未找到 ffprobe，请先安装 FFmpeg 并加入 PATH。") from exc
    except subprocess.CalledProcessError as exc:
        raise MediaError(f"无法读取视频时长：{exc.stderr.strip()}") from exc
    return result.stdout.strip()


def probe_duration(path: Path) -> float:
    try:
        value = _run_ffprobe(path)
        return round(float(value), 2)
    except MediaError:
        import cv2

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps <= 0:
            return 0.0
        return round(frames / fps, 2)


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / "audio.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(audio_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise MediaError("未找到 ffmpeg，请先安装 FFmpeg 并加入 PATH。") from exc
    except subprocess.CalledProcessError as exc:
        raise MediaError(f"音频提取失败：{exc.stderr.strip()[-500:]}") from exc

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise MediaError("音频提取结果为空，可能该视频没有音频轨道。")
    return audio_path


def format_ts(seconds: float) -> str:
    if isinstance(seconds, str):
        try:
            seconds = float(seconds)
        except ValueError:
            return seconds
    seconds = max(0, int(float(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
