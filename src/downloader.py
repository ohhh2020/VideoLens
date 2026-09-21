from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".flv"}


class DownloadError(RuntimeError):
    pass


def is_local_video_path(value: str) -> bool:
    path = Path(value)
    return path.exists() and path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def _find_video_file(out_dir: Path, prepared: Path, video_id: str) -> Path:
    if prepared.exists() and prepared.suffix.lower() in VIDEO_EXTENSIONS:
        return prepared

    candidates = [
        path
        for path in out_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and video_id in path.stem
    ]
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)

    raise DownloadError(f"下载完成，但未找到视频文件。目标目录：{out_dir}")


def extract_metadata(url: str, settings) -> dict:
    if is_local_video_path(url):
        from .media import probe_duration

        path = Path(url)
        return {
            "id": hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()[:12],
            "title": path.stem,
            "duration": probe_duration(path),
            "webpage_url": str(path.resolve()),
            "thumbnail": "",
            "is_local": True,
        }

    from yt_dlp import YoutubeDL

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    if settings.cookies_file and settings.cookies_file.exists():
        options["cookiefile"] = str(settings.cookies_file)

    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001 - 转换为用户可读错误
        raise DownloadError(f"无法解析视频链接：{exc}") from exc

    if info is None:
        raise DownloadError("无法解析视频链接，请检查 URL 是否正确。")

    return {
        "id": info.get("id") or hashlib.md5(url.encode("utf-8")).hexdigest()[:12],
        "title": info.get("title") or "未命名视频",
        "duration": info.get("duration") or 0,
        "webpage_url": info.get("webpage_url") or url,
        "thumbnail": info.get("thumbnail") or "",
        "is_local": False,
    }


def download_video(
    url: str,
    settings,
    progress_callback: Callable[[float, str], None] | None = None,
) -> Path:
    if is_local_video_path(url):
        return Path(url)

    settings.video_dir.mkdir(parents=True, exist_ok=True)

    from yt_dlp import YoutubeDL

    def _hook(data: dict) -> None:
        if progress_callback is None:
            return
        status = data.get("status")
        if status == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
            downloaded = data.get("downloaded_bytes") or 0
            percent = (downloaded / total) if total else 0
            progress_callback(percent, "正在下载视频")
        elif status == "finished":
            progress_callback(1.0, "视频下载完成")

    options = {
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(settings.video_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "restrictfilenames": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_hook],
    }
    if settings.cookies_file and settings.cookies_file.exists():
        options["cookiefile"] = str(settings.cookies_file)

    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:  # noqa: BLE001
        raise DownloadError(f"视频下载失败：{exc}") from exc

    video_id = (info or {}).get("id") or hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
    prepared = Path(ydl.prepare_filename(info or {}))
    return _find_video_file(settings.video_dir, prepared, video_id)

