"""媒体层异常体系。

单独成文件是为了让 :mod:`core.media.ffmpeg_utils`、:mod:`core.media.parser`、
:mod:`core.media.downloader` 与 :mod:`core.media.scene_detect` 共享同一套
异常根，而不产生循环 import：

    errors.py  ←  ffmpeg_utils.py  ←  parser.py / scene_detect.py
"""

from __future__ import annotations

__all__ = [
    "DownloadError",
    "FFmpegError",
    "FFmpegNotFound",
    "MediaError",
    "ProbeError",
    "UnsupportedMediaError",
]


class MediaError(RuntimeError):
    """媒体层所有异常的基类，便于上层一次性兜底。"""


class FFmpegNotFound(MediaError):
    """找不到 ffmpeg / ffprobe 可执行文件。"""


class FFmpegError(MediaError):
    """ffmpeg / ffprobe 进程以非零码退出。

    属性
    ----
    command: 实际执行的 argv（首元素为可执行文件路径）
    returncode: 进程退出码
    stderr: stderr 尾部若干字符，便于直接定位"参数写错"还是"文件损坏"
    """

    def __init__(
        self,
        message: str,
        *,
        command: list[str] | None = None,
        returncode: int | None = None,
        stderr: str = "",
    ) -> None:
        self.command = list(command or [])
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(message)


class ProbeError(MediaError):
    """ffprobe 无法解析媒体元数据（文件损坏 / 非媒体文件 / 缺少流）。"""


class UnsupportedMediaError(MediaError):
    """媒体格式或编码不受支持。"""


class DownloadError(MediaError):
    """下载失败（网络错误、404、防盗链、需要登录等）。"""
