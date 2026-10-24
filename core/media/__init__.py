"""媒体处理子包（FFmpeg / yt-dlp / PySceneDetect）。

本层只做「把视频变成可分析的素材」：下载、探测、切分、抽帧。不涉及任何模型推理。
"""

from __future__ import annotations

from core.media.scene_detect import (
    FFmpegNotFound,
    ExtractionReport,
    FrameExtractionError,
    Keyframe,
    MediaError,
    MissingDependencyError,
    ProbeError,
    ProbeInfo,
    Scene,
    SceneDetectConfig,
    SceneDetectError,
    SceneDetector,
    format_hms,
    group_by_clip,
    ms_to_timecode,
)

__all__ = [
    "FFmpegNotFound",
    "ExtractionReport",
    "FrameExtractionError",
    "Keyframe",
    "MediaError",
    "MissingDependencyError",
    "ProbeError",
    "ProbeInfo",
    "Scene",
    "SceneDetectConfig",
    "SceneDetectError",
    "SceneDetector",
    "format_hms",
    "group_by_clip",
    "ms_to_timecode",
]
