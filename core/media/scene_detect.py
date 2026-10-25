"""智能场景检测与关键帧抽取。

设计目标（对应需求文档表 1「关键帧抽取与视觉理解」/ 表 2 精度约束）::

    镜头切分   PySceneDetect(Content/Adaptive/Hash) → 失败降级 ffmpeg scene 滤镜
               → 再失败降级固定时长等分。三级链路保证任何输入都有场景结果。
    清晰度     每场景在内部等距取若干候选时间点，抽帧后用**归一化拉普拉斯方差**
               打分（先缩到固定短边 480 再算，使阈值与源分辨率解耦），
               只保留 1~2 张最清晰的帧。
    输出规格   短边 ≥720p（源本身低于 720p 时保留原分辨率并标注），
               JPEG ≤200KB（quality 二分 + 逐级降尺寸），毫秒级时间戳。

溯源铁律：每个 :class:`Keyframe` 都携带 ``video_id`` / ``keyframe_id`` /
``timestamp_ms``，且可通过 :meth:`SceneDetector.bind_clips` 绑定 ``clip_id``，
满足「任何对外输出的溯源信息必须具备 video_id、clip_id 与毫秒级时间戳」。

时间戳精度说明：抽帧使用 ``-ss`` 输入快进（ffmpeg 会精确解码到目标时间点），
误差上界为一帧（25fps → 40ms，60fps → 17ms），优于需求要求的 ≤100ms；
``Keyframe.requested_ms`` 保留请求值，``timestamp_ms`` 为按帧率吸附后的值。
"""

from __future__ import annotations

import io
import inspect
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

__all__ = [
    "MediaError",
    "FFmpegNotFound",
    "ProbeError",
    "SceneDetectError",
    "FrameExtractionError",
    "MissingDependencyError",
    "SceneDetectConfig",
    "ProbeInfo",
    "Scene",
    "Keyframe",
    "ExtractionReport",
    "SceneDetector",
    "format_hms",
    "ms_to_timecode",
]

logger = logging.getLogger("zhiying.media.scene")

#: 覆盖 ffmpeg / ffprobe 可执行文件路径的环境变量（禁止在代码里硬编码安装路径）。
ENV_FFMPEG = "ZHIYING_FFMPEG"
ENV_FFPROBE = "ZHIYING_FFPROBE"

_DEFAULT_JPEG_SUBSAMPLING = 0  # 4:4:4，中文/文字类画面更锐利


# --------------------------------------------------------------------------- #
# 异常体系
# --------------------------------------------------------------------------- #
class MediaError(RuntimeError):
    """媒体处理层的公共基类。"""


class MissingDependencyError(MediaError):
    """运行所需的第三方依赖缺失（numpy / PySceneDetect / Pillow）。"""


class FFmpegNotFound(MediaError):
    """找不到 ffmpeg 或 ffprobe 可执行文件。"""


class ProbeError(MediaError):
    """ffprobe 无法解析媒体信息。"""


class SceneDetectError(MediaError):
    """三级场景检测链路全部失败。"""


class FrameExtractionError(MediaError):
    """抽帧或关键帧编码失败。"""


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def ms_to_timecode(timestamp_ms: int) -> str:
    """毫秒时间戳 → ``HH:MM:SS.mmm``（需求文档规定的统一时间格式）。"""
    if timestamp_ms < 0:
        timestamp_ms = 0
    hours, rem = divmod(int(timestamp_ms), 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


format_hms = ms_to_timecode


def _parse_rate(rate: str | None, default: float = 25.0) -> float:
    """解析 ffprobe 的 ``r_frame_rate``（形如 ``30000/1001``）。"""
    if not rate:
        return default
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            den_f = float(den)
            if den_f <= 0:
                return default
            value = float(num) / den_f
        else:
            value = float(rate)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value) or value <= 0:
        return default
    return value


def _even(value: float, minimum: int = 2) -> int:
    """取最近的偶数（JPEG/H.264 对偶数尺寸更友好）。"""
    result = int(round(value))
    if result % 2:
        result += 1
    return max(minimum, result)


def _resolve_binary(kind: str) -> str:
    """定位 ffmpeg / ffprobe：环境变量 → PATH → 常见安装目录。"""
    env_name = ENV_FFMPEG if kind == "ffmpeg" else ENV_FFPROBE
    override = os.environ.get(env_name) or os.environ.get(env_name.replace("ZHIYING_", "ZY_"))
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return str(candidate)
        logger.warning("%s 指向的路径不存在: %s（继续在 PATH 中查找）", env_name, candidate)
    found = shutil.which(kind)
    if found:
        return found
    suffixes = (".exe", "") if os.name == "nt" else ("",)
    for base in ("/usr/bin", "/usr/local/bin", "/opt/homebrew/bin", "C:/ffmpeg/bin"):
        for suffix in suffixes:
            candidate = Path(base) / f"{kind}{suffix}"
            if candidate.is_file():
                return str(candidate)
    raise FFmpegNotFound(
        f"系统中找不到 {kind}。请先安装 FFmpeg，或通过环境变量 {env_name} 指定可执行文件路径。"
    )


# --------------------------------------------------------------------------- #
# 配置与数据模型
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SceneDetectConfig:
    """场景检测与抽帧的全部可调参数（不写死任何权重或安装路径）。"""

    # -- 场景检测 --
    detector: str = "content"  # content | adaptive | hash | ffmpeg | fixed
    threshold: float = 27.0
    min_scene_len_ms: int = 1_200
    downscale: int = 2
    frame_skip: int = 0
    fixed_scene_ms: int = 30_000  # 最终降级：固定时长等分
    merge_short_scenes: bool = True
    max_scene_ms: int = 30_000  # 超过该长度的镜头内部再细分（避免长镜头只出 1 帧）

    # -- 候选帧与清晰度 --
    frames_per_scene: int = 2  # 每个有效视觉区间最多 1~2 张
    min_frames_per_scene: int = 1
    candidate_cap: int = 6  # 单场景候选上限，控制 ffmpeg 调用次数
    blur_threshold: float = 60.0  # 归一化拉普拉斯方差阈值（480p 灰度）
    blur_probe_short_side: int = 480  # 打分前统一缩放到该短边，阈值与源分辨率解耦
    drop_blurry: bool = False  # True 时丢弃全部候选都模糊的场景

    # -- 候选去重（同一场景内别把名额浪费在长得一样的帧上）--
    #: 判据 = 64×64 灰度按 4×4 分块，取**最差**那一块的平均绝对差，同时要求平均色接近。
    #: 用"最差块"而不是全图均值：幻灯片只改一行标题时，全图均值只动 0.5，
    #: 最差块能到 5~9，两者相差 30 倍以上，阈值才好定。
    #: 单看结构指纹（aHash）必须配平均色：纯红页和纯蓝页的 aHash 完全一样。
    dedupe_frames: bool = True
    dedupe_max_diff: float = 1.5
    dedupe_color_distance: float = 30.0
    dedupe_probe_side: int = 64
    dedupe_grid: int = 4

    # -- 输出规格 --
    target_short_side: int = 1_280  # 抽帧工作分辨率（只缩不放）
    min_short_side: int = 720  # 关键帧短边下限
    max_jpeg_bytes: int = 200 * 1024
    jpeg_quality_max: int = 92
    jpeg_quality_min: int = 55
    jpeg_subsampling: int = _DEFAULT_JPEG_SUBSAMPLING
    jpeg_prefix: str = "kf"

    # -- 运行参数 --
    max_workers: int = 4
    timeout_s: float = 90.0
    ffmpeg_bin: str | None = None
    ffprobe_bin: str | None = None
    keep_probe_dir: bool = False

    # -- ffmpeg 检测档加速（长视频默认路径，见 orchestrator._scene_boost）--
    #: 用 NVDEC 硬解码（``-hwaccel cuda``）。失败自动回退软解，不中断。
    ffmpeg_hwaccel: bool = False
    #: ffmpeg scene 滤镜前先把画面缩到该宽度（0 = 不缩）。场景打分在
    #: 480px 上足够可靠，1080p 全幅解码是纯浪费。
    ffmpeg_detect_width: int = 0

    def resolved_ffmpeg(self) -> str:
        return self.ffmpeg_bin or _resolve_binary("ffmpeg")

    def resolved_ffprobe(self) -> str:
        return self.ffprobe_bin or _resolve_binary("ffprobe")

    def validate(self) -> None:
        """参数自检，尽早暴露配置错误。"""
        if self.frames_per_scene < 1:
            raise ValueError("frames_per_scene 必须 ≥ 1")
        if self.min_frames_per_scene > self.frames_per_scene:
            raise ValueError("min_frames_per_scene 不能大于 frames_per_scene")
        if self.min_scene_len_ms <= 0:
            raise ValueError("min_scene_len_ms 必须为正")
        if self.max_scene_ms <= self.min_scene_len_ms:
            raise ValueError("max_scene_ms 必须大于 min_scene_len_ms")
        if self.max_jpeg_bytes <= 0:
            raise ValueError("max_jpeg_bytes 必须为正")
        if not (0 < self.jpeg_quality_min <= self.jpeg_quality_max <= 100):
            raise ValueError("JPEG quality 需满足 0 < min <= max <= 100")
        if self.max_workers < 1:
            raise ValueError("max_workers 必须 ≥ 1")


@dataclass(frozen=True, slots=True)
class ProbeInfo:
    """ffprobe 提取的媒体基础信息。"""

    path: Path
    duration_ms: int
    width: int
    height: int
    fps: float
    video_codec: str
    has_audio: bool
    rotation: int = 0

    @property
    def display_width(self) -> int:
        """考虑旋转后的展示宽度。"""
        return self.height if abs(self.rotation) % 180 == 90 else self.width

    @property
    def display_height(self) -> int:
        return self.width if abs(self.rotation) % 180 == 90 else self.height

    @property
    def short_side(self) -> int:
        return min(self.display_width, self.display_height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "duration_ms": self.duration_ms,
            "duration": ms_to_timecode(self.duration_ms),
            "width": self.display_width,
            "height": self.display_height,
            "fps": round(self.fps, 3),
            "video_codec": self.video_codec,
            "has_audio": self.has_audio,
            "rotation": self.rotation,
        }


@dataclass(frozen=True, slots=True)
class Scene:
    """一个镜头区间（毫秒）。"""

    index: int
    start_ms: int
    end_ms: int
    detector: str = "content"

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def midpoint_ms(self) -> int:
        return self.start_ms + self.duration_ms // 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "start": ms_to_timecode(self.start_ms),
            "end": ms_to_timecode(self.end_ms),
            "detector": self.detector,
        }


@dataclass(frozen=True, slots=True)
class Keyframe:
    """一张关键帧及其溯源信息。"""

    keyframe_id: str
    video_id: str
    timestamp_ms: int
    path: Path
    width: int
    height: int
    size_bytes: int
    scene_index: int
    clip_id: str | None = None
    sharpness: float = 0.0
    sharpness_ok: bool = True
    quality_ok: bool = True
    requested_ms: int = 0
    jpeg_quality: int = 0
    notes: tuple[str, ...] = ()

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    @property
    def timestamp(self) -> str:
        return ms_to_timecode(self.timestamp_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyframe_id": self.keyframe_id,
            "video_id": self.video_id,
            "clip_id": self.clip_id,
            "timestamp_ms": self.timestamp_ms,
            "timestamp": self.timestamp,
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "size_bytes": self.size_bytes,
            "scene_index": self.scene_index,
            "sharpness": round(self.sharpness, 2),
            "sharpness_ok": self.sharpness_ok,
            "quality_ok": self.quality_ok,
            "requested_ms": self.requested_ms,
            "jpeg_quality": self.jpeg_quality,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class ExtractionReport:
    """一次抽帧任务的汇总结果（可直接落 JSON 供上层引用）。"""

    video_id: str
    video_path: Path
    probe: ProbeInfo
    scenes: list[Scene] = field(default_factory=list)
    keyframes: list[Keyframe] = field(default_factory=list)
    detector_used: str = ""
    elapsed_ms: int = 0
    candidate_count: int = 0
    degraded: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    @property
    def keyframe_count(self) -> int:
        return len(self.keyframes)

    def frames_for_scene(self, scene_index: int) -> list[Keyframe]:
        return [kf for kf in self.keyframes if kf.scene_index == scene_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_path": str(self.video_path),
            "probe": self.probe.to_dict(),
            "detector_used": self.detector_used,
            "elapsed_ms": self.elapsed_ms,
            "candidate_count": self.candidate_count,
            "scene_count": self.scene_count,
            "keyframe_count": self.keyframe_count,
            "degraded": list(self.degraded),
            "warnings": list(self.warnings),
            "scenes": [scene.to_dict() for scene in self.scenes],
            "keyframes": [kf.to_dict() for kf in self.keyframes],
        }


@dataclass(frozen=True, slots=True)
class _Candidate:
    """场景内部的一个抽帧候选点及其打分结果。"""

    requested_ms: int
    png_path: Path
    sharpness: float


# --------------------------------------------------------------------------- #
# 主类
# --------------------------------------------------------------------------- #
class SceneDetector:
    """镜头切分 + 清晰度择优 + 规格化 JPEG 输出。

    典型用法::

        detector = SceneDetector()
        report = detector.extract_keyframes(
            "data/raw/demo.mp4", video_id="demo", out_dir="data/frames/demo"
        )
        for kf in report.keyframes:
            print(kf.keyframe_id, kf.timestamp, kf.size_bytes)
    """

    def __init__(
        self,
        config: SceneDetectConfig | None = None,
        *,
        logger_: logging.Logger | None = None,
    ) -> None:
        self._cfg = config or SceneDetectConfig()
        self._cfg.validate()
        self._log = logger_ or logger
        self._ffmpeg = self._cfg.resolved_ffmpeg()
        self._ffprobe = self._cfg.resolved_ffprobe()
        self._np: Any | None = None
        self._pil_image: Any | None = None
        self._count_lock = threading.Lock()

    # -- 属性 --
    @property
    def config(self) -> SceneDetectConfig:
        return self._cfg

    # ------------------------------------------------------------------ #
    # 依赖
    # ------------------------------------------------------------------ #
    def _numpy(self) -> Any:
        if self._np is None:
            try:
                import numpy as np  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - 环境已装
                raise MissingDependencyError(
                    "清晰度评估需要 numpy: pip install numpy"
                ) from exc
            self._np = np
        return self._np

    def _pil(self) -> Any:
        if self._pil_image is None:
            try:
                from PIL import Image  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - 环境已装
                raise MissingDependencyError(
                    "关键帧编码需要 Pillow: pip install pillow"
                ) from exc
            self._pil_image = Image
        return self._pil_image

    # ------------------------------------------------------------------ #
    # 进程调用
    # ------------------------------------------------------------------ #
    def _run(self, argv: Sequence[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[bytes]:
        """执行外部命令；统一 stdout/stderr 捕获与异常包装。"""
        cmd = list(argv)
        self._log.debug("exec: %s", " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else ""))
        try:
            proc = subprocess.run(  # noqa: S603 - 参数全部由本模块拼装，无 shell
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout or self._cfg.timeout_s,
                check=False,
            )
        except FileNotFoundError as exc:
            raise FFmpegNotFound(f"无法执行 {cmd[0]}：{exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise MediaError(f"命令超时（{timeout or self._cfg.timeout_s}s）: {' '.join(cmd[:4])}") from exc
        return proc

    @staticmethod
    def _stderr_text(proc: subprocess.CompletedProcess[bytes]) -> str:
        raw = proc.stderr or b""
        for encoding in ("utf-8", "gbk", "latin-1"):
            try:
                return raw.decode(encoding, errors="replace").strip()
            except (LookupError, UnicodeDecodeError):  # pragma: no cover
                continue
        return raw.decode("utf-8", errors="replace").strip()  # pragma: no cover

    # ------------------------------------------------------------------ #
    # ffprobe
    # ------------------------------------------------------------------ #
    def probe(self, video_path: str | Path) -> ProbeInfo:
        """读取媒体基础信息；无视频流或时长缺失时抛 :class:`ProbeError`。"""
        path = Path(video_path).expanduser()
        if not path.is_file():
            raise ProbeError(f"媒体文件不存在: {path}")
        proc = self._run(
            [
                self._ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ]
        )
        if proc.returncode != 0:
            raise ProbeError(f"ffprobe 解析失败（{path}）: {self._stderr_text(proc)[:400]}")
        try:
            payload = json.loads(proc.stdout.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise ProbeError(f"ffprobe 输出不是合法 JSON（{path}）: {exc}") from exc

        streams = payload.get("streams") or []
        video_stream = None
        has_audio = False
        for stream in streams:
            codec_type = stream.get("codec_type")
            if codec_type == "audio":
                has_audio = True
            elif codec_type == "video" and video_stream is None:
                # 排除封面图（attached_pic）
                disposition = stream.get("disposition") or {}
                if disposition.get("attached_pic") in (1, "1", True):
                    continue
                video_stream = stream
        if video_stream is None:
            raise ProbeError(f"文件中没有可用的视频流: {path}")

        fmt = payload.get("format") or {}
        duration_s = 0.0
        for raw in (fmt.get("duration"), video_stream.get("duration")):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                duration_s = value
                break
        if duration_s <= 0:
            raise ProbeError(f"无法确定视频时长: {path}")

        rotation = 0
        for side in video_stream.get("side_data_list") or []:
            if "rotation" in side:
                try:
                    rotation = int(round(float(side["rotation"])))
                except (TypeError, ValueError):
                    rotation = 0
                break
        else:
            tags = video_stream.get("tags") or {}
            try:
                rotation = int(round(float(tags.get("rotate", 0))))
            except (TypeError, ValueError):
                rotation = 0

        width = int(video_stream.get("width") or 0)
        height = int(video_stream.get("height") or 0)
        if width <= 0 or height <= 0:
            raise ProbeError(f"视频分辨率无效（{width}x{height}）: {path}")

        return ProbeInfo(
            path=path,
            duration_ms=int(round(duration_s * 1000)),
            width=width,
            height=height,
            fps=_parse_rate(video_stream.get("r_frame_rate") or video_stream.get("avg_frame_rate")),
            video_codec=str(video_stream.get("codec_name") or "unknown"),
            has_audio=has_audio,
            rotation=rotation,
        )

    # ------------------------------------------------------------------ #
    # 场景检测
    # ------------------------------------------------------------------ #
    def detect_scenes(
        self,
        video_path: str | Path,
        probe: ProbeInfo | None = None,
    ) -> tuple[list[Scene], str, list[str]]:
        """三级降级链路检测镜头切点。

        返回 ``(scenes, detector_used, degraded_reasons)``。
        """
        info = probe or self.probe(video_path)
        degraded: list[str] = []

        if self._cfg.detector in {"content", "adaptive", "hash"}:
            try:
                scenes = self._detect_with_scenedetect(info, self._cfg.detector)
                normalized = self._normalize_scenes(scenes, info, self._cfg.detector)
                if normalized:
                    return normalized, f"scenedetect:{self._cfg.detector}", degraded
                degraded.append("scenedetect 未返回任何场景")
            except Exception as exc:  # noqa: BLE001 - 任何异常都降级
                degraded.append(f"scenedetect 失败: {type(exc).__name__}: {exc}")
                self._log.warning("PySceneDetect 检测失败，降级 ffmpeg: %s", exc)

        if self._cfg.detector != "fixed":
            try:
                scenes = self._detect_with_ffmpeg(info)
                normalized = self._normalize_scenes(scenes, info, "ffmpeg")
                if normalized:
                    return normalized, "ffmpeg:scene", degraded
                degraded.append("ffmpeg scene 滤镜未检测到切换点")
            except Exception as exc:  # noqa: BLE001
                degraded.append(f"ffmpeg scene 失败: {type(exc).__name__}: {exc}")
                self._log.warning("ffmpeg scene 检测失败，降级固定等分: %s", exc)

        scenes = self._fixed_scenes(info)
        degraded.append(f"降级为固定时长等分（{self._cfg.fixed_scene_ms}ms/段）")
        normalized = self._normalize_scenes(scenes, info, "fixed")
        if not normalized:  # pragma: no cover - 理论上不可能
            raise SceneDetectError(f"无法为 {info.path} 生成任何场景区间")
        return normalized, "fixed", degraded

    def _detect_with_scenedetect(self, info: ProbeInfo, detector: str) -> list[Scene]:
        """使用 PySceneDetect 检测，返回原始（未归一化）场景列表。

        需要同时兼容 scenedetect 0.6（``detect_scenes(downscale=...)``）与
        0.7+（``downscale`` 已移除、``min_scene_len`` 语义为帧数/Timecode），
        因此所有关键字参数都先按实际签名过滤再传入。
        """
        try:
            from scenedetect import open_video  # noqa: PLC0415
            from scenedetect import ContentDetector, AdaptiveDetector, HashDetector  # noqa: PLC0415
            from scenedetect import SceneManager  # noqa: PLC0415
        except ImportError as exc:
            raise MissingDependencyError("PySceneDetect 未安装: pip install scenedetect") from exc

        fps = float(info.fps or 25.0)
        # min_scene_len 在 0.6 为秒、0.7 为帧数；统一折算成帧数在各版本都安全
        min_len_frames = max(1, int(round(self._cfg.min_scene_len_ms / 1000.0 * fps)))

        def build(cls: Any, **kwargs: Any) -> Any:
            allowed = inspect.signature(cls.__init__).parameters
            return cls(**{k: v for k, v in kwargs.items() if k in allowed})

        if detector == "adaptive":
            detector_obj = build(
                AdaptiveDetector,
                adaptive_threshold=self._cfg.threshold,
                min_scene_len=min_len_frames,
            )
        elif detector == "hash":
            detector_obj = build(HashDetector, threshold=self._cfg.threshold)
        else:
            detector_obj = build(
                ContentDetector,
                threshold=self._cfg.threshold,
                min_scene_len=min_len_frames,
            )

        video = open_video(str(info.path))
        manager = SceneManager()
        manager.add_detector(detector_obj)

        detect_kwargs: dict[str, Any] = {"video": video, "show_progress": False}
        detect_params = inspect.signature(manager.detect_scenes).parameters
        if "frame_skip" in detect_params and self._cfg.frame_skip:
            detect_kwargs["frame_skip"] = self._cfg.frame_skip
        if "downscale" in detect_params and self._cfg.downscale > 1:
            detect_kwargs["downscale"] = self._cfg.downscale
        manager.detect_scenes(**detect_kwargs)

        try:
            raw = manager.get_scene_list(start_in_scene=True)
        except TypeError:  # pragma: no cover - 老版本无该关键字
            raw = manager.get_scene_list()

        scenes: list[Scene] = []
        for index, (start_tc, end_tc) in enumerate(raw):
            start_ms = int(round(start_tc.get_seconds() * 1000))
            end_ms = int(round(end_tc.get_seconds() * 1000))
            scenes.append(
                Scene(index=index, start_ms=start_ms, end_ms=end_ms, detector=f"scenedetect:{detector}")
            )
        self._log.debug(
            "PySceneDetect 原始场景 %d 段（fps=%.3f, min_len=%d 帧）", len(scenes), fps, min_len_frames
        )
        return scenes

    def _detect_with_ffmpeg(self, info: ProbeInfo) -> list[Scene]:
        """用 ffmpeg 的 ``scene`` 打分滤镜检测切点（PySceneDetect 缺失时的降级）。

        ``ffmpeg_hwaccel`` 开启时先试 NVDEC 硬解（``-hwaccel cuda``，实测
        10 倍实时），失败自动回退软解 —— 驱动/构建不支持时不能让检测
        直接掉到"固定等分"那一档。
        """
        scene_threshold = max(0.05, min(0.9, 0.35))
        if self._cfg.ffmpeg_hwaccel:
            try:
                return self._run_ffmpeg_scene(info, scene_threshold, hwaccel=True)
            except MediaError:
                self._log.warning("NVDEC 硬解不可用，回退软解继续场景检测")
        return self._run_ffmpeg_scene(info, scene_threshold, hwaccel=False)

    def _run_ffmpeg_scene(
        self, info: ProbeInfo, scene_threshold: float, *, hwaccel: bool
    ) -> list[Scene]:
        cmd: list[str] = [self._ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "info"]
        if hwaccel:
            # -hwaccel 必须在 -i 之前；解码在 GPU，滤镜/读帧仍在 CPU
            cmd += ["-hwaccel", "cuda"]
        filters: list[str] = []
        if self._cfg.ffmpeg_detect_width > 0:
            filters.append(f"scale={int(self._cfg.ffmpeg_detect_width)}:-2")
        filters.append(f"select='gt(scene\\,{scene_threshold})'")
        filters.append("metadata=print")
        cmd += [
            "-i",
            str(info.path),
            "-an",
            "-sn",
            "-filter:v",
            ",".join(filters),
            "-f",
            "null",
            "-",
        ]
        proc = self._run(cmd, timeout=max(self._cfg.timeout_s, 600.0))
        if proc.returncode != 0:
            raise MediaError(f"ffmpeg scene 滤镜失败: {self._stderr_text(proc)[:400]}")
        text = self._stderr_text(proc)
        cut_ms: list[int] = []
        # metadata=print 输出形如：frame:12 pts:12000 pts_time:12.48
        for match in re.finditer(r"pts_time:([0-9]+(?:\.[0-9]+)?)", text):
            try:
                value = float(match.group(1))
            except ValueError:  # pragma: no cover
                continue
            if math.isfinite(value) and value >= 0:
                cut_ms.append(int(round(value * 1000)))
        cut_ms = sorted({ms for ms in cut_ms if 0 < ms < info.duration_ms})
        boundaries = [0, *cut_ms, info.duration_ms]
        scenes = [
            Scene(index=i, start_ms=boundaries[i], end_ms=boundaries[i + 1], detector="ffmpeg:scene")
            for i in range(len(boundaries) - 1)
        ]
        return scenes

    def _fixed_scenes(self, info: ProbeInfo) -> list[Scene]:
        step = max(self._cfg.min_scene_len_ms, self._cfg.fixed_scene_ms)
        scenes: list[Scene] = []
        start = 0
        index = 0
        while start < info.duration_ms:
            end = min(start + step, info.duration_ms)
            if end < info.duration_ms and info.duration_ms - end < self._cfg.min_scene_len_ms and scenes:
                # 尾巴太短，并入上一段，避免产生大量碎片
                prev = scenes[-1]
                scenes[-1] = replace(prev, end_ms=info.duration_ms)
                break
            scenes.append(Scene(index=index, start_ms=start, end_ms=end, detector="fixed"))
            index += 1
            start = end
        return scenes

    def _normalize_scenes(self, scenes: Sequence[Scene], info: ProbeInfo, detector: str) -> list[Scene]:
        """裁剪到片段范围、排序去重、合并过短镜头、必要时细分超长镜头。"""
        clipped: list[Scene] = []
        for scene in sorted(scenes, key=lambda s: s.start_ms):
            start = max(0, min(scene.start_ms, info.duration_ms))
            end = max(0, min(scene.end_ms, info.duration_ms))
            if end - start <= 0:
                continue
            clipped.append(Scene(index=0, start_ms=start, end_ms=end, detector=scene.detector or detector))
        if not clipped:
            return []

        # 首尾补齐，保证完整覆盖
        if clipped[0].start_ms > 0:
            clipped[0] = replace(clipped[0], start_ms=0)
        if clipped[-1].end_ms < info.duration_ms:
            clipped[-1] = replace(clipped[-1], end_ms=info.duration_ms)

        merged: list[Scene] = []
        for scene in clipped:
            if merged and merged[-1].end_ms > scene.start_ms:
                # 真重叠（多检测器拼接或浮点误差）→ 合并；相邻但不重叠要保留
                merged[-1] = replace(merged[-1], end_ms=max(merged[-1].end_ms, scene.end_ms))
                continue
            if (
                self._cfg.merge_short_scenes
                and merged
                and scene.duration_ms < self._cfg.min_scene_len_ms
            ):
                merged[-1] = replace(merged[-1], end_ms=max(merged[-1].end_ms, scene.end_ms))
                continue
            merged.append(scene)

        # 首个镜头过短时并入下一个，避免留下 200ms 碎片
        if (
            self._cfg.merge_short_scenes
            and len(merged) > 1
            and merged[0].duration_ms < self._cfg.min_scene_len_ms
        ):
            head, second = merged[0], merged[1]
            merged[1] = replace(second, start_ms=head.start_ms)
            merged.pop(0)

        # 超长镜头内部细分，避免"长镜头只有 1 帧"丢失细节
        expanded: list[Scene] = []
        for scene in merged:
            if scene.duration_ms > self._cfg.max_scene_ms * 2:
                pieces = max(2, math.ceil(scene.duration_ms / self._cfg.max_scene_ms))
                step = scene.duration_ms / pieces
                for k in range(pieces):
                    start = int(round(scene.start_ms + k * step))
                    end = int(round(scene.start_ms + (k + 1) * step)) if k < pieces - 1 else scene.end_ms
                    expanded.append(
                        Scene(index=0, start_ms=start, end_ms=end, detector=scene.detector)
                    )
            else:
                expanded.append(scene)

        return [
            Scene(index=i, start_ms=s.start_ms, end_ms=s.end_ms, detector=s.detector)
            for i, s in enumerate(expanded)
            if s.end_ms > s.start_ms
        ]

    # ------------------------------------------------------------------ #
    # 抽帧
    # ------------------------------------------------------------------ #
    def _target_size(self, info: ProbeInfo, short_side: int) -> tuple[int, int]:
        """按短边目标计算输出尺寸（只缩不放，保持宽高比，取偶数）。"""
        width = info.display_width
        height = info.display_height
        current_short = min(width, height)
        if current_short <= short_side:
            return _even(width), _even(height)
        scale = short_side / float(current_short)
        return _even(width * scale), _even(height * scale)

    def _grab_png(self, info: ProbeInfo, timestamp_ms: int, dest: Path) -> None:
        """在指定毫秒处抽取一帧 PNG（工作分辨率）。"""
        dest.parent.mkdir(parents=True, exist_ok=True)
        width, height = self._target_size(info, self._cfg.target_short_side)
        seek_s = max(0.0, timestamp_ms / 1000.0)
        proc = self._run(
            [
                self._ffmpeg,
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-ss",
                f"{seek_s:.3f}",
                "-i",
                str(info.path),
                "-frames:v",
                "1",
                "-an",
                "-sn",
                "-vf",
                f"scale={width}:{height}:flags=lanczos",
                "-y",
                str(dest),
            ]
        )
        if proc.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
            raise FrameExtractionError(
                f"抽帧失败 @{ms_to_timecode(timestamp_ms)}（{info.path.name}）: "
                f"{self._stderr_text(proc)[:300]}"
            )

    def _sharpness(self, png_path: Path) -> float:
        """归一化拉普拉斯方差：先缩到固定短边，使阈值与源分辨率无关。"""
        np = self._numpy()
        Image = self._pil()
        with Image.open(png_path) as raw:
            gray = raw.convert("L")
            short = min(gray.size)
            target = self._cfg.blur_probe_short_side
            if short > target:
                ratio = target / float(short)
                size = (max(1, int(round(gray.width * ratio))), max(1, int(round(gray.height * ratio))))
                gray = gray.resize(size, Image.BILINEAR)
            array = np.asarray(gray, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] < 3:
            return 0.0
        # 4-邻域拉普拉斯（避免引入 scipy 依赖）
        lap = -4.0 * array
        lap[1:, :] += array[:-1, :]
        lap[:-1, :] += array[1:, :]
        lap[:, 1:] += array[:, :-1]
        lap[:, :-1] += array[:, 1:]
        return float(lap[1:-1, 1:-1].var())

    def _fingerprint(self, png_path: Path) -> tuple[Any, tuple[float, float, float]]:
        """64×64 灰度图 + 平均 RGB，作为"这张帧长什么样"的指纹。

        为什么不用 aHash：实测纯红页和纯蓝页的 8×8 aHash **完全相同**
        （都是全亮 0xffff…），只有配上平均色才能区分。而 aHash 对"只差一行标题"
        又太迟钝，所以改用分块灰度差。
        """
        np = self._numpy()
        Image = self._pil()
        side = max(8, self._cfg.dedupe_probe_side)
        with Image.open(png_path) as raw:
            gray = np.asarray(
                raw.convert("L").resize((side, side), Image.BILINEAR), dtype=np.float32
            )
            pixel = np.asarray(
                raw.convert("RGB").resize((1, 1), Image.BOX), dtype=np.float32
            ).reshape(-1)
        color = (float(pixel[0]), float(pixel[1]), float(pixel[2]))
        return gray, color

    def _is_duplicate(
        self,
        left: tuple[Any, tuple[float, float, float]],
        right: tuple[Any, tuple[float, float, float]],
    ) -> bool:
        """双判据：分块最大灰度差够小 **且** 平均色够近，才算重复。"""
        color_delta = (
            abs(left[1][0] - right[1][0])
            + abs(left[1][1] - right[1][1])
            + abs(left[1][2] - right[1][2])
        )
        if color_delta >= self._cfg.dedupe_color_distance:
            return False
        np = self._numpy()
        diff = np.abs(left[0] - right[0])
        side = diff.shape[0]
        grid = max(1, min(self._cfg.dedupe_grid, side))
        step = max(1, side // grid)
        worst = 0.0
        for row in range(0, side, step):
            for col in range(0, side, step):
                block = diff[row : row + step, col : col + step]
                if block.size:
                    value = float(block.mean())
                    if value > worst:
                        worst = value
                        if worst >= self._cfg.dedupe_max_diff:
                            return False
        return worst < self._cfg.dedupe_max_diff

    def _dedupe(
        self,
        ordered: list[_Candidate],
        limit: int,
        report: ExtractionReport,
        scene: Scene,
    ) -> list[_Candidate]:
        """按清晰度降序挑帧，跳过和已选帧视觉重复的候选。

        名额只有 1~2 张，选两张几乎一样的帧等于只覆盖一个画面。去重失败
        （PIL/numpy 不可用、图片读不出）时退回原来的"直接取前 N 个"，不拖垮抽帧。
        """
        if not self._cfg.dedupe_frames or limit <= 1 or len(ordered) <= 1:
            return ordered[:limit]
        picks: list[_Candidate] = []
        marks: list[tuple[Any, tuple[float, float, float]]] = []
        for candidate in ordered:
            if len(picks) >= limit:
                break
            try:
                mark = self._fingerprint(candidate.png_path)
            except Exception:  # noqa: BLE001 - 指纹算不出来不该让整个场景失败
                return ordered[:limit]
            if any(self._is_duplicate(mark, other) for other in marks):
                continue
            picks.append(candidate)
            marks.append(mark)
        if len(picks) < min(self._cfg.min_frames_per_scene, len(ordered)):
            # 候选全都长得一样：至少留一帧，否则时间轴上这一场景就空了
            for candidate in ordered:
                if candidate not in picks:
                    picks.append(candidate)
                if len(picks) >= self._cfg.min_frames_per_scene:
                    break
            report.warnings.append(
                f"场景 {scene.index} 候选帧视觉上高度相似，去重后仅保留 {len(picks)} 张"
            )
        return picks

    def _plan_candidates(self, scene: Scene) -> list[int]:
        """在一个场景内部等距取候选时间点（避开首尾，减少转场残留）。"""
        duration = scene.duration_ms
        if duration <= 0:
            return []
        wanted = max(self._cfg.frames_per_scene + 1, 2)
        wanted = min(self._cfg.candidate_cap, max(wanted, 3 if duration >= 4_000 else 1))
        if duration < 1_500:
            wanted = 1
        points: list[int] = []
        if wanted == 1:
            points.append(scene.midpoint_ms)
        else:
            # 内部均分：从 10% 到 90%，避开切点附近的过渡帧
            for k in range(wanted):
                ratio = 0.1 + (0.8 * k / (wanted - 1))
                points.append(int(round(scene.start_ms + duration * ratio)))
        seen: set[int] = set()
        unique: list[int] = []
        for point in points:
            clamped = max(scene.start_ms, min(point, max(scene.start_ms, scene.end_ms - 1)))
            if clamped not in seen:
                seen.add(clamped)
                unique.append(clamped)
        return unique

    def _align_to_frame(self, timestamp_ms: int, fps: float) -> int:
        """把时间戳吸附到帧网格，减少亚帧抖动。"""
        if fps <= 0:
            return int(timestamp_ms)
        frame_index = round(timestamp_ms / 1000.0 * fps)
        return int(round(frame_index / fps * 1000.0))

    # ------------------------------------------------------------------ #
    # JPEG 编码（≤ max_jpeg_bytes）
    # ------------------------------------------------------------------ #
    def _encode_jpeg(self, image: Any, quality: int) -> bytes:
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=self._cfg.jpeg_subsampling,
            optimize=True,
        )
        return buffer.getvalue()

    def _write_within_budget(self, png_path: Path, dest: Path) -> tuple[int, int, int, bool, tuple[str, ...]]:
        """把 PNG 压成 ≤max_jpeg_bytes 的 JPEG。

        返回 ``(width, height, bytes, quality_ok, notes)``。
        策略：quality 二分 → 仍超限则按 0.9 逐级降尺寸（下限 min_short_side）→
        最后兜底写最低质量并标注 ``quality_ok=False``。
        """
        Image = self._pil()
        notes: list[str] = []
        budget = self._cfg.max_jpeg_bytes
        with Image.open(png_path) as raw:
            image = raw.convert("RGB")

            def attempt(img: Any) -> tuple[bytes, int] | None:
                lo, hi = self._cfg.jpeg_quality_min, self._cfg.jpeg_quality_max
                best: tuple[bytes, int] | None = None
                data = self._encode_jpeg(img, hi)
                if len(data) <= budget:
                    return data, hi
                while lo <= hi:
                    mid = (lo + hi) // 2
                    candidate = self._encode_jpeg(img, mid)
                    if len(candidate) <= budget:
                        best = (candidate, mid)
                        lo = mid + 1
                    else:
                        hi = mid - 1
                return best

            result = attempt(image)
            scale = 1.0
            while result is None:
                short = min(image.size)
                if short <= self._cfg.min_short_side:
                    break
                scale *= 0.9
                size = (
                    max(self._cfg.min_short_side, int(round(image.width * scale))),
                    max(self._cfg.min_short_side, int(round(image.height * scale))),
                )
                if min(size) >= short:
                    break
                image = image.resize(size, Image.LANCZOS)
                notes.append(f"降到 {size[0]}x{size[1]} 以满足 {budget}B 预算")
                result = attempt(image)

            if result is None:
                data = self._encode_jpeg(image, self._cfg.jpeg_quality_min)
                dest.write_bytes(data)
                notes.append(f"最低质量仍超出预算（{len(data)}B > {budget}B）")
                return image.width, image.height, len(data), False, tuple(notes)

            data, quality = result
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            if image.width < self._cfg.min_short_side or image.height < self._cfg.min_short_side:
                notes.append(
                    f"短边 {min(image.size)}px 低于 {self._cfg.min_short_side}px 下限"
                )
            return image.width, image.height, len(data), True, tuple(notes)

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def extract_keyframes(
        self,
        video_path: str | Path,
        video_id: str,
        out_dir: str | Path,
        *,
        scenes: Sequence[Scene] | None = None,
        clip_resolver: Callable[[int], str | None] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> ExtractionReport:
        """检测场景并为每个场景导出 1~2 张关键帧。

        Args:
            video_path: 输入视频。
            video_id: 溯源 ID（必填，进入每条 Keyframe）。
            out_dir: JPEG 输出目录。
            scenes: 复用已有场景列表（避免重复检测）。
            clip_resolver: 可选回调，把毫秒时间戳映射到 ``clip_id``。
            progress: 可选回调，``(done_scenes, total_scenes)``。
        """
        started = time.monotonic()
        info = self.probe(video_path)
        output_root = Path(out_dir).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)

        if scenes is None:
            scene_list, detector_used, degraded = self.detect_scenes(info.path, probe=info)
        else:
            scene_list = list(scenes)
            detector_used = "provided"
            degraded = []
        report = ExtractionReport(
            video_id=video_id,
            video_path=info.path,
            probe=info,
            scenes=scene_list,
            detector_used=detector_used,
            degraded=degraded,
        )

        if info.short_side < self._cfg.min_short_side:
            report.warnings.append(
                f"源视频短边仅 {info.short_side}px，低于关键帧 {self._cfg.min_short_side}px 规格"
                "（不做放大以免伪造分辨率）"
            )

        probe_root = Path(tempfile.mkdtemp(prefix="zhiying-kf-"))
        # 注意：PNG 探针文件在 _extract_all 内部即被压缩成 JPEG，
        # 因此临时目录可以在全部场景处理完后安全删除。
        try:
            results = self._extract_all(
                info, scene_list, probe_root, report, progress, video_id, output_root, clip_resolver
            )
        finally:
            if self._cfg.keep_probe_dir:
                self._log.info("保留探针目录: %s", probe_root)
            else:
                shutil.rmtree(probe_root, ignore_errors=True)

        keyframes = [kf for scene_frames in results for kf in scene_frames]
        keyframes.sort(key=lambda kf: (kf.timestamp_ms, kf.scene_index))
        # 重新编号，保证 ID 与时间顺序一致（便于人工核对）。
        # ID 必须以 video_id 打头：仅用视频内序号（kf-00000）会让不同视频的关键帧
        # 撞主键，后导入的覆盖前一个视频的视觉描述，而 video_id 列仍是旧值。
        keyframes = [
            replace(kf, keyframe_id=f"{video_id}-{self._cfg.jpeg_prefix}-{i:05d}")
            for i, kf in enumerate(keyframes)
        ]
        report.keyframes = keyframes
        report.elapsed_ms = int((time.monotonic() - started) * 1000)
        self._log.info(
            "抽帧完成 video=%s scenes=%d keyframes=%d 耗时=%.2fs detector=%s",
            video_id,
            len(scene_list),
            len(keyframes),
            report.elapsed_ms / 1000.0,
            detector_used,
        )
        return report

    def _extract_all(
        self,
        info: ProbeInfo,
        scenes: Sequence[Scene],
        probe_root: Path,
        report: ExtractionReport,
        progress: Callable[[int, int], None] | None,
        video_id: str,
        out_root: Path,
        clip_resolver: Callable[[int], str | None] | None,
    ) -> list[list[Keyframe]]:
        """并行处理所有场景，返回每个场景产出的关键帧。"""
        total = len(scenes)
        results: list[list[Keyframe]] = [[] for _ in scenes]
        if not total:
            return results

        def work(item: tuple[int, Scene]) -> tuple[int, list[Keyframe]]:
            index, scene = item
            frames = self._process_scene(
                info, scene, probe_root, report, video_id, out_root, clip_resolver
            )
            return index, frames

        done = 0
        workers = max(1, min(self._cfg.max_workers, total))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kf") as pool:
            for index, frames in pool.map(work, list(enumerate(scenes))):
                results[index] = frames
                done += 1
                if progress is not None:
                    try:
                        progress(done, total)
                    except Exception:  # noqa: BLE001 - 回调异常不影响主流程
                        self._log.debug("progress 回调异常", exc_info=True)
        return results

    def _process_scene(
        self,
        info: ProbeInfo,
        scene: Scene,
        probe_root: Path,
        report: ExtractionReport,
        video_id: str,
        out_root: Path,
        clip_resolver: Callable[[int], str | None] | None,
    ) -> list[Keyframe]:
        """单场景：取候选 → 清晰度打分 → 选 top-N → 编码为预算内 JPEG。"""
        points = self._plan_candidates(scene)
        if not points:
            return []
        candidates: list[_Candidate] = []
        for seq, timestamp_ms in enumerate(points):
            png = probe_root / f"s{scene.index:05d}_{seq}_{timestamp_ms:09d}.png"
            try:
                self._grab_png(info, timestamp_ms, png)
                sharpness = self._sharpness(png)
            except MediaError as exc:
                report.warnings.append(
                    f"场景 {scene.index} 候选 {ms_to_timecode(timestamp_ms)} 抽帧失败: {exc}"
                )
                continue
            except Exception as exc:  # noqa: BLE001 - 依赖缺失等
                report.warnings.append(
                    f"场景 {scene.index} 候选 {ms_to_timecode(timestamp_ms)} 处理异常: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            with self._count_lock:
                report.candidate_count += 1
            candidates.append(_Candidate(requested_ms=timestamp_ms, png_path=png, sharpness=sharpness))

        if not candidates:
            report.warnings.append(f"场景 {scene.index} 没有任何可用候选帧")
            return []

        candidates.sort(key=lambda c: c.sharpness, reverse=True)
        sharp_enough = [c for c in candidates if c.sharpness >= self._cfg.blur_threshold]
        picks: list[_Candidate]
        if not sharp_enough:
            if self._cfg.drop_blurry:
                report.warnings.append(
                    f"场景 {scene.index} 全部候选低于清晰度阈值 "
                    f"{self._cfg.blur_threshold}（最高 {candidates[0].sharpness:.1f}），按配置丢弃"
                )
                return []
            # 关键降级：整段模糊也要留证据帧，否则时间轴上会缺一块
            picks = candidates[: max(1, self._cfg.min_frames_per_scene)]
            report.warnings.append(
                f"场景 {scene.index} 全部候选偏模糊（最高 {candidates[0].sharpness:.1f} < "
                f"{self._cfg.blur_threshold}），保留最优帧作为证据"
            )
        else:
            limit = max(
                self._cfg.min_frames_per_scene, min(self._cfg.frames_per_scene, len(sharp_enough))
            )
            picks = self._dedupe(sharp_enough, limit, report, scene)

        # 避免同一场景两张帧时间上过于接近（<200ms 视为同一视觉信息）
        if len(picks) > 1 and abs(picks[1].requested_ms - picks[0].requested_ms) < 200:
            picks = picks[:1]
        picks.sort(key=lambda c: c.requested_ms)

        frames: list[Keyframe] = []
        for offset, candidate in enumerate(picks):
            frame = self._encode_candidate(
                info, scene, candidate, offset, video_id, out_root, clip_resolver, report
            )
            if frame is not None:
                frames.append(frame)
        return frames

    def _encode_candidate(
        self,
        info: ProbeInfo,
        scene: Scene,
        candidate: _Candidate,
        offset: int,
        video_id: str,
        out_root: Path,
        clip_resolver: Callable[[int], str | None] | None,
        report: ExtractionReport,
    ) -> Keyframe | None:
        """把一个候选 PNG 压成预算内 JPEG 并组装 :class:`Keyframe`。"""
        filename = (
            f"{self._cfg.jpeg_prefix}_{scene.index:05d}_{offset}_{candidate.requested_ms:09d}.jpg"
        )
        dest = out_root / filename
        try:
            width, height, size, quality_ok, notes = self._write_within_budget(
                candidate.png_path, dest
            )
        except Exception as exc:  # noqa: BLE001 - 单帧失败不影响整体
            report.warnings.append(
                f"场景 {scene.index} 关键帧编码失败 @{ms_to_timecode(candidate.requested_ms)}: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

        timestamp_ms = self._align_to_frame(candidate.requested_ms, info.fps)
        note_list = list(notes)
        if timestamp_ms != candidate.requested_ms:
            note_list.append(
                f"时间戳按 {info.fps:.2f}fps 吸附：{candidate.requested_ms} → {timestamp_ms}ms"
            )
        clip_id = None
        if clip_resolver is not None:
            try:
                clip_id = clip_resolver(timestamp_ms)
            except Exception:  # noqa: BLE001 - 回调异常不应丢帧
                self._log.debug("clip_resolver 异常", exc_info=True)
        return Keyframe(
            keyframe_id=f"{video_id}-{self._cfg.jpeg_prefix}-s{scene.index:05d}-{offset}",
            video_id=video_id,
            timestamp_ms=timestamp_ms,
            path=dest,
            width=width,
            height=height,
            size_bytes=size,
            scene_index=scene.index,
            clip_id=clip_id,
            sharpness=candidate.sharpness,
            sharpness_ok=candidate.sharpness >= self._cfg.blur_threshold,
            quality_ok=quality_ok,
            requested_ms=candidate.requested_ms,
            notes=tuple(note_list),
        )

    # ------------------------------------------------------------------ #
    # 与切片绑定
    # ------------------------------------------------------------------ #
    @staticmethod
    def bind_clips(
        keyframes: Iterable[Keyframe],
        clips: Iterable[tuple[str, int, int]],
        *,
        prefer_core: bool = True,
    ) -> list[Keyframe]:
        """把关键帧绑定到时间区间覆盖它的切片。

        Args:
            keyframes: 关键帧列表。
            clips: ``(clip_id, start_ms, end_ms)`` 序列；传入的核心区间越精确越好。
            prefer_core: 多个区间重叠时，优先匹配「区间最短」的那个（通常是核心区间）。
        """
        intervals = [(cid, int(start), int(end)) for cid, start, end in clips if end > start]
        bound: list[Keyframe] = []
        for keyframe in keyframes:
            hits = [
                (end - start, cid)
                for cid, start, end in intervals
                if start <= keyframe.timestamp_ms < end
            ]
            if not hits:
                bound.append(keyframe)
                continue
            hits.sort(key=lambda item: (item[0] if prefer_core else -item[0], item[1]))
            bound.append(replace(keyframe, clip_id=hits[0][1]))
        return bound


def group_by_clip(keyframes: Iterable[Keyframe]) -> dict[str, list[str]]:
    """按 ``clip_id`` 聚合关键帧 ID，供切片对象写入 ``keyframe_ids``。"""
    grouped: dict[str, list[str]] = {}
    for keyframe in keyframes:
        if not keyframe.clip_id:
            continue
        grouped.setdefault(keyframe.clip_id, []).append(keyframe.keyframe_id)
    for ids in grouped.values():
        ids.sort()
    return grouped
