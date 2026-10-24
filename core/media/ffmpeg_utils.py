"""FFmpeg / FFprobe 统一封装。

媒体层所有 ffmpeg 调用都走这里，避免 ``parser.py``、``scene_detect.py``、
``downloader.py`` 各写一套参数拼装与错误解析。

职责边界
--------
* **只负责"调对命令"**：可执行文件定位、超时、stderr 归一、元数据解析。
* **不负责业务决策**：抽几帧、切多长、压到多大，都由调用方决定。

可执行文件定位顺序
------------------
1. 环境变量 ``ZHIYING_FFMPEG`` / ``ZHIYING_FFPROBE``（完整文件路径）
2. 环境变量 ``ZHIYING_FFMPEG_DIR``（bin 目录，取其中 ffmpeg/ffprobe）
3. ``shutil.which`` —— 常规 PATH 查找
4. ``imageio_ffmpeg.get_ffmpeg_exe()``（若安装了 imageio-ffmpeg）
5. 抛出 :class:`~core.media.errors.FFmpegNotFound`

典型用法::

    from core.media.ffmpeg_utils import probe_media, run_ffmpeg, extract_audio

    info = probe_media("demo.mp4")
    print(info.duration_ms, info.width, info.height, info.fps)

    extract_audio("demo.mp4", "demo.wav")            # 16kHz 单声道 pcm_s16le
"""

from __future__ import annotations

import functools
import json
import logging
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.media.errors import FFmpegError, FFmpegNotFound, ProbeError

logger = logging.getLogger("zhiying.media.ffmpeg")

__all__ = [
    "DEFAULT_FFMPEG_TIMEOUT",
    "DEFAULT_PROBE_TIMEOUT",
    "MediaInfo",
    "extract_audio",
    "frame_at",
    "media_summary",
    "parse_rate",
    "probe_media",
    "resolve_binary",
    "run_ffmpeg",
    "stderr_tail",
]

DEFAULT_FFMPEG_TIMEOUT: float = 1_800.0
"""单个 ffmpeg 进程的默认超时（秒）。1 小时视频转码留足余量。"""

DEFAULT_PROBE_TIMEOUT: float = 30.0
"""ffprobe 的默认超时（秒）。远端/损坏文件可能卡住，必须有上限。"""

_ENV_FFMPEG = "ZHIYING_FFMPEG"
_ENV_FFPROBE = "ZHIYING_FFPROBE"
_ENV_DIR = "ZHIYING_FFMPEG_DIR"
_ENV_HINT = "ZHIYING_FFMPEG_DIR"
_BINARY_NAMES: Mapping[str, str] = {"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"}


# --------------------------------------------------------------------------- #
# 可执行文件定位
# --------------------------------------------------------------------------- #
def _candidate_from_dir(directory: str | Path, kind: str) -> Path | None:
    base = Path(os.path.expanduser(str(directory)))
    if not base.is_dir():
        return None
    stem = _BINARY_NAMES[kind]
    for name in (stem, f"{stem}.exe"):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def _scan_path_dirs(kind: str) -> Path | None:
    """在 PATH 各目录里精确匹配（含 .exe），不依赖 PATHEXT。"""
    stem = _BINARY_NAMES[kind]
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        found = _candidate_from_dir(entry, kind)
        if found is not None:
            return found
    return None


@functools.lru_cache(maxsize=8)
def resolve_binary(kind: str) -> str:
    """返回 ``ffmpeg`` / ``ffprobe`` 的可执行文件路径。

    :raises FFmpegNotFound: 所有定位手段都失败
    """
    if kind not in _BINARY_NAMES:
        raise ValueError(f"kind 只能是 ffmpeg 或 ffprobe，收到 {kind!r}")

    explicit = os.environ.get(_ENV_FFMPEG if kind == "ffmpeg" else _ENV_FFPROBE, "").strip()
    if explicit:
        path = Path(os.path.expanduser(explicit))
        if path.is_file():
            logger.debug("%s 来自环境变量: %s", kind, path)
            return str(path)
        logger.warning("环境变量指向的 %s 不存在: %s，继续自动探测", kind, path)

    hint_dir = os.environ.get(_ENV_DIR, "").strip()
    if hint_dir:
        found = _candidate_from_dir(hint_dir, kind)
        if found is not None:
            logger.debug("%s 来自 %s=%s", kind, _ENV_DIR, found)
            return str(found)
        logger.warning("%s 目录下找不到 %s: %s", _ENV_DIR, kind, hint_dir)

    which = shutil.which(_BINARY_NAMES[kind])
    if which:
        return which

    scanned = _scan_path_dirs(kind)
    if scanned is not None:
        return str(scanned)

    if kind == "ffmpeg":
        try:  # imageio-ffmpeg 自带静态编译版本，常见于 notebook 环境
            import imageio_ffmpeg  # type: ignore[import-not-found]

            bundled = imageio_ffmpeg.get_ffmpeg_exe()
            if bundled and Path(bundled).is_file():
                logger.info("使用 imageio-ffmpeg 自带的 ffmpeg: %s", bundled)
                return str(bundled)
        except Exception:  # noqa: BLE001 - 可选依赖，任何失败都只是"没有"
            pass

    raise FFmpegNotFound(
        f"找不到 {kind}。请安装 FFmpeg 并加入 PATH，"
        f"或设置环境变量 {_ENV_FFMPEG if kind == 'ffmpeg' else _ENV_FFPROBE}"
        f"（完整路径）或 {_ENV_HINT}（bin 目录）。"
        f"当前 PATH 首段: {os.environ.get('PATH', '')[:160]}"
    )


def reset_binary_cache() -> None:
    """清空可执行文件定位缓存（改了环境变量后调用）。"""
    resolve_binary.cache_clear()


# --------------------------------------------------------------------------- #
# 进程执行
# --------------------------------------------------------------------------- #
def stderr_tail(proc: subprocess.CompletedProcess[bytes], limit: int = 1_200) -> str:
    """把 stderr 字节流解成可读文本的尾部。"""
    raw = proc.stderr or b""
    if isinstance(raw, str):  # 理论上不会，但兼容 text=True 的调用
        text = raw
    else:
        text = raw.decode("utf-8", errors="replace")
    lines = [line for line in text.strip().splitlines() if line.strip()]
    joined = "\n".join(lines)
    return joined[-limit:] if len(joined) > limit else joined


def run_ffmpeg(
    argv: Sequence[str],
    *,
    timeout: float | None = DEFAULT_FFMPEG_TIMEOUT,
    check: bool = True,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """执行一条 ffmpeg/ffprobe 命令（argv 首元素必须是完整路径）。

    :param check: ``True`` 时非零退出码抛 :class:`FFmpegError`
    :raises FFmpegError: 非零退出、超时、可执行文件缺失
    """
    if not argv:
        raise ValueError("argv 不能为空")
    command = [str(part) for part in argv]
    logger.debug("执行: %s", " ".join(command[:12]) + (" ..." if len(command) > 12 else ""))
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            timeout=timeout,
            cwd=str(cwd) if cwd is not None else None,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(
            f"{Path(command[0]).name} 执行超时（{timeout}s）: {' '.join(command[:6])} ...",
            command=command,
        ) from exc
    except FileNotFoundError as exc:
        raise FFmpegNotFound(f"可执行文件不存在: {command[0]}") from exc
    except OSError as exc:
        raise FFmpegError(f"启动 {command[0]} 失败: {exc}", command=command) from exc

    if proc.returncode != 0 and check:
        raise FFmpegError(
            f"{Path(command[0]).name} 退出码 {proc.returncode}: {stderr_tail(proc, 600)}",
            command=command,
            returncode=proc.returncode,
            stderr=stderr_tail(proc),
        )
    return proc


# --------------------------------------------------------------------------- #
# 元数据解析
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MediaInfo:
    """ffprobe 解析出的媒体元数据（不含业务判断）。"""

    path: Path
    duration_ms: int
    width: int
    height: int
    fps: float
    has_video: bool
    has_audio: bool
    format_name: str = ""
    rotation: int = 0
    nb_frames: int = 0
    sample_rate: int = 0
    channels: int = 0
    bit_rate: int = 0
    video_codec: str = ""
    audio_codec: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def display_width(self) -> int:
        """按旋转角换算后的显示宽度（手机竖拍视频宽高互换）。"""
        return self.height if self.rotation % 180 else self.width

    @property
    def display_height(self) -> int:
        return self.width if self.rotation % 180 else self.height

    @property
    def short_side(self) -> int:
        return min(self.display_width, self.display_height)

    @property
    def long_side(self) -> int:
        return max(self.display_width, self.display_height)

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "duration_ms": self.duration_ms,
            "width": self.display_width,
            "height": self.display_height,
            "coded_width": self.width,
            "coded_height": self.height,
            "fps": round(self.fps, 3),
            "rotation": self.rotation,
            "has_video": self.has_video,
            "has_audio": self.has_audio,
            "nb_frames": self.nb_frames,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_rate": self.bit_rate,
            "format_name": self.format_name,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
        }


def parse_rate(text: str | None, default: float = 25.0) -> float:
    """解析 ``"30000/1001"`` 这类帧率表达，非法时返回 ``default``。"""
    if not text:
        return default
    raw = str(text).strip()
    if not raw or raw in {"0/0", "N/A"}:
        return default
    try:
        if "/" in raw:
            numerator, _, denominator = raw.partition("/")
            den = float(denominator)
            if den == 0:
                return default
            value = float(numerator) / den
        else:
            value = float(raw)
    except (TypeError, ValueError):
        return default
    if value <= 0 or value > 1_000:
        return default
    return value


def _ms(value: Any) -> int:
    """秒（可能为字符串/None）→ 毫秒。"""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return 0
    if seconds <= 0:
        return 0
    return int(round(seconds * 1000))


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rotation_of(stream: Mapping[str, Any]) -> int:
    """从 stream 推断旋转角（ffprobe 有 ``side_data_list`` 与 ``tags.rotate`` 两种写法）。"""
    for side in stream.get("side_data_list") or []:
        if isinstance(side, Mapping) and "rotation" in side:
            try:
                return int(round(float(side["rotation"]))) % 360
            except (TypeError, ValueError):
                continue
    tags = stream.get("tags") or {}
    if isinstance(tags, Mapping):
        for key in ("rotate", "rotation"):
            if key in tags:
                try:
                    return int(round(float(tags[key]))) % 360
                except (TypeError, ValueError):
                    continue
    # displaymatrix 可能在 stream 顶层
    if "rotation" in stream:
        try:
            return int(round(float(stream["rotation"]))) % 360
        except (TypeError, ValueError):
            pass
    return 0


def probe_media(
    path: str | Path,
    *,
    ffprobe: str | None = None,
    timeout: float | None = DEFAULT_PROBE_TIMEOUT,
) -> MediaInfo:
    """用 ffprobe 读取媒体元数据。

    :raises ProbeError: 文件不存在、不是媒体、或缺视频流
    """
    media_path = Path(os.path.expanduser(str(path)))
    if not media_path.is_file():
        raise ProbeError(f"媒体文件不存在: {media_path}")

    binary = ffprobe or resolve_binary("ffprobe")
    argv = [
        binary,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-i", str(media_path),
    ]
    proc = run_ffmpeg(argv, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise ProbeError(
            f"ffprobe 无法解析 {media_path.name}: {stderr_tail(proc, 600)}"
        )
    try:
        payload = json.loads(proc.stdout.decode("utf-8", errors="replace") or "{}")
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe 输出不是合法 JSON: {exc}") from exc

    streams: list[Mapping[str, Any]] = [
        s for s in (payload.get("streams") or []) if isinstance(s, Mapping)
    ]
    video = next(
        (s for s in streams
         if s.get("codec_type") == "video" and (s.get("disposition") or {}).get("attached_pic") != 1),
        None,
    )
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt: Mapping[str, Any] = payload.get("format") or {}

    if video is None:
        raise ProbeError(f"{media_path.name} 不含视频流（纯音频或纯图片序列都不支持）")

    duration_ms = _ms(fmt.get("duration")) or _ms(video.get("duration"))
    if duration_ms <= 0:
        raise ProbeError(
            f"{media_path.name} 时长未知（duration=0），可能是流式容器或文件截断"
        )

    fps = parse_rate(video.get("avg_frame_rate"), 0.0) or parse_rate(video.get("r_frame_rate"), 25.0)
    width = _int(video.get("width"))
    height = _int(video.get("height"))
    if width <= 0 or height <= 0:
        raise ProbeError(f"{media_path.name} 分辨率非法: {width}x{height}")

    rotation = _rotation_of(video)
    nb_frames = _int(video.get("nb_frames"))

    return MediaInfo(
        path=media_path,
        duration_ms=duration_ms,
        width=width,
        height=height,
        fps=fps,
        has_video=True,
        has_audio=audio is not None,
        format_name=str(fmt.get("format_name") or ""),
        rotation=rotation,
        nb_frames=nb_frames,
        sample_rate=_int((audio or {}).get("sample_rate")),
        channels=_int((audio or {}).get("channels")),
        bit_rate=_int(fmt.get("bit_rate")),
        video_codec=str(video.get("codec_name") or ""),
        audio_codec=str((audio or {}).get("codec_name") or ""),
        raw=payload,
    )


def media_summary(info: MediaInfo) -> str:
    """一行人类可读摘要，用于日志。"""
    return (
        f"{info.path.name} | {info.duration_ms / 1000:.1f}s | "
        f"{info.display_width}x{info.display_height}@{info.fps:.2f}fps | "
        f"v={info.video_codec or '-'} a={info.audio_codec or '-'}"
        f"{'' if info.has_audio else '(无音轨)'}"
        f"{f' rot={info.rotation}' if info.rotation else ''}"
    )


# --------------------------------------------------------------------------- #
# 常用操作
# --------------------------------------------------------------------------- #
def extract_audio(
    source: str | Path,
    dest: str | Path,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    codec: str = "pcm_s16le",
    start_ms: int = 0,
    duration_ms: int | None = None,
    ffmpeg: str | None = None,
    timeout: float | None = DEFAULT_FFMPEG_TIMEOUT,
    overwrite: bool = True,
) -> Path:
    """抽取/转码音轨（默认 16kHz 单声道 PCM，faster-whisper 的标准输入）。

    :raises FFmpegError: 源文件无音轨或转码失败
    """
    src = Path(source)
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    binary = ffmpeg or resolve_binary("ffmpeg")

    argv: list[str] = [
        binary,
        "-hide_banner", "-loglevel", "error",
        "-nostdin",
        "-y" if overwrite else "-n",
    ]
    if start_ms > 0:
        argv += ["-ss", f"{start_ms / 1000:.3f}"]
    argv += ["-i", str(src)]
    if duration_ms is not None and duration_ms > 0:
        argv += ["-t", f"{duration_ms / 1000:.3f}"]
    argv += [
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-c:a", codec,
        str(out),
    ]
    proc = run_ffmpeg(argv, timeout=timeout, check=False)
    if proc.returncode != 0:
        detail = stderr_tail(proc, 600)
        if "does not contain any stream" in detail or "Stream map" in detail:
            raise FFmpegError(f"{src.name} 不含音轨，无法转写", command=argv,
                              returncode=proc.returncode, stderr=detail)
        raise FFmpegError(f"音频抽取失败: {detail}", command=argv,
                          returncode=proc.returncode, stderr=detail)
    if not out.is_file() or out.stat().st_size == 0:
        raise FFmpegError(f"音频抽取产物为空: {out}", command=argv)
    logger.debug("音频已导出: %s (%.1f KB)", out.name, out.stat().st_size / 1024)
    return out


def frame_at(
    source: str | Path,
    dest: str | Path,
    *,
    timestamp_ms: int,
    scale: tuple[int, int] | None = None,
    codec: str = "png",
    extra_filters: Iterable[str] = (),
    ffmpeg: str | None = None,
    timeout: float | None = 120.0,
    overwrite: bool = True,
) -> Path:
    """在指定毫秒位置抽一帧。

    ``-ss`` 放在 ``-i`` 之前（输入侧定位），对大文件远快于输出侧定位。
    """
    if timestamp_ms < 0:
        raise ValueError(f"timestamp_ms 不能为负: {timestamp_ms}")
    src = Path(source)
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    binary = ffmpeg or resolve_binary("ffmpeg")

    filters = list(extra_filters)
    if scale is not None:
        width, height = scale
        filters.append(f"scale={max(2, width)}:{max(2, height)}")

    argv: list[str] = [
        binary,
        "-hide_banner", "-loglevel", "error",
        "-nostdin",
        "-y" if overwrite else "-n",
        "-ss", f"{timestamp_ms / 1000:.3f}",
        "-i", str(src),
    ]
    if filters:
        argv += ["-vf", ",".join(filters)]
    argv += ["-frames:v", "1", "-c:v", codec, str(out)]
    proc = run_ffmpeg(argv, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise FFmpegError(f"抽帧失败 @{timestamp_ms}ms: {stderr_tail(proc, 400)}",
                          command=argv, returncode=proc.returncode, stderr=stderr_tail(proc))
    if not out.is_file() or out.stat().st_size == 0:
        raise FFmpegError(f"抽帧产物为空 @{timestamp_ms}ms: {out}", command=argv)
    return out


# 某些环境下多线程并发调用 ffmpeg 会争抢 GPU 解码器，这里给调用方一个集中开关
_FFMPEG_SEMAPHORE: threading.Semaphore | None = None


def set_concurrency(limit: int) -> None:
    """限制本进程内并发 ffmpeg 进程数（``0`` 表示取消限制）。"""
    global _FFMPEG_SEMAPHORE
    _FFMPEG_SEMAPHORE = threading.Semaphore(limit) if limit > 0 else None


def get_concurrency_guard() -> threading.Semaphore | None:
    """返回当前的并发限流闸门，供调用方在 spawn 前 ``acquire()``。"""
    return _FFMPEG_SEMAPHORE
