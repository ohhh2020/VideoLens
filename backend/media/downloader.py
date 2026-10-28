"""远程视频下载（yt-dlp 封装）。

需求依据
--------
* yt-dlp 失败重试 3 次，指数退避 + 断点续传（表4 故障处理）
* 云端 URL ≤2048 字符；下载后可探测时长，超 2h 拒收（表 输入约束）
* 优先复用已下载文件：同 ``video_id`` 目录下已有完整产物则直接返回

设计要点
--------
1. **进度可观测**：``progress_hooks`` 转成 :class:`DownloadProgress` 回调，
   统一 ``downloaded_bytes / total_bytes / percent / speed / eta`` 四个字段，
   Web 层与 CLI 都消费同一结构。
2. **断点续传**：``continuedl=True`` + 保留 ``.part``；重试时不清目录，
   因此第二次尝试通常只补差距。``DownloadResult.resumed`` 标记是否为续传。
3. **失败分类**：``_FATAL_MARKERS`` 命中（404/私有/需要会员/地区限制/无可用格式）
   立即放弃；其余（网络抖动、超时、429）按指数退避重试。
4. **产物布局**：``<videos_dir>/<video_id>/source.<ext>`` + 同名 ``.info.json``
   与字幕 ``source.<lang>.<ext>``。目录名用 ``video_id`` 而非标题，
   避免中文/特殊字符在 Windows 上炸路径，也保证幂等复用。
5. **不做格式转码**：下载下来就是原容器（mp4/webm/mkv），转码交给
   ``ffmpeg_utils.extract_audio`` / ``scene_detect``，职责不重叠。
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Final, Mapping

from backend.runtime import runtime_path
from backend.media.errors import DownloadError, MediaError
from backend.media.ffmpeg_utils import MediaInfo, probe_media, resolve_binary
from backend.media.parser import MediaSource, Platform, SourceKind, sanitize_filename

logger = logging.getLogger("zhiying.media.downloader")

__all__ = [
    "DownloadConfig",
    "DownloadProgress",
    "DownloadResult",
    "DownloaderUnavailable",
    "MediaDownloader",
    "ProgressCallback",
    "SUBTITLE_FORMATS",
]

SUBTITLE_FORMATS: Final[tuple[str, ...]] = ("vtt", "srt", "ass", "ssa", "json3")

ProgressCallback = Callable[["DownloadProgress"], None]
"""``(progress) -> None``；调用发生在 yt-dlp 的 hook 线程里，应保持轻量。"""

_FATAL_MARKERS: Final[tuple[str, ...]] = (
    "unsupported url",
    "is not a valid url",
    "video unavailable",
    "this video is private",
    "private video",
    "removed by the uploader",
    "account has been terminated",
    "http error 404",
    "http error 403",
    "http error 410",
    "sign in to confirm your age",
    "confirm your age",
    "members-only",
    "join this channel",
    "this video is only available for",
    "not available in your country",
    "geo restricted",
    "requested format is not available",
    "no video formats found",
    "playlist does not exist",
)

_RETRYABLE_MARKERS: Final[tuple[str, ...]] = (
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "remote end closed",
    "temporary failure",
    "temporarily unavailable",
    "http error 429",
    "http error 500",
    "http error 502",
    "http error 503",
    "http error 504",
    "fragment",
    "unable to download",
    "incompleteread",
)


class DownloaderUnavailable(MediaError):
    """yt-dlp 未安装或版本过旧，无法执行下载。"""


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class DownloadConfig:
    """下载行为配置（全部可由 ``configs/app_config.yaml`` 注入）。"""

    videos_dir: Path = field(default_factory=lambda: runtime_path("data/videos"))
    temp_dir: Path | None = None
    max_retries: int = 3
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    socket_timeout_s: float = 30.0
    retries_per_fragment: int = 10
    concurrent_fragments: int = 4
    resume: bool = True
    cookies_from_browser: str | None = None
    cookie_file: Path | None = None
    max_file_bytes: int | None = 4 * 1024**3
    max_duration_ms: int | None = 7_200_000
    format_selector: str = "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*[height<=1080]+ba/b[height<=1080]/b"
    merge_output_format: str = "mp4"
    subtitle_langs: tuple[str, ...] = ("zh-Hans", "zh-CN", "zh", "en")
    subtitle_formats: tuple[str, ...] = ("vtt", "srt", "ass")
    write_subtitles: bool = True
    write_info_json: bool = True
    quiet: bool = True
    force: bool = False

    def validate(self) -> None:
        if self.max_retries < 0:
            raise ValueError(f"max_retries 不能为负: {self.max_retries}")
        if self.backoff_base < 1.0:
            raise ValueError(f"backoff_base 应 >= 1.0: {self.backoff_base}")
        if self.backoff_max < self.backoff_base:
            raise ValueError("backoff_max 应 >= backoff_base")
        if self.socket_timeout_s <= 0:
            raise ValueError(f"socket_timeout_s 应为正: {self.socket_timeout_s}")
        if self.concurrent_fragments < 1:
            raise ValueError("concurrent_fragments 应 >= 1")
        for fmt in self.subtitle_formats:
            if fmt not in SUBTITLE_FORMATS:
                raise ValueError(f"不支持的字幕格式 {fmt!r}；可选 {SUBTITLE_FORMATS}")

    @classmethod
    def from_config(cls, config: Any) -> DownloadConfig:
        """从 :class:`backend.config.AppConfig` 构造。

        路径必须走 ``cfg.path()`` 拿**绝对路径**。直接取 ``paths.videos`` 的
        相对字符串时，yt-dlp 会把相对的 ``data/videos`` 解析到 ``paths`` 基准
        目录之下，产出 ``data\\temp\\data\\videos\\...`` 这种双重路径 ——
        实测触发 FileNotFoundError 与四轮退避重试，纯浪费一两分钟。
        """
        media = getattr(config, "media", None)
        download = getattr(media, "download", None)
        paths = getattr(config, "paths", None)
        resolve = getattr(config, "path", None)
        if callable(resolve):
            videos = Path(resolve("videos"))
            temp = Path(resolve("temp"))
        else:  # 裸配置对象（部分单测）：退回字符串并按项目根解析
            root = Path(getattr(config, "root_dir", Path.cwd()))
            videos = (root / str(getattr(paths, "videos", "data/videos"))).resolve()
            temp = (root / str(getattr(paths, "temp", "data/temp"))).resolve()
        limits = getattr(config, "limits", None)
        kwargs: dict[str, Any] = {
            "videos_dir": videos,
            "temp_dir": temp,
            "max_file_bytes": getattr(limits, "max_file_bytes", None),
            "max_duration_ms": getattr(limits, "max_duration_ms", None),
        }
        # 下载源分辨率上限（默认 720p）：分析链路真实需求只有关键帧 720 短边
        # 与视觉模型 448px，1080p 源只是白付两三倍下载时间。
        max_h = getattr(download, "format_max_height", None)
        if max_h:
            h = max(144, int(max_h))
            kwargs["format_selector"] = (
                f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/"
                f"bv*[height<={h}]+ba/b[height<={h}]/b"
            )
        for src, dst in (
            ("max_retries", "max_retries"),
            ("backoff_base", "backoff_base"),
            ("backoff_max", "backoff_max"),
            ("socket_timeout_s", "socket_timeout_s"),
            ("retries_per_fragment", "retries_per_fragment"),
            ("concurrent_fragments", "concurrent_fragments"),
            ("resume", "resume"),
            ("cookies_from_browser", "cookies_from_browser"),
        ):
            value = getattr(download, src, None)
            if value is not None:
                kwargs[dst] = value
        return cls(**kwargs)


# --------------------------------------------------------------------------- #
# 进度 / 结果
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class DownloadProgress:
    """一次进度事件（不可变，可安全跨线程传递）。"""

    video_id: str
    status: str
    """``downloading`` / ``finished`` / ``error`` / ``merging`` / ``postprocess``"""
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    speed_bps: float | None = None
    eta_s: float | None = None
    filename: str = ""
    attempt: int = 1

    @property
    def percent(self) -> float:
        if not self.total_bytes:
            return 0.0
        return min(100.0, self.downloaded_bytes * 100.0 / self.total_bytes)

    @property
    def human(self) -> str:
        total = (
            f"{self.total_bytes / 1024**2:.1f} MB" if self.total_bytes else "未知大小"
        )
        speed = f"{self.speed_bps / 1024**2:.1f} MB/s" if self.speed_bps else "-"
        return (
            f"{self.percent:5.1f}%  {self.downloaded_bytes / 1024**2:.1f}/{total}  "
            f"{speed}  剩余 {self.eta_s:.0f}s"
            if self.eta_s
            else f"{self.percent:5.1f}%  {self.downloaded_bytes / 1024**2:.1f}/{total}  {speed}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "status": self.status,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "percent": round(self.percent, 2),
            "speed_bps": self.speed_bps,
            "eta_s": self.eta_s,
            "filename": self.filename,
            "attempt": self.attempt,
        }


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """下载产物描述，直接喂给 ``probe`` / ``extract_audio`` / ``scene_detect``。"""

    video_id: str
    path: Path
    title: str = ""
    uploader: str = ""
    webpage_url: str = ""
    platform: Platform = Platform.GENERIC
    duration_ms: int | None = None
    size_bytes: int = 0
    format_id: str = ""
    ext: str = ""
    attempts: int = 1
    resumed: bool = False
    elapsed_ms: int = 0
    subtitles: tuple[Path, ...] = ()
    info_json: Path | None = None
    warnings: tuple[str, ...] = ()
    info: MediaInfo | None = None

    @property
    def has_subtitles(self) -> bool:
        return bool(self.subtitles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "path": str(self.path),
            "title": self.title,
            "uploader": self.uploader,
            "webpage_url": self.webpage_url,
            "platform": self.platform.value,
            "duration_ms": self.duration_ms,
            "size_bytes": self.size_bytes,
            "format_id": self.format_id,
            "ext": self.ext,
            "attempts": self.attempts,
            "resumed": self.resumed,
            "elapsed_ms": self.elapsed_ms,
            "subtitles": [str(p) for p in self.subtitles],
            "info_json": str(self.info_json) if self.info_json is not None else None,
            "warnings": list(self.warnings),
        }

    def as_source(self) -> MediaSource:
        """转回 :class:`MediaSource`，供下游统一处理。"""
        return MediaSource(
            video_id=self.video_id,
            kind=SourceKind.LOCAL_FILE,
            raw=self.webpage_url or str(self.path),
            path=self.path,
            url=self.webpage_url or None,
            platform=self.platform,
            suffix=self.ext,
            size_bytes=self.size_bytes,
            info=self.info,
            warnings=self.warnings,
        )


# --------------------------------------------------------------------------- #
# 下载器
# --------------------------------------------------------------------------- #
class MediaDownloader:
    """基于 yt-dlp 的下载器。

    >>> dl = MediaDownloader(DownloadConfig(videos_dir=Path("data/videos")))
    >>> src = parse_source("https://www.bilibili.com/video/BV1xx411c7mD")
    >>> result = dl.download(src)
    >>> result.path.exists()
    True
    """

    def __init__(self, config: DownloadConfig | None = None) -> None:
        self.config = config or DownloadConfig()
        self.config.validate()
        self._ydl_module: Any = None
        self._ffmpeg: str | None = None

    # ---- 延迟导入 --------------------------------------------------------- #
    @property
    def ydl(self) -> Any:
        """延迟导入 yt_dlp：未安装只在真正下载时报错，不影响模块 import。"""
        if self._ydl_module is None:
            try:
                import yt_dlp  # type: ignore[import-untyped]
            except ImportError as exc:  # pragma: no cover - 取决于环境
                raise DownloaderUnavailable(
                    "未安装 yt-dlp。请执行 `python -m pip install -r requirements.txt` 后重试。"
                ) from exc
            self._ydl_module = yt_dlp
        return self._ydl_module

    def ffmpeg_path(self) -> str | None:
        """合并音视频流需要 ffmpeg；缺失时降级为单文件格式。"""
        if self._ffmpeg is None:
            try:
                self._ffmpeg = resolve_binary("ffmpeg")
            except MediaError:
                logger.warning("找不到 ffmpeg，将只下载单文件格式（可能分辨率较低）")
                self._ffmpeg = ""
        return self._ffmpeg or None

    # ---- 公共 API --------------------------------------------------------- #
    def download(
        self,
        source: MediaSource,
        *,
        progress: ProgressCallback | None = None,
        force: bool = False,
    ) -> DownloadResult:
        """下载 ``source`` 指向的远程视频。

        :raises DownloadError: 所有重试耗尽，或命中不可重试错误
        :raises DownloaderUnavailable: yt-dlp 不可用
        """
        if source.kind is not SourceKind.REMOTE_URL or not source.url:
            raise DownloadError(f"download() 只接受远程 URL，收到 {source.kind.value}")

        started = time.monotonic()
        target_dir = self._target_dir(source.video_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        cached = self._reuse(source, target_dir, force=force)
        if cached is not None:
            logger.info("复用已下载产物: %s", cached.path.name)
            return replace(cached, elapsed_ms=int((time.monotonic() - started) * 1000))

        attempts_allowed = self.config.max_retries + 1
        errors: list[str] = []
        for attempt in range(1, attempts_allowed + 1):
            try:
                result = self._attempt(source, target_dir, attempt, progress)
            except DownloadError as exc:
                message = str(exc)
                fatal = self._is_fatal(message)
                errors.append(f"[#{attempt}] {message}")
                logger.warning(
                    "下载失败（第 %d/%d 次，%s）: %s",
                    attempt,
                    attempts_allowed,
                    "不可重试" if fatal else "可重试",
                    message[:300],
                )
                if fatal or attempt >= attempts_allowed:
                    break
                delay = min(self.config.backoff_max, self.config.backoff_base**attempt)
                logger.info("%.1fs 后重试（指数退避）", delay)
                if progress is not None:
                    self._safe_progress(
                        progress,
                        DownloadProgress(
                            video_id=source.video_id,
                            status="retrying",
                            attempt=attempt + 1,
                        ),
                    )
                time.sleep(delay)
            else:
                elapsed = int((time.monotonic() - started) * 1000)
                return replace(result, elapsed_ms=elapsed)

        raise DownloadError(
            f"下载失败（{len(errors)} 次尝试）: {source.url}\n" + "\n".join(errors)
        )

    def probe_result(self, result: DownloadResult) -> DownloadResult:
        """下载后探测时长/分辨率，并按 ``max_duration_ms`` 拒收。"""
        info = probe_media(result.path)
        limit = self.config.max_duration_ms
        if limit is not None and info.duration_ms > limit:
            raise DownloadError(
                f"远程视频时长 {info.duration_ms / 1000:.1f}s 超过上限 {limit / 1000:.1f}s"
            )
        warnings = result.warnings
        if not info.has_audio:
            warnings = (*warnings, "视频不含音轨，语音转写将跳过")
        return replace(
            result,
            info=info,
            duration_ms=info.duration_ms,
            size_bytes=info.path.stat().st_size,
            warnings=warnings,
        )

    def fetch(
        self,
        source: MediaSource,
        *,
        progress: ProgressCallback | None = None,
        force: bool = False,
        probe: bool = True,
    ) -> DownloadResult:
        """``download`` + ``probe_result`` 的组合入口。"""
        result = self.download(source, progress=progress, force=force)
        return self.probe_result(result) if probe else result

    # ---- 内部实现 --------------------------------------------------------- #
    def _target_dir(self, video_id: str) -> Path:
        return Path(self.config.videos_dir) / sanitize_filename(
            video_id, fallback="video"
        )

    def _reuse(
        self, source: MediaSource, target_dir: Path, *, force: bool
    ) -> DownloadResult | None:
        if force:
            return None
        info_json = target_dir / "source.info.json"
        for candidate in sorted(target_dir.glob("source.*")):
            if candidate.suffix in {
                ".part",
                ".ytdl",
                ".json",
                ".vtt",
                ".srt",
                ".ass",
                ".ssa",
            }:
                continue
            if candidate.stat().st_size <= 0:
                continue
            meta = self._read_info(info_json)
            subtitles = tuple(
                p
                for p in target_dir.glob("source.*")
                if p.suffix.lower().lstrip(".") in self.config.subtitle_formats
            )
            return DownloadResult(
                video_id=source.video_id,
                path=candidate,
                title=str(meta.get("title") or ""),
                uploader=str(meta.get("uploader") or ""),
                webpage_url=str(meta.get("webpage_url") or source.url or ""),
                platform=source.platform,
                duration_ms=int(meta["duration"] * 1000)
                if meta.get("duration")
                else None,
                size_bytes=candidate.stat().st_size,
                format_id=str(meta.get("format_id") or ""),
                ext=candidate.suffix.lstrip("."),
                attempts=0,
                resumed=True,
                subtitles=subtitles,
                info_json=info_json if info_json.is_file() else None,
            )
        return None

    def _read_info(self, path: Path) -> Mapping[str, Any]:
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            logger.debug("无法解析 info.json: %s", path)
            return {}
        return payload if isinstance(payload, Mapping) else {}

    def _build_options(
        self,
        source: MediaSource,
        target_dir: Path,
        attempt: int,
        progress: ProgressCallback | None,
    ) -> dict[str, Any]:
        cfg = self.config
        # 相对模板配绝对目录，确保 yt-dlp 真正使用临时目录和最终输出目录。
        outtmpl = "source.%(ext)s"
        hooks = (
            [self._hook(source.video_id, attempt, progress)]
            if progress is not None
            else []
        )

        options: dict[str, Any] = {
            "outtmpl": {"default": outtmpl},
            "format": cfg.format_selector,
            "merge_output_format": cfg.merge_output_format,
            "noplaylist": True,
            "quiet": cfg.quiet,
            "no_warnings": False,
            "noprogress": True,
            "socket_timeout": cfg.socket_timeout_s,
            "retries": cfg.retries_per_fragment,
            "fragment_retries": cfg.retries_per_fragment,
            "concurrent_fragment_downloads": cfg.concurrent_fragments,
            "continuedl": cfg.resume,
            "overwrites": False,
            "progress_hooks": hooks,
            "postprocessor_hooks": hooks,
            "writethumbnail": False,
            "writesubtitles": cfg.write_subtitles,
            "writeautomaticsub": cfg.write_subtitles,
            "subtitleslangs": list(cfg.subtitle_langs),
            "subtitlesformat": "/".join(cfg.subtitle_formats),
            "writeinfojson": cfg.write_info_json,
            "restrictfilenames": False,
            "windowsfilenames": True,
            "ignoreerrors": False,
            "nocheckcertificate": False,
            "logger": _YdlLogger(),
        }
        ffmpeg = self.ffmpeg_path()
        if ffmpeg:
            options["ffmpeg_location"] = ffmpeg
        else:
            # 无 ffmpeg 时不能用需要合并的格式选择器
            options["format"] = "b[ext=mp4]/b"
            options["merge_output_format"] = None
        if cfg.cookie_file is not None and Path(cfg.cookie_file).is_file():
            options["cookiefile"] = str(cfg.cookie_file)
        elif cfg.cookies_from_browser:
            options["cookiesfrombrowser"] = (cfg.cookies_from_browser,)
        if source.platform == Platform.BILIBILI and cfg.write_subtitles and not (
            options.get("cookiefile") or options.get("cookiesfrombrowser")
        ):
            # B站平台字幕要求登录；匿名下载不做必然被拒绝的字幕请求。
            # 三个必需本地模型仍照常使用，语音由 Whisper 转写，不是模型降级。
            options["writesubtitles"] = options["writeautomaticsub"] = False
            logger.info("B站未配置登录信息，跳过受限平台字幕，语音使用 Whisper 转写")
        options["paths"] = {"home": str(target_dir.resolve())}
        if cfg.temp_dir is not None:
            # 每个视频独立临时目录，防止并发任务的 source.part 互相覆盖。
            temp = Path(cfg.temp_dir).resolve() / target_dir.name
            if cfg.resume and any(target_dir.glob("*.part")):
                # 旧版本的断点文件位于最终目录，沿用该目录完成本次续传。
                temp = target_dir.resolve()
            temp.mkdir(parents=True, exist_ok=True)
            options["paths"]["temp"] = str(temp)
        if cfg.max_file_bytes:
            options["max_filesize"] = cfg.max_file_bytes
        return options

    def _attempt(
        self,
        source: MediaSource,
        target_dir: Path,
        attempt: int,
        progress: ProgressCallback | None,
    ) -> DownloadResult:
        options = self._build_options(source, target_dir, attempt, progress)
        ydl_mod = self.ydl
        resumed = any(target_dir.glob("*.part")) or any(
            Path(options["paths"].get("temp", target_dir)).glob("*.part")
        )
        try:
            with ydl_mod.YoutubeDL(options) as ydl:
                info = ydl.extract_info(source.url, download=True)
        except Exception as exc:  # yt-dlp 的异常层次不稳定，这里统一归类
            raise DownloadError(f"{type(exc).__name__}: {exc}") from exc

        if not isinstance(info, Mapping):
            raise DownloadError(f"yt-dlp 未返回元数据（{type(info).__name__}）")

        entries = info.get("entries")
        if entries:
            first = next((e for e in entries if isinstance(e, Mapping)), None)
            if first is None:
                raise DownloadError("播放列表为空")
            info = first

        path = self._locate_output(target_dir)
        if path is None:
            raise DownloadError(
                f"下载完成但找不到产物文件（目录 {target_dir}）；可能是仅音频/仅封面，或格式不可用"
            )
        if (
            self.config.max_file_bytes
            and path.stat().st_size > self.config.max_file_bytes
        ):
            path.unlink(missing_ok=True)
            raise DownloadError(
                f"下载产物 {path.stat().st_size / 1024**3:.2f} GB 超过上限，已删除"
            )

        subtitles = self._collect_subtitles(target_dir)
        info_json = target_dir / "source.info.json"
        warnings: list[str] = []
        if not subtitles and self.config.write_subtitles:
            warnings.append("未获取到字幕，将走语音识别路径")
        duration = info.get("duration")
        return DownloadResult(
            video_id=source.video_id,
            path=path,
            title=str(info.get("title") or ""),
            uploader=str(info.get("uploader") or info.get("channel") or ""),
            webpage_url=str(info.get("webpage_url") or source.url or ""),
            platform=source.platform,
            duration_ms=int(float(duration) * 1000) if duration else None,
            size_bytes=path.stat().st_size,
            format_id=str(info.get("format_id") or ""),
            ext=path.suffix.lstrip("."),
            attempts=attempt,
            resumed=resumed,
            subtitles=subtitles,
            info_json=info_json if info_json.is_file() else None,
            warnings=tuple(warnings),
        )

    def _locate_output(self, target_dir: Path) -> Path | None:
        """在目标目录里找主产物（排除字幕/元数据/临时文件）。"""
        skip = {
            ".part",
            ".ytdl",
            ".json",
            "jpg",
            "jpeg",
            "png",
            "webp",
            "vtt",
            "srt",
            "ass",
            "ssa",
        }
        candidates = [
            p
            for p in target_dir.glob("source.*")
            if p.is_file()
            and p.suffix.lstrip(".").lower() not in skip
            and p.stat().st_size > 0
        ]
        if not candidates:
            return None
        # 优先已知视频容器，其次取体积最大
        preferred = [
            p
            for p in candidates
            if p.suffix.lstrip(".").lower()
            in {"mp4", "mkv", "webm", "flv", "mov", "avi", "ts", "m4v"}
        ]
        pool = preferred or candidates
        return max(pool, key=lambda p: p.stat().st_size)

    def _collect_subtitles(self, target_dir: Path) -> tuple[Path, ...]:
        found = [
            p
            for p in target_dir.glob("source.*")
            if p.is_file()
            and p.suffix.lstrip(".").lower() in self.config.subtitle_formats
        ]
        return tuple(sorted(found, key=lambda p: p.name))

    def _hook(
        self, video_id: str, attempt: int, progress: ProgressCallback | None
    ) -> Callable[[Mapping[str, Any]], None]:
        def _inner(payload: Mapping[str, Any]) -> None:
            if progress is None:
                return
            status = str(payload.get("status") or "")
            total = payload.get("total_bytes") or payload.get("total_bytes_estimate")
            event = DownloadProgress(
                video_id=video_id,
                status=status,
                downloaded_bytes=int(payload.get("downloaded_bytes") or 0),
                total_bytes=int(total) if total else None,
                speed_bps=float(payload["speed"]) if payload.get("speed") else None,
                eta_s=float(payload["eta"]) if payload.get("eta") is not None else None,
                filename=str(payload.get("filename") or ""),
                attempt=attempt,
            )
            self._safe_progress(progress, event)

        return _inner

    @staticmethod
    def _safe_progress(progress: ProgressCallback, event: DownloadProgress) -> None:
        try:
            progress(event)
        except Exception as exc:  # noqa: BLE001 - 回调是外部代码，不能让它炸掉下载
            logger.debug("进度回调抛异常，已忽略: %s", exc)

    @staticmethod
    def _is_fatal(message: str) -> bool:
        lowered = message.lower()
        if any(marker in lowered for marker in _FATAL_MARKERS):
            return True
        if any(marker in lowered for marker in _RETRYABLE_MARKERS):
            return False
        # 未知错误：保守重试（有退避与上限，不会无限消耗）
        return False

    # ---- 清理 ------------------------------------------------------------- #
    def purge(self, video_id: str) -> bool:
        """删除某视频的下载目录；返回是否真的删了东西。"""
        target = self._target_dir(video_id)
        if not target.exists():
            return False
        shutil.rmtree(target, ignore_errors=True)
        logger.info("已清理下载目录: %s", target)
        return True


class _YdlLogger:
    """把 yt-dlp 的日志接到标准 logging（默认它直接 print 到 stdout）。"""

    def debug(self, msg: str) -> None:
        if msg.startswith("[debug] "):
            return
        logger.debug(msg)

    def info(self, msg: str) -> None:
        logger.debug(msg)

    def warning(self, msg: str) -> None:
        logger.warning("yt-dlp: %s", msg)

    def error(self, msg: str) -> None:
        logger.error("yt-dlp: %s", msg)
