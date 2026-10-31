"""输入解析与合法性校验（本地文件 / 远程 URL）。

上游（CLI、Web、MCP）拿到的永远是一段字符串：可能是 ``D:\\videos\\a.mp4``、
可能是 ``https://www.bilibili.com/video/BV1xx411c7mD``、也可能是用户手滑贴进来的
``file:///D:/a.mp4`` 或带空格的 ``"D:\\my videos\\a.mp4"``。

本模块负责把它们统一成 :class:`MediaSource`，并完成**在下载/转码之前**就该拒绝的检查：

* URL 长度（默认 ≤2048 字符）
* 本地扩展名白名单（MP4/MKV/AVI/MOV/FLV/WEBM/TS/M4V/3GP…）
* 本地文件大小（默认 ≤4GB）
* 视频时长（默认 ≤2h）—— 需要 ffprobe，故仅对本地文件在 :func:`parse_local` 内做
* 可读性 / 是否为目录 / 是否为设备文件

设计约束
--------
* **不做网络 IO**：URL 可达性、yt-dlp 支持性由 :mod:`backend.media.downloader` 负责。
  这样 ``parse_source`` 可以在任何上下文（含无网络）安全调用。
* **不做隐式下载**：返回的 :class:`MediaSource` 里 ``path`` 对 URL 输入恒为 ``None``，
  调用方必须先下后探。
* 时长校验允许延迟：``probe=False`` 时只做静态检查，返回的 ``duration_ms`` 为 ``None``，
  由调用方在下载/探测后调用 :meth:`MediaSource.with_probe` 补齐。
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterable
from urllib.parse import unquote, urlsplit

from backend.media.errors import MediaError, UnsupportedMediaError
from backend.media.ffmpeg_utils import MediaInfo, probe_media

logger = logging.getLogger("zhiying.media.parser")

__all__ = [
    "DEFAULT_ALLOWED_EXTENSIONS",
    "MAX_PATH_LENGTH",
    "MediaSource",
    "Platform",
    "SourceKind",
    "VideoTooLargeError",
    "VideoTooLongError",
    "checksum_file",
    "derive_video_id",
    "guess_platform",
    "parse_source",
    "sanitize_filename",
]

MAX_PATH_LENGTH: Final[int] = 32_767
"""Windows 长路径上限（NTFS 未开启 LongPathsEnabled 时实际为 260）。"""

DEFAULT_ALLOWED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        "mp4",
        "mkv",
        "avi",
        "mov",
        "flv",
        "webm",
        "ts",
        "m4v",
        "3gp",
        "mpg",
        "mpeg",
        "wmv",
    }
)

_SCHEME_RE: Final = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*)://")
_URL_RE: Final = re.compile(r"^https?://", re.IGNORECASE)
_WIN_DRIVE_RE: Final = re.compile(r"^[A-Za-z]:[\\/]")
_WIN_UNC_RE: Final = re.compile(r"^\\\\[^\\]+\\")
_SPACE_RE: Final = re.compile(r"\s+")
_UNSAFE_NAME_RE: Final = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class VideoTooLongError(MediaError):
    """视频时长超过 ``limits.max_duration_ms``。"""


class VideoTooLargeError(MediaError):
    """本地文件体积超过 ``limits.max_file_bytes``。"""


class SourceKind(str, Enum):
    """输入来源类型。"""

    LOCAL_FILE = "local_file"
    REMOTE_URL = "remote_url"

    @property
    def is_local(self) -> bool:
        return self is SourceKind.LOCAL_FILE


class Platform(str, Enum):
    """已知平台（仅用于下载器选择与提示语，不做强约束）。"""

    YOUTUBE = "youtube"
    BILIBILI = "bilibili"
    DOUYIN = "douyin"
    KUAISHOU = "kuaishou"
    XIGUA = "xigua"
    YOUKU = "youku"
    IQIYI = "iqiyi"
    TENCENT = "tencent"
    WEIBO = "weibo"
    XIAOHONGSHU = "xiaohongshu"
    VIMEO = "vimeo"
    TWITCH = "twitch"
    DAILYMOTION = "dailymotion"
    GENERIC = "generic"


_PLATFORM_RULES: Final[tuple[tuple[Platform, tuple[str, ...]], ...]] = (
    (Platform.YOUTUBE, ("youtube.com", "youtu.be", "youtube-nocookie.com")),
    (Platform.BILIBILI, ("bilibili.com", "b23.tv", "bili2233.cn")),
    (Platform.DOUYIN, ("douyin.com", "iesdouyin.com")),
    (Platform.KUAISHOU, ("kuaishou.com", "gifshow.com", "chenzhongtech.com")),
    (Platform.XIGUA, ("ixigua.com", "xigua.com")),
    (Platform.YOUKU, ("youku.com", "m.youku.com")),
    (Platform.IQIYI, ("iqiyi.com", "iq.com")),
    (Platform.TENCENT, ("v.qq.com", "qq.com")),
    (Platform.WEIBO, ("weibo.com", "weibo.cn", "m.weibo.cn")),
    (Platform.XIAOHONGSHU, ("xiaohongshu.com", "xhslink.com")),
    (Platform.VIMEO, ("vimeo.com",)),
    (Platform.TWITCH, ("twitch.tv",)),
    (Platform.DAILYMOTION, ("dailymotion.com", "dai.ly")),
)


# --------------------------------------------------------------------------- #
# 源描述
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MediaSource:
    """一段待处理视频的归一化描述。

    溯源铁律的一部分：``video_id`` 在此处生成（本地文件用内容指纹，URL 用规范化
    地址指纹），后续所有 ``clip_id`` / ``keyframe_id`` / 引用都挂在它下面。
    """

    video_id: str
    kind: SourceKind
    raw: str
    """用户原始输入，原样保留用于回显与排障。"""

    path: Path | None = None
    url: str | None = None
    platform: Platform = Platform.GENERIC
    suffix: str = ""
    size_bytes: int | None = None
    info: MediaInfo | None = None
    warnings: tuple[str, ...] = ()

    # ---- 派生属性 ---------------------------------------------------------- #
    @property
    def duration_ms(self) -> int | None:
        return self.info.duration_ms if self.info is not None else None

    @property
    def is_local(self) -> bool:
        return self.kind is SourceKind.LOCAL_FILE

    @property
    def display_name(self) -> str:
        """用于日志/UI 的短名：本地取文件名，URL 取主机+路径末段。"""
        if self.path is not None:
            return self.path.name
        if self.url:
            parts = urlsplit(self.url)
            tail = parts.path.rstrip("/").rsplit("/", 1)[-1]
            return f"{parts.netloc}/{tail}" if tail else parts.netloc
        return self.raw[:80]

    @property
    def requires_download(self) -> bool:
        return self.kind is SourceKind.REMOTE_URL

    def with_probe(self, info: MediaInfo) -> MediaSource:
        """补齐探测结果（下载完成后调用）。"""
        warnings = self.warnings
        if not info.has_audio:
            warnings = (*warnings, "视频不含音轨，语音转写将跳过，仅做视觉理解")
        return replace(
            self, info=info, size_bytes=info.path.stat().st_size, warnings=warnings
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON 友好形式；``info`` 摊平后只保留溯源需要的字段。"""
        return {
            "video_id": self.video_id,
            "kind": self.kind.value,
            "raw": self.raw,
            "path": str(self.path) if self.path is not None else None,
            "url": self.url,
            "platform": self.platform.value,
            "suffix": self.suffix,
            "size_bytes": self.size_bytes,
            "duration_ms": self.duration_ms,
            "warnings": list(self.warnings),
            "probe": self.info.to_dict() if self.info is not None else None,
        }


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def guess_platform(url: str) -> Platform:
    """按主机名匹配已知平台；未命中返回 :attr:`Platform.GENERIC`。"""
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return Platform.GENERIC
    if not host:
        return Platform.GENERIC
    for platform, domains in _PLATFORM_RULES:
        for domain in domains:
            if host == domain or host.endswith("." + domain):
                return platform
    return Platform.GENERIC


def sanitize_filename(
    name: str, *, fallback: str = "video", max_length: int = 120
) -> str:
    """把任意文本压成可用作文件名/目录名的安全串。

    处理：Unicode NFC 归一 → 去控制字符与 Windows 保留字符 → 折叠空白
    → 去首尾点与空格 → 截断 → 空串时用 ``fallback``。
    """
    text = unicodedata.normalize("NFC", str(name or ""))
    text = _UNSAFE_NAME_RE.sub("_", text)
    text = _SPACE_RE.sub(" ", text).strip(" .")
    if len(text) > max_length:
        text = text[:max_length].rstrip(" .")
    return text or fallback


def checksum_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """计算内容指纹：sha256(大小 + 首块 + 末块)。

    对 1GB 视频只需读 2MB，比全量 sha256 快三个数量级；用于去重足够
    （同大小且首尾 1MB 全同的两段不同视频概率可忽略）。
    """
    file_path = Path(path)
    size = file_path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with file_path.open("rb") as handle:
        digest.update(handle.read(chunk))
        if size > chunk * 2:
            handle.seek(-chunk, os.SEEK_END)
            digest.update(handle.read(chunk))
    return digest.hexdigest()


def derive_video_id(seed: str, *, prefix: str = "v", length: int = 16) -> str:
    """由种子串生成稳定 ``video_id``（``v`` + 16 位 hex）。"""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:length]}"


def _normalize_url(raw: str) -> str:
    """URL 规范化：去 fragment、统一 scheme/host 大小写、保留 query。

    query 必须保留——YouTube 的 ``?v=`` 与 B 站的 ``?p=`` 都是身份的一部分。
    """
    parts = urlsplit(raw.strip())
    scheme = parts.scheme.lower() or "https"
    host = (parts.hostname or "").lower()
    if not host:
        raise UnsupportedMediaError(f"URL 缺少主机名: {raw[:120]}")
    netloc = host
    if parts.port and not (
        (scheme == "http" and parts.port == 80)
        or (scheme == "https" and parts.port == 443)
    ):
        netloc = f"{host}:{parts.port}"
    path = re.sub(r"/{2,}", "/", parts.path) or "/"
    query = f"?{parts.query}" if parts.query else ""
    return f"{scheme}://{netloc}{path}{query}"


def _suffix_of(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


def _strip_quotes(text: str) -> str:
    """去掉用户从资源管理器/聊天框复制过来的包裹引号（含中文全角引号）。"""
    stripped = text.strip()
    pairs = {'"': '"', "'": "'", "“": "”", "‘": "’", "「": "」", "『": "』"}
    if len(stripped) >= 2:
        first, last = stripped[0], stripped[-1]
        if pairs.get(first) == last or (first == last and first in {'"', "'"}):
            return stripped[1:-1].strip()
    return stripped


def _check_length(raw: str, *, field_name: str, limit: int) -> None:
    if len(raw) > limit:
        raise UnsupportedMediaError(
            f"{field_name} 超长：{len(raw)} 字符 > 上限 {limit}；"
            f"请缩短后重试（当前开头：{raw[:60]!r}）"
        )


def _unwrap_local(raw: str | Path) -> str:
    """剥掉包裹本地路径的引号/``file://`` 前缀/行内空白。"""
    text = _strip_quotes(str(raw).strip())
    lowered = text.lower()
    if lowered.startswith("file://"):
        # file:///D:/a.mp4 → /D:/a.mp4 → D:/a.mp4 ；file://server/share → \\server\share
        remainder = text[len("file://") :]
        if remainder.startswith("/") and _WIN_DRIVE_RE.match(remainder[1:]):
            remainder = remainder[1:]
        elif not remainder.startswith("/"):
            remainder = "\\\\" + remainder
        text = unquote(remainder)
    return text.strip()


def _looks_like_local(text: str, base_dir: Path | None = None) -> bool:
    if _WIN_DRIVE_RE.match(text) or _WIN_UNC_RE.match(text):
        return True
    if text.startswith(("/", "./", "~", "\\\\")):
        return True
    if _URL_RE.match(text):
        return False
    match = _SCHEME_RE.match(text)
    if match is not None:
        return match.group("scheme").lower() in {"file"}
    # 裸文件名（demo.mp4）：扩展名在白名单里就按本地处理，
    # 否则 `parse_source("demo.mp4")` 这种最常见的 CLI 用法会被误判成 URL。
    if _suffix_of(Path(text)) in DEFAULT_ALLOWED_EXTENSIONS:
        return True
    if base_dir is not None and (Path(base_dir) / text).exists():
        return True
    # 形如 "videos/a.mp4" 的相对路径
    return "/" in text or "\\" in text


# --------------------------------------------------------------------------- #
# 本地解析
# --------------------------------------------------------------------------- #
def _resolve_local(raw: str | Path, *, base_dir: Path | None) -> Path:
    text = _unwrap_local(raw)
    if not text:
        raise UnsupportedMediaError("本地路径为空")
    expanded = Path(os.path.expandvars(os.path.expanduser(text)))
    if not expanded.is_absolute() and base_dir is not None:
        expanded = base_dir / expanded
    try:
        resolved = expanded.resolve(strict=False)
    except OSError:  # 某些 UNC 路径在无网络时 resolve 会抛
        resolved = expanded.absolute()
    # 文件链接保留为媒体访问路径，身份仍由下方的原始目标生成。
    if expanded.is_symlink():
        return Path(os.path.abspath(expanded))
    return resolved


def parse_local(
    raw: str | Path,
    *,
    allowed_extensions: Iterable[str] = DEFAULT_ALLOWED_EXTENSIONS,
    max_bytes: int | None = None,
    max_duration_ms: int | None = None,
    base_dir: Path | None = None,
    do_probe: bool = True,
    ffprobe: str | None = None,
) -> MediaSource:
    """校验并（可选）探测一个本地视频文件。

    :raises UnsupportedMediaError: 路径非法、扩展名不支持、文件为空
    :raises VideoTooLargeError: 超过 ``max_bytes``
    :raises VideoTooLongError: 超过 ``max_duration_ms``
    :raises ProbeError: ffprobe 无法解析
    """
    path = _resolve_local(raw, base_dir=base_dir)
    if len(str(path)) > MAX_PATH_LENGTH:
        raise UnsupportedMediaError(
            f"路径超长（>{MAX_PATH_LENGTH}）：{str(path)[:120]}…"
        )
    if not path.exists():
        raise UnsupportedMediaError(f"文件不存在: {path}")
    if path.is_dir():
        raise UnsupportedMediaError(
            f"这是一个目录而不是视频文件: {path}；请指定具体文件或批量入口"
        )
    if not path.is_file():
        raise UnsupportedMediaError(f"不是普通文件（可能是设备/管道）: {path}")

    allowed = {e.lower().lstrip(".") for e in allowed_extensions}
    suffix = _suffix_of(path)
    if suffix and suffix not in allowed:
        raise UnsupportedMediaError(
            f"不支持的视频格式 .{suffix}；支持: {', '.join(sorted(allowed))}"
        )

    size = path.stat().st_size
    if size <= 0:
        raise UnsupportedMediaError(f"文件大小为 0，可能下载不完整: {path}")
    if max_bytes is not None and size > max_bytes:
        raise VideoTooLargeError(
            f"文件过大：{size / 1024**3:.2f} GB > 上限 {max_bytes / 1024**3:.2f} GB"
        )

    try:
        video_id = derive_video_id(f"file:{path.resolve()}:{size}")
    except OSError:  # pragma: no cover - 指纹失败不该阻断主流程
        video_id = derive_video_id(f"file:{path}")

    warnings: list[str] = []
    info: MediaInfo | None = None
    if do_probe:
        info = probe_media(path, ffprobe=ffprobe)
        if max_duration_ms is not None and info.duration_ms > max_duration_ms:
            raise VideoTooLongError(
                f"视频过长：{info.duration_ms / 1000:.1f}s > 上限 "
                f"{max_duration_ms / 1000:.1f}s（{max_duration_ms / 3_600_000:.1f}h）"
            )
        if not info.has_audio:
            warnings.append("视频不含音轨，语音转写将跳过，仅做视觉理解")

    return MediaSource(
        video_id=video_id,
        kind=SourceKind.LOCAL_FILE,
        raw=raw,
        path=path,
        platform=Platform.GENERIC,
        suffix=suffix,
        size_bytes=size,
        info=info,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------- #
# 远程解析
# --------------------------------------------------------------------------- #
def parse_url(
    raw: str,
    *,
    max_url_length: int = 2_048,
) -> MediaSource:
    """校验并归一化一个远程视频 URL（**不发起网络请求**）。"""
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    if not text:
        raise UnsupportedMediaError("URL 为空")
    text = _strip_quotes(text)
    _check_length(text, field_name="URL", limit=max_url_length)
    if not _URL_RE.match(text):
        raise UnsupportedMediaError(
            f"只支持 http/https 链接，收到 {text[:60]!r}；本地文件请直接给路径（无需 file:// 前缀）"
        )
    url = _normalize_url(text)
    platform = guess_platform(url)
    suffix = _suffix_of(Path(urlsplit(url).path))
    warnings: tuple[str, ...] = ()
    if platform is Platform.GENERIC:
        warnings = ("未识别的站点，将按通用模式尝试；部分站点可能需要 cookies",)
    return MediaSource(
        video_id=derive_video_id(f"url:{url}"),
        kind=SourceKind.REMOTE_URL,
        raw=raw,
        url=url,
        platform=platform,
        suffix=suffix if suffix in DEFAULT_ALLOWED_EXTENSIONS else "",
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# 统一入口
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ParseOptions:
    """一次性收拢所有校验阈值，便于 CLI 直接由配置构造。"""

    allowed_extensions: frozenset[str] = DEFAULT_ALLOWED_EXTENSIONS
    max_file_bytes: int | None = None
    max_duration_ms: int | None = None
    max_url_length: int = 2_048
    base_dir: Path | None = None
    do_probe: bool = True
    ffprobe: str | None = None

    @classmethod
    def from_config(cls, config: Any) -> ParseOptions:
        """从 :class:`backend.config.AppConfig` 构造（缺字段时回退默认值）。"""
        limits = getattr(config, "limits", None)
        media = getattr(config, "media", None)
        raw_root = Path(str(getattr(getattr(config, "app", None), "root", ".") or "."))
        if not raw_root.is_absolute():
            from backend.config import detect_project_root

            raw_root = detect_project_root() / raw_root
        root = raw_root.resolve()
        extensions = getattr(media, "allowed_extensions", None)
        return cls(
            allowed_extensions=(
                frozenset(str(e).lower().lstrip(".") for e in extensions)
                if extensions
                else DEFAULT_ALLOWED_EXTENSIONS
            ),
            max_file_bytes=getattr(limits, "max_file_bytes", None),
            max_duration_ms=getattr(limits, "max_duration_ms", None),
            max_url_length=getattr(limits, "max_url_length", 2_048),
            base_dir=root,
            do_probe=True,
        )


def parse_source(
    raw: str | Path,
    options: ParseOptions | None = None,
    **overrides: Any,
) -> MediaSource:
    """统一入口：把用户输入解析成 :class:`MediaSource`。

    >>> parse_source("D:/clips/demo.mp4").kind
    <SourceKind.LOCAL_FILE: 'local_file'>
    >>> parse_source("https://youtu.be/xxxx").platform
    <Platform.YOUTUBE: 'youtube'>

    :param options: 校验参数；``None`` 时用默认值
    :param overrides: 覆盖 ``ParseOptions`` 的单个字段
    """
    opts = options or ParseOptions()
    if overrides:
        opts = replace(opts, **overrides)

    text = _strip_quotes(str(raw or "").strip())
    if not text:
        raise UnsupportedMediaError("输入为空")

    if _URL_RE.match(text):
        return parse_url(text, max_url_length=opts.max_url_length)
    if _looks_like_local(text, opts.base_dir):
        return parse_local(
            text,
            allowed_extensions=opts.allowed_extensions,
            max_bytes=opts.max_file_bytes,
            max_duration_ms=opts.max_duration_ms,
            base_dir=opts.base_dir,
            do_probe=opts.do_probe,
            ffprobe=opts.ffprobe,
        )
    # 无分隔符、无 scheme：可能是裸域名（youtu.be/xxx）或写错的路径
    raise UnsupportedMediaError(
        f"无法识别输入 {text[:80]!r}：既不是存在的本地路径，也不是 http/https 链接"
    )
