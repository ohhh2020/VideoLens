"""字幕解析（SRT / WebVTT / ASS / SSA）。

为什么自己写
------------
需求要求"优先复用视频自带字幕"，而字幕格式的边角情况极多：BOM、GBK 编码、
WebVTT 的 ``NOTE``/``STYLE`` 块与 cue settings、ASS 的 ``Dialogue`` 千分位
与 ``\\N`` 硬换行、以及各种站点的自动字幕带 ``<c.color>`` 内联标签。
引入 ``pysrt``/``webvtt-py`` 会多两个依赖，且它们对 ASS 支持并不完整，
所以这里做一个约 300 行、零依赖、可单测的解析器。

输出统一为 :class:`SubtitleCue`，时间单位**毫秒整数**——这是全项目的溯源基准。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Iterable, Sequence

from backend.media.errors import MediaError

logger = logging.getLogger("zhiying.media.subtitles")

__all__ = [
    "SUPPORTED_SUBTITLE_SUFFIXES",
    "SubtitleCue",
    "SubtitleError",
    "detect_encoding",
    "load_subtitles",
    "merge_cues",
    "parse_ass",
    "parse_srt",
    "parse_subtitle",
    "parse_vtt",
    "strip_markup",
]

SUPPORTED_SUBTITLE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {"srt", "vtt", "ass", "ssa", "txt"}
)

_ENCODINGS: Final[tuple[str, ...]] = (
    "utf-8-sig",
    "utf-8",
    "utf-16",
    "gb18030",
    "big5",
    "latin-1",
)

# 00:00:01,000 / 00:00:01.000 / 0:01.000 / 00:01
_TIMECODE_RE: Final = re.compile(
    r"(?:(?P<h>\d{1,3}):)?(?P<m>\d{1,3}):(?P<s>\d{1,2})"
    r"(?:(?P<sep>[,.])(?P<ms>\d{1,3}))?"
)


class SubtitleError(MediaError):
    """字幕文件无法解析（编码不可识别、格式不认识、时间轴全非法）。"""


@dataclass(frozen=True, slots=True)
class SubtitleCue:
    """一条字幕。``index`` 为文件内顺序（从 0 起），便于回溯原始行。"""

    start_ms: int
    end_ms: int
    text: str
    index: int = 0

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def __post_init__(self) -> None:  # pragma: no cover - 轻量校验
        if self.end_ms < self.start_ms:
            raise ValueError(f"字幕时间轴倒挂: {self.start_ms} > {self.end_ms}")

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
        }


# --------------------------------------------------------------------------- #
# 编码
# --------------------------------------------------------------------------- #
def detect_encoding(path: str | Path, sample_bytes: int = 65536) -> str:
    """按 BOM 与试解码挑一个能用的编码，都不行时回退 ``utf-8`` + replace。"""
    file_path = Path(path)
    with file_path.open("rb") as handle:
        sample = handle.read(sample_bytes)
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    for encoding in _ENCODINGS:
        try:
            sample.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        return encoding
    logger.warning("无法确定字幕编码，按 utf-8 容错读取: %s", file_path.name)
    return "utf-8"


def _read_text(path: Path) -> str:
    encoding = detect_encoding(path)
    try:
        return path.read_text(encoding=encoding, errors="replace")
    except OSError as exc:
        raise SubtitleError(f"读取字幕失败: {path} ({exc})") from exc


# --------------------------------------------------------------------------- #
# 文本清洗
# --------------------------------------------------------------------------- #
_TAG_RE: Final = re.compile(r"<[^>]{1,40}>|\{[^}]{1,40}\}")
_ASS_OVERRIDE_RE: Final = re.compile(r"\{[^}]*\}")
_MULTISPACE_RE: Final = re.compile(r"[ \t]{2,}")


def strip_markup(text: str) -> str:
    """剥掉字幕里的内联标签，只留可读文本。

    覆盖：``<i>``/``<b>``/``<font ...>``、WebVTT 的 ``<c.green>``、``<00:00:01.000>``
    时间戳标记、ASS 的 ``{\\an8}`` 覆盖块、``\\N``/``\\n`` 硬换行。
    """
    cleaned = text.replace("\\N", " ").replace("\\n", " ")
    cleaned = _ASS_OVERRIDE_RE.sub("", cleaned)
    cleaned = re.sub(r"<(?:\d{1,2}:)?\d{1,2}:\d{1,2}[.,]\d{1,3}>", "", cleaned)
    cleaned = _TAG_RE.sub("", cleaned)
    cleaned = cleaned.replace("\u200b", "").replace("\ufeff", "")
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


# --------------------------------------------------------------------------- #
# 时间码
# --------------------------------------------------------------------------- #
def _timecode_to_ms(text: str) -> int | None:
    match = _TIMECODE_RE.search(text.strip())
    if match is None:
        return None
    hours = int(match.group("h") or 0)
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    frac = match.group("ms") or "0"
    # ".5" 表示 500ms，".05" 表示 50ms
    millis = int(frac.ljust(3, "0")[:3])
    total = ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis
    return total if total >= 0 else None


def _split_timing(line: str) -> tuple[int, int] | None:
    """把 ``00:00:01,000 --> 00:00:04,000 align:start`` 解成毫秒区间。"""
    if "-->" not in line:
        return None
    left, _, right = line.partition("-->")
    start = _timecode_to_ms(left)
    # 右侧可能带 cue settings（WebVTT），先取前两个 token 再解析
    right_token = right.strip().split()
    end = _timecode_to_ms(right_token[0]) if right_token else None
    if start is None or end is None:
        return None
    if end < start:
        start, end = end, start
    return start, end


# --------------------------------------------------------------------------- #
# 各格式解析
# --------------------------------------------------------------------------- #
def _normalize_newlines(text: str) -> str:
    """统一换行，并折叠 ``\\r\\r\\n`` 这类双 CR（某些 Windows 字幕工具会产出）。"""
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    return out


def _parse_timed(text: str, *, drop_lines: tuple[str, ...] = ()) -> list[SubtitleCue]:
    """行驱动的字幕解析器（SRT / WebVTT 共用）。

    为什么不用"空行分块"：真实的字幕文件里经常出现
    ``\\r\\r\\n``、多余空行、正文内空行，块切分会把一条 cue 劈成两半，
    导致整份字幕解析失败。改成扫描时间轴行、把其后的行收进当前 cue，
    对空行数量完全不敏感。

    :param drop_lines: 需要整行丢弃的前缀（如 WebVTT 的 ``NOTE``/``STYLE``）
    """
    lines = _normalize_newlines(text).split("\n")
    cues: list[SubtitleCue] = []
    timing: tuple[int, int] | None = None
    buffer: list[str] = []
    dropping = False  # 处于 NOTE/STYLE 之类的非 cue 块中

    def flush() -> None:
        nonlocal timing, buffer
        if timing is not None and buffer:
            body = strip_markup(" ".join(buffer))
            if body:
                cues.append(
                    SubtitleCue(
                        start_ms=timing[0], end_ms=timing[1], text=body, index=len(cues)
                    )
                )
        timing = None
        buffer = []

    for raw in lines:
        line = raw.strip()
        if not line:
            # 空行结束当前正文；不在块内则忽略
            if timing is not None:
                flush()
            dropping = False
            continue
        if dropping:
            continue
        if any(line.startswith(prefix) for prefix in drop_lines):
            flush()
            dropping = True
            continue
        found = _split_timing(line)
        if found is not None:
            flush()
            timing = found
            buffer = []
            continue
        if timing is None:
            continue  # cue 序号、文件头、cue 标识行
        if line.isdigit() and not buffer:
            continue  # 紧贴时间轴的序号（不同工具顺序不一）
        buffer.append(line)
    flush()
    return cues


def parse_srt(text: str) -> list[SubtitleCue]:
    """解析 SubRip。容忍缺编号、多余空行、CRLF / 双 CR、时间轴交叠。"""
    return _parse_timed(text)


def parse_vtt(text: str) -> list[SubtitleCue]:
    """解析 WebVTT：跳过 ``NOTE``/``STYLE``/``REGION``/``X-TIMESTAMP-MAP`` 块。"""
    return _parse_timed(
        text,
        drop_lines=("WEBVTT", "NOTE", "STYLE", "REGION", "X-TIMESTAMP-MAP"),
    )


_ASS_DIALOGUE: Final = re.compile(r"^Dialogue:\s*(?P<rest>.*)$", re.IGNORECASE)
_ASS_FIELDS: Final = 10
"""ASS 的 Dialogue 有 10 个逗号分隔字段，第 10 个（Text）自身可含逗号。"""


def parse_ass(text: str) -> list[SubtitleCue]:
    """解析 Advanced SubStation Alpha / SSA 的 ``Dialogue`` 行。"""
    cues: list[SubtitleCue] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith(("dialogue:", "comment:")):
            if line.lower().startswith("comment:"):
                continue
            match = _ASS_DIALOGUE.match(line)
            if match is None:
                continue
            parts = match.group("rest").split(",", _ASS_FIELDS - 1)
            if len(parts) < _ASS_FIELDS:
                continue
            start = _timecode_to_ms(parts[1])
            end = _timecode_to_ms(parts[2])
            if start is None or end is None:
                continue
            if end < start:
                start, end = end, start
            body = strip_markup(parts[9])
            if not body:
                continue
            cues.append(
                SubtitleCue(start_ms=start, end_ms=end, text=body, index=len(cues))
            )
    return cues


def parse_subtitle(text: str, *, suffix: str = "") -> list[SubtitleCue]:
    """按内容/后缀自动分发解析器。"""
    head = text.lstrip()[:512].lower()
    ext = suffix.lower().lstrip(".")
    if ext in {"ass", "ssa"} or "[script info]" in head or "dialogue:" in head:
        cues = parse_ass(text)
        if cues:
            return cues
    if ext == "vtt" or head.startswith("webvtt"):
        cues = parse_vtt(text)
        if cues:
            return cues
    cues = parse_srt(text)
    if cues:
        return cues
    # 内容被判错时再兜一圈：VTT 与 ASS 都试
    for parser in (parse_vtt, parse_ass):
        cues = parser(text)
        if cues:
            logger.debug("字幕按 %s 兜底解析成功", parser.__name__)
            return cues
    return []


# --------------------------------------------------------------------------- #
# 合并 / 清洗
# --------------------------------------------------------------------------- #
def merge_cues(
    cues: Sequence[SubtitleCue],
    *,
    max_gap_ms: int = 120,
    min_duration_ms: int = 200,
    max_duration_ms: int = 30_000,
    join_text: bool = True,
) -> list[SubtitleCue]:
    """整理字幕时间轴。

    * 丢弃空文本、纯标点
    * ``join_text=True`` 时，相邻且间隔 ``<= max_gap_ms`` 的 cue 合并
      （自动字幕常被切成碎句）；``False`` 时只整理时间轴，不粘文本
    * 时长不足 ``min_duration_ms`` 的补齐，超过 ``max_duration_ms`` 的截断
    * 重叠部分按"后者起点前移"消除，保证单调不交叠

    ``join_text=False`` 用于 ASR 输入：字幕的 cue 边界本身是语义边界，
    盲目合并会把"复读幻觉行"和后面的正文粘成一条，清洗就失效了。
    """
    prepared: list[SubtitleCue] = []
    for cue in cues:
        body = cue.text.strip()
        if not body or not re.search(r"\w", body, re.UNICODE):
            continue
        start = max(0, cue.start_ms)
        end = max(start + 1, cue.end_ms)
        prepared.append(replace(cue, start_ms=start, end_ms=end, text=body))
    if not prepared:
        return []

    prepared.sort(key=lambda c: (c.start_ms, c.end_ms))
    merged: list[SubtitleCue] = []
    for cue in prepared:
        if join_text and merged and cue.start_ms - merged[-1].end_ms <= max_gap_ms:
            prev = merged[-1]
            joined = f"{prev.text} {cue.text}".strip()
            merged[-1] = replace(
                prev, end_ms=max(prev.end_ms, cue.end_ms), text=joined, index=prev.index
            )
        else:
            merged.append(replace(cue, index=len(merged)))

    fixed: list[SubtitleCue] = []
    for i, cue in enumerate(merged):
        start = cue.start_ms
        end = cue.end_ms
        if end - start < min_duration_ms:
            end = start + min_duration_ms
        if end - start > max_duration_ms:
            end = start + max_duration_ms
        if i + 1 < len(merged) and end > merged[i + 1].start_ms:
            end = max(start + 1, merged[i + 1].start_ms)
        fixed.append(replace(cue, start_ms=start, end_ms=end, index=i))
    return fixed


def load_subtitles(
    paths: Iterable[str | Path],
    *,
    preferred_order: Sequence[str] = ("srt", "vtt", "ass", "ssa"),
) -> tuple[list[SubtitleCue], Path | None, list[str]]:
    """按偏好顺序依次尝试多个字幕文件，返回第一个可用的解析结果。

    :return: ``(cues, 命中的文件, 告警列表)``；全部失败时 ``cues == []``
    """
    order = {ext.lower(): i for i, ext in enumerate(preferred_order)}
    candidates = [Path(p) for p in paths if Path(p).is_file()]
    candidates.sort(
        key=lambda p: (order.get(p.suffix.lower().lstrip("."), len(order)), p.name)
    )
    warnings: list[str] = []
    for path in candidates:
        try:
            text = _read_text(path)
            cues = merge_cues(parse_subtitle(text, suffix=path.suffix), join_text=False)
        except SubtitleError as exc:
            warnings.append(str(exc))
            continue
        if not cues:
            warnings.append(f"字幕文件解析后为空: {path.name}")
            continue
        logger.info("使用字幕 %s（%d 条）", path.name, len(cues))
        return cues, path, warnings
    if not candidates:
        warnings.append("没有可用的字幕文件")
    return [], None, warnings
