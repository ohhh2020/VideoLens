"""自适应语音切片器。

把 faster-whisper 输出的毫秒级转录片段，合并成适合检索与上下文注入的切片。

算法要点（对应需求文档表 1「语音转写与切片」/ 表 2 精度约束）::

    1. 候选边界  ①静音点：相邻片段间隔 > 0.6s，切点落在静音区内部；
                 ②句末标点：。！？；… 与 . ! ? ;（数字/缩写里的 ``.`` 不误判）；
                 ③段边界：换段通常意味着换气或换句；
                 ④软边界：，、： 与空格。
    2. 最优切分  在 [10s, 30s] 软硬区间内，对所有候选按
                 ``w_时长接近 × 接近度 + w_边界 × 边界权重 − w_越界 × 越界量``
                 打分取最高者；窗口内无任何候选时（超长无标点语音）落到
                 :meth:`_hard_cut`，把切点吸附到最近字符边界或静音区。
    3. 边界重叠  相邻切片各向外扩 ``overlap_ms/2``，使边界处恰好重叠 1.5s，
                 兼顾语义连续与去重可控；重叠不计入 ``core_text``。

输出结构（:class:`Clip` 与 :meth:`Clip.to_dict`）包含 ``clip_id`` /
``start_time`` / ``end_time`` / ``text`` / ``keyframe_ids``，并附带毫秒字段
``start_ms`` / ``end_ms`` / ``core_start_ms`` / ``core_end_ms`` 用于精确溯源。
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ChunkError",
    "Word",
    "RawSegment",
    "ChunkConfig",
    "Clip",
    "Chunker",
    "segments_from_whisper",
    "bind_keyframes",
]

logger = logging.getLogger("zhiying.rag.chunker")

#: 中文/日文句末（含省略号），优先级最高的文本边界。
_CJK_SENTENCE_ENDS = frozenset("。！？；…‼⁉")
#: 拉丁句末。
_LATIN_SENTENCE_ENDS = frozenset(".!?;")
#: 软边界：停顿但不是句末。
_SOFT_BREAKS = frozenset("，、：,:")
#: 需要清理的控制字符与零宽字符。
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u2060"), None)
_WHITESPACE_RE = re.compile(r"[\t\r\f\v\u00a0\u3000]+")


class ChunkError(RuntimeError):
    """切片过程的致命错误（配置非法、输入结构无法识别等）。"""


# --------------------------------------------------------------------------- #
# 输入模型
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Word:
    """词级时间戳（faster-whisper ``word_timestamps=True`` 的产物）。"""

    start_ms: int
    end_ms: int
    text: str

    @classmethod
    def from_any(cls, obj: Any) -> "Word":
        if isinstance(obj, Word):
            return obj
        if isinstance(obj, Mapping):
            start = _coerce_ms(obj, "start_ms", "start", "t0")
            end = _coerce_ms(obj, "end_ms", "end", "t1")
            text = str(obj.get("text") or obj.get("word") or "")
        else:
            start = _coerce_ms(getattr(obj, "start_ms", None), None, None)
            if start < 0:
                start = int(round(float(getattr(obj, "start", 0.0) or 0.0) * 1000))
            end = _coerce_ms(getattr(obj, "end_ms", None), None, None)
            if end < 0:
                end = int(round(float(getattr(obj, "end", 0.0) or 0.0) * 1000))
            text = str(getattr(obj, "word", None) or getattr(obj, "text", "") or "")
        return cls(start_ms=start, end_ms=end, text=text)


@dataclass(frozen=True, slots=True)
class RawSegment:
    """一条转录片段（毫秒）。

    ``from_any`` 兼容三种输入：
      * :class:`RawSegment` 本身；
      * ``dict``（``start_ms``/``end_ms`` 优先，否则按秒解释 ``start``/``end``）；
      * faster-whisper 的 ``Segment`` 对象（``.start``/``.end`` 为秒）。
    """

    index: int
    start_ms: int
    end_ms: int
    text: str
    words: tuple[Word, ...] = ()

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @classmethod
    def from_any(cls, obj: Any, index: int = 0) -> "RawSegment":
        if isinstance(obj, RawSegment):
            return obj
        if isinstance(obj, Mapping):
            start = _coerce_ms(obj, "start_ms", "start", "t0")
            end = _coerce_ms(obj, "end_ms", "end", "t1")
            text = str(obj.get("text") or obj.get("content") or "").strip()
            raw_words = obj.get("words") or ()
            seg_index = int(obj.get("index", obj.get("id", index)) or index)
        else:
            start = _coerce_ms(getattr(obj, "start_ms", None), None, None)
            if start < 0:
                start = int(round(float(getattr(obj, "start", 0.0) or 0.0) * 1000))
            end = _coerce_ms(getattr(obj, "end_ms", None), None, None)
            if end < 0:
                end = int(round(float(getattr(obj, "end", 0.0) or 0.0) * 1000))
            text = str(getattr(obj, "text", "") or "").strip()
            raw_words = getattr(obj, "words", None) or ()
            seg_index = int(getattr(obj, "id", index) or index)
        try:
            words = tuple(Word.from_any(item) for item in raw_words)
        except TypeError:  # pragma: no cover - 不可迭代的脏数据
            words = ()
        return cls(index=seg_index, start_ms=start, end_ms=end, text=text, words=words)


def _coerce_ms(
    source: Any, ms_key: str | None, sec_key: str | None, alt_key: str | None = None
) -> int:
    """从映射中读取毫秒时间戳；毫秒键优先，其次按秒换算。未找到时返回 -1。"""
    if isinstance(source, Mapping):
        for key in (ms_key, alt_key):
            if key and key in source and source[key] is not None:
                try:
                    return int(round(float(source[key])))
                except (TypeError, ValueError):
                    continue
        if sec_key and sec_key in source and source[sec_key] is not None:
            try:
                return int(round(float(source[sec_key]) * 1000.0))
            except (TypeError, ValueError):
                return -1
    elif source is not None:
        try:
            return int(round(float(source)))
        except (TypeError, ValueError):
            return -1
    return -1


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """切片策略参数（全部可外部注入，禁止硬编码进业务逻辑）。"""

    # -- 时长区间（毫秒）--
    min_ms: int = 10_000
    max_ms: int = 30_000
    target_ms: int = 20_000
    hard_max_ms: int = 45_000  # 任何情况下单切片不得跨越该长度
    min_clip_ms: int = 3_000  # 尾部短于该值的残余直接并入上一段

    # -- 边界识别 --
    silence_threshold_ms: int = 600  # 段间间隔大于该值视为静音点
    silence_inner_offset_ms: int = 300  # 静音区内切点距语音结束的最大偏移
    use_silence: bool = True
    use_sentence: bool = True
    use_segment_boundary: bool = True
    use_soft_breaks: bool = True

    # -- 重叠 --
    overlap_ms: int = 1_500  # 相邻切片在边界处共享的时长（各向外扩一半）

    # -- 打分权重 --
    weight_proximity: float = 0.40
    weight_boundary: float = 0.50
    weight_overshoot: float = 0.10
    weight_tail_penalty: float = 0.60  # 切点后剩余不足 min_ms 时的惩罚强度

    # -- 文本与 ID --
    clean_text: bool = True
    clip_id_template: str = "{video_id}-c{index:04d}"
    stable_ids: bool = False
    max_chars: int = 2_000  # 单个切片文本上限（超出截断并标注）

    def validate(self) -> None:
        if not (0 < self.min_ms < self.max_ms):
            raise ValueError("需要 0 < min_ms < max_ms")
        if not (self.min_ms <= self.target_ms <= self.max_ms):
            raise ValueError("target_ms 必须落在 [min_ms, max_ms] 内")
        if self.hard_max_ms < self.max_ms:
            raise ValueError("hard_max_ms 不能小于 max_ms")
        if self.overlap_ms < 0:
            raise ValueError("overlap_ms 不能为负")
        if self.overlap_ms >= self.min_ms:
            raise ValueError("overlap_ms 必须小于 min_ms，否则相邻切片会互相吞并")
        if self.silence_threshold_ms < 0:
            raise ValueError("silence_threshold_ms 不能为负")
        if self.min_clip_ms <= 0:
            raise ValueError("min_clip_ms 必须为正")


# --------------------------------------------------------------------------- #
# 输出模型
# --------------------------------------------------------------------------- #
def _hms(timestamp_ms: int) -> str:
    """毫秒 → ``HH:MM:SS.mmm``（与媒体层保持同一格式，但不跨层依赖）。"""
    value = max(0, int(timestamp_ms))
    hours, rem = divmod(value, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


@dataclass(frozen=True, slots=True)
class Clip:
    """一个语音切片及其溯源信息。

    ``start_ms`` / ``end_ms`` 是**含重叠**的可播放区间；
    ``core_start_ms`` / ``core_end_ms`` 是不含重叠的纯净核心区间，
    检索与去重应以核心区间为准。
    """

    clip_id: str
    video_id: str
    index: int
    start_ms: int
    end_ms: int
    core_start_ms: int
    core_end_ms: int
    text: str
    core_text: str = ""
    keyframe_ids: tuple[str, ...] = ()
    segment_indices: tuple[int, ...] = ()
    sentence_count: int = 0
    char_count: int = 0
    truncated: bool = False

    # -- 呈现用属性 --
    @property
    def start_time(self) -> str:
        return _hms(self.start_ms)

    @property
    def end_time(self) -> str:
        return _hms(self.end_ms)

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def core_duration_ms(self) -> int:
        return max(0, self.core_end_ms - self.core_start_ms)

    @property
    def overlap_prev_ms(self) -> int:
        return max(0, self.core_start_ms - self.start_ms)

    @property
    def overlap_next_ms(self) -> int:
        return max(0, self.end_ms - self.core_end_ms)

    def with_keyframes(self, keyframe_ids: Sequence[str]) -> "Clip":
        return replace(self, keyframe_ids=tuple(keyframe_ids))

    def to_dict(self) -> dict[str, Any]:
        """对外载荷：首层即含需求规定的五个字段。"""
        return {
            # —— 需求规定字段 ——
            "clip_id": self.clip_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "keyframe_ids": list(self.keyframe_ids),
            # —— 溯源与工程字段 ——
            "video_id": self.video_id,
            "index": self.index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "core_start_ms": self.core_start_ms,
            "core_end_ms": self.core_end_ms,
            "core_start_time": _hms(self.core_start_ms),
            "core_end_time": _hms(self.core_end_ms),
            "core_duration_ms": self.core_duration_ms,
            "core_text": self.core_text or self.text,
            "overlap_prev_ms": self.overlap_prev_ms,
            "overlap_next_ms": self.overlap_next_ms,
            "segment_indices": list(self.segment_indices),
            "sentence_count": self.sentence_count,
            "char_count": self.char_count,
            "truncated": self.truncated,
        }


# --------------------------------------------------------------------------- #
# 字符级时间轴
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _Boundary:
    """一个候选切分点。"""

    t_ms: int
    kind: str
    weight: float


_BOUNDARY_WEIGHTS: Mapping[str, float] = {
    "silence": 1.00,
    "sentence": 0.85,
    "segment": 0.55,
    "soft": 0.30,
}


class _CharTimeline:
    """把转录片段铺平成「字符 → 时间区间」序列，支持任意区间取文与反查时间。"""

    __slots__ = ("chars", "starts", "ends", "seg_of", "seg_spans", "gaps", "total_ms")

    def __init__(self) -> None:
        self.chars: list[str] = []
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.seg_of: list[int] = []
        self.seg_spans: list[
            tuple[int, int, int]
        ] = []  # (seg_index, char_begin, char_end)
        self.gaps: list[tuple[int, int]] = []  # 段间静音区间
        self.total_ms: int = 0

    # -- 构建 --
    @classmethod
    def build(cls, segments: Sequence[RawSegment]) -> "_CharTimeline":
        timeline = cls()
        prev_end: int | None = None
        for seg in segments:
            text = seg.text
            if not text:
                continue
            char_begin = len(timeline.chars)
            spans = _assign_char_times(seg, text)
            for char, start_ms, end_ms in spans:
                timeline.chars.append(char)
                timeline.starts.append(start_ms)
                timeline.ends.append(end_ms)
                timeline.seg_of.append(seg.index)
            timeline.seg_spans.append((seg.index, char_begin, len(timeline.chars)))
            if prev_end is not None and seg.start_ms > prev_end:
                timeline.gaps.append((prev_end, seg.start_ms))
            prev_end = max(prev_end or 0, seg.end_ms)
        timeline.total_ms = max(prev_end or 0, max(timeline.ends, default=0))
        return timeline

    # -- 查询 --
    def __len__(self) -> int:
        return len(self.chars)

    def first_index_at_or_after(self, t_ms: int) -> int:
        """返回第一个 ``end_ms > t_ms`` 的字符下标（不存在则返回 ``len``）。"""
        lo, hi = 0, len(self.ends)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.ends[mid] <= t_ms:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def text_between(self, start_ms: int, end_ms: int) -> str:
        """取与 ``[start_ms, end_ms)`` 相交的字符并按需补空格。"""
        if end_ms <= start_ms or not self.chars:
            return ""
        begin = self.first_index_at_or_after(start_ms)
        out: list[str] = []
        prev_seg: int | None = None
        for i in range(begin, len(self.chars)):
            if self.starts[i] >= end_ms:
                break
            char = self.chars[i]
            seg = self.seg_of[i]
            if (
                out
                and prev_seg is not None
                and seg != prev_seg
                and _needs_space(out[-1], char)
            ):
                out.append(" ")
            out.append(char)
            prev_seg = seg
        return "".join(out)

    def nearest_boundary(self, t_ms: int, window_ms: int = 4_000) -> int | None:
        """找离 ``t_ms`` 最近且不超过 ``window_ms`` 的字符结束时刻。"""
        if not self.ends:
            return None
        index = self.first_index_at_or_after(t_ms)
        candidates: list[int] = []
        if index < len(self.ends):
            candidates.append(self.ends[index])
        if index > 0:
            candidates.append(self.ends[index - 1])
        if not candidates:
            return None
        best = min(candidates, key=lambda value: abs(value - t_ms))
        return best if abs(best - t_ms) <= window_ms else None

    def segment_indices_between(self, start_ms: int, end_ms: int) -> tuple[int, ...]:
        hits: list[int] = []
        for seg_index, char_begin, char_end in self.seg_spans:
            if char_begin >= char_end:
                continue
            seg_start = self.starts[char_begin]
            seg_end = self.ends[char_end - 1]
            if seg_end > start_ms and seg_start < end_ms:
                hits.append(seg_index)
        return tuple(dict.fromkeys(hits))


def _needs_space(prev: str, cur: str) -> bool:
    """拉丁文本跨段拼接时补一个空格；中文不补。"""
    return prev.isascii() and cur.isascii() and (prev.isalnum() and cur.isalnum())


def _assign_char_times(seg: RawSegment, text: str) -> list[tuple[str, int, int]]:
    """为一段文本的每个字符分配时间区间（词时间戳优先，否则线性插值）。"""
    count = len(text)
    if count == 0:
        return []
    duration = max(1, seg.duration_ms)

    word_texts = [word.text for word in seg.words]
    joined = "".join(word_texts)
    if seg.words and joined and _normalize_ws(joined) == _normalize_ws(text):
        # 词级时间戳与整段文本对齐 → 用词区间细分到字符
        result: list[tuple[str, int, int]] = []
        cursor = 0
        for word in seg.words:
            piece = word.text
            for offset, char in enumerate(piece):
                if cursor >= count:
                    break
                span = max(1, word.end_ms - word.start_ms)
                start = word.start_ms + int(round(span * offset / max(1, len(piece))))
                end = word.start_ms + int(
                    round(span * (offset + 1) / max(1, len(piece)))
                )
                result.append((char, start, max(start, end)))
                cursor += 1
        if len(result) == count:
            return result

    # 线性插值兜底
    return [
        (
            char,
            seg.start_ms + int(round(duration * i / count)),
            seg.start_ms + int(round(duration * (i + 1) / count)),
        )
        for i, char in enumerate(text)
    ]


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", "", text)


# --------------------------------------------------------------------------- #
# 主类
# --------------------------------------------------------------------------- #
class Chunker:
    """自适应语音切片器。

    典型用法::

        chunker = Chunker(ChunkConfig(min_ms=10_000, max_ms=30_000))
        clips = chunker.chunk("demo", whisper_segments)
        for clip in clips:
            print(clip.clip_id, clip.start_time, clip.end_time, clip.text[:20])
    """

    def __init__(
        self,
        config: ChunkConfig | None = None,
        *,
        logger_: logging.Logger | None = None,
    ) -> None:
        self._cfg = config or ChunkConfig()
        self._cfg.validate()
        self._log = logger_ or logger

    @property
    def config(self) -> ChunkConfig:
        return self._cfg

    # ------------------------------------------------------------------ #
    # 公开接口
    # ------------------------------------------------------------------ #
    def chunk(
        self,
        video_id: str,
        segments: Iterable[Any],
        *,
        keyframe_map: Mapping[str, Sequence[str]] | None = None,
        total_duration_ms: int | None = None,
    ) -> list[Clip]:
        """把转录片段合并为 10~30s 切片。

        Args:
            video_id: 溯源 ID。
            segments: faster-whisper 片段（对象或 dict 均可）。
            keyframe_map: 可选的 ``clip_id → keyframe_id`` 预绑定表。
            total_duration_ms: 视频总时长；缺省时取最后一段的结束时间。
        """
        if not video_id or not str(video_id).strip():
            raise ChunkError("video_id 不能为空（溯源要求）")

        normalized = self._normalize_segments(segments)
        if not normalized:
            self._log.warning(
                "video_id=%s 没有可用的转录文本，返回空切片列表", video_id
            )
            return []

        timeline = _CharTimeline.build(normalized)
        if not timeline.chars:
            self._log.warning("video_id=%s 转录文本为空，返回空切片列表", video_id)
            return []

        # 切分只铺到**最后一个字符结束**：视频尾部常有长音乐/静音（字幕结束后
        # 还剩几分钟），按总时长继续切会产出一串空文本切片进 BM25/向量库。
        # 尾部的视觉覆盖由 _materialize 里"末片 end=end_limit"兜住。
        split_limit = max(timeline.total_ms, max(seg.end_ms for seg in normalized))
        end_limit = split_limit
        if total_duration_ms and total_duration_ms > 0:
            end_limit = max(end_limit, int(total_duration_ms))

        cores = self._split(timeline, split_limit)
        clips = self._materialize(video_id, timeline, cores, end_limit)

        if keyframe_map:
            clips = [
                clip.with_keyframes(keyframe_map.get(clip.clip_id, ()))
                if clip.clip_id in keyframe_map
                else clip
                for clip in clips
            ]

        self._log.info(
            "切片完成 video=%s segments=%d clips=%d 平均时长=%.1fs",
            video_id,
            len(normalized),
            len(clips),
            (sum(c.core_duration_ms for c in clips) / len(clips) / 1000.0)
            if clips
            else 0.0,
        )
        return clips

    def to_json(self, clips: Sequence[Clip]) -> list[dict[str, Any]]:
        """便捷导出：切片列表 → JSON 可序列化结构。"""
        return [clip.to_dict() for clip in clips]

    # ------------------------------------------------------------------ #
    # 输入规范化
    # ------------------------------------------------------------------ #
    def _normalize_segments(self, segments: Iterable[Any]) -> list[RawSegment]:
        """排序、清洗、修坏时间、合并重叠。"""
        parsed: list[RawSegment] = []
        for index, item in enumerate(segments):
            try:
                seg = RawSegment.from_any(item, index=index)
            except Exception as exc:  # noqa: BLE001 - 脏数据不应中断整批
                self._log.warning("跳过无法解析的转录片段 #%d: %s", index, exc)
                continue
            text = _clean_text(seg.text) if self._cfg.clean_text else seg.text
            start = int(seg.start_ms)
            end = int(seg.end_ms)
            if start < 0:
                start = 0
            if end <= start:
                # 时间戳缺失或回退：按字符速率估一个宽度
                est = max(300, int(len(text) / 4.5 * 1000)) if text else 500
                end = start + est
            if not text:
                continue
            parsed.append(
                RawSegment(
                    index=seg.index,
                    start_ms=start,
                    end_ms=end,
                    text=text,
                    words=seg.words,
                )
            )

        if not parsed:
            return []

        parsed.sort(key=lambda s: (s.start_ms, s.index))

        merged: list[RawSegment] = []
        for seg in parsed:
            if not merged:
                merged.append(seg)
                continue
            prev = merged[-1]
            if seg.start_ms < prev.end_ms:
                # 时间重叠：截断到上一段结束后，避免同一句被重复计入。
                # 文本必须跟着开头一起裁，否则字符-时间映射整体前移、溯源错位。
                new_start = prev.end_ms
                if seg.end_ms <= new_start:
                    self._log.warning(
                        "片段 %s 完全被前一段覆盖（%d~%d），已丢弃",
                        seg.index,
                        seg.start_ms,
                        seg.end_ms,
                    )
                    continue
                merged.append(
                    RawSegment(
                        index=seg.index,
                        start_ms=new_start,
                        end_ms=seg.end_ms,
                        text=_trim_text_for_start(seg, new_start),
                        words=_trim_words_for_start(seg, new_start),
                    )
                )
                continue
            merged.append(seg)
        return merged

    # ------------------------------------------------------------------ #
    # 切分
    # ------------------------------------------------------------------ #
    def _collect_boundaries(self, timeline: _CharTimeline) -> list[_Boundary]:
        """汇总所有候选切分点，同一时刻只保留权重最高的一类。"""
        best: dict[int, _Boundary] = {}

        def push(t_ms: int, kind: str) -> None:
            if t_ms <= 0:
                return
            weight = _BOUNDARY_WEIGHTS[kind]
            current = best.get(t_ms)
            if current is None or weight > current.weight:
                best[t_ms] = _Boundary(t_ms=t_ms, kind=kind, weight=weight)

        cfg = self._cfg
        chars = timeline.chars

        if cfg.use_sentence or cfg.use_soft_breaks:
            for i, char in enumerate(chars):
                t_ms = timeline.ends[i]
                if cfg.use_sentence and _is_sentence_end(chars, i, char):
                    push(t_ms, "sentence")
                elif cfg.use_soft_breaks and char in _SOFT_BREAKS:
                    push(t_ms, "soft")

        if cfg.use_segment_boundary:
            for seg_index, char_begin, char_end in timeline.seg_spans:
                if char_end > char_begin:
                    push(timeline.ends[char_end - 1], "segment")

        if cfg.use_silence:
            for gap_start, gap_end in timeline.gaps:
                if gap_end - gap_start < cfg.silence_threshold_ms:
                    continue
                offset = min(cfg.silence_inner_offset_ms, (gap_end - gap_start) // 2)
                push(gap_start + offset, "silence")

        return sorted(best.values(), key=lambda item: item.t_ms)

    def _split(self, timeline: _CharTimeline, end_limit: int) -> list[tuple[int, int]]:
        """贪心切分，返回核心区间 ``[(core_start, core_end), ...]``。"""
        cfg = self._cfg
        boundaries = self._collect_boundaries(timeline)
        cores: list[tuple[int, int]] = []
        cursor = 0
        guard = 0

        while cursor < end_limit:
            guard += 1
            if guard > 100_000:  # pragma: no cover - 防御性死循环保护
                raise ChunkError("切片循环次数异常，输入时间轴可能损坏")
            remaining = end_limit - cursor
            if remaining <= cfg.max_ms:
                # 剩余量本身就在合法窗口内（含极短残尾）→ 一次收尾，避免切碎
                core_end = end_limit
            else:
                core_end = self._pick_cut(
                    cursor, end_limit, boundaries
                ) or self._hard_cut(timeline, cursor, end_limit)
            # 强制前进，杜绝零长度切片
            core_end = max(core_end, min(end_limit, cursor + 1_000))
            core_end = min(core_end, end_limit)
            cores.append((cursor, core_end))
            cursor = core_end

        # 尾部残片并入上一片：<min_clip_ms 的独立切片没有检索价值
        while len(cores) >= 2 and cores[-1][1] - cores[-1][0] < cfg.min_clip_ms:
            prev_start, _ = cores[-2]
            _, last_end = cores[-1]
            cores[-2:] = [(prev_start, last_end)]

        return cores

    def _pick_cut(
        self, start_ms: int, end_limit: int, boundaries: Sequence[_Boundary]
    ) -> int:
        """在 [min_ms, max_ms] 窗口内挑分数最高的候选切点。"""
        cfg = self._cfg
        best_legal: tuple[float, int] = (-math.inf, 0)
        best_any: tuple[float, int] = (-math.inf, 0)
        span = max(1, cfg.max_ms - cfg.min_ms)
        for boundary in boundaries:
            if boundary.t_ms <= start_ms:
                continue
            duration = boundary.t_ms - start_ms
            if duration < cfg.min_ms:
                continue
            if duration > cfg.max_ms:
                break
            proximity = 1.0 - abs(duration - cfg.target_ms) / span
            overshoot = max(
                0.0, (duration - cfg.target_ms) / max(1, cfg.max_ms - cfg.target_ms)
            )
            score = (
                cfg.weight_proximity * proximity
                + cfg.weight_boundary * boundary.weight
                - cfg.weight_overshoot * overshoot
            )
            # 前瞻惩罚：切在这里会让剩下的一小截短于 min_ms（软下限），
            # 除非当前切片已经撑到 max_ms（不得不切），否则强烈回避。
            tail = end_limit - boundary.t_ms
            legal = tail >= cfg.min_ms or duration >= cfg.max_ms
            if 0 < tail < cfg.min_ms:
                score -= cfg.weight_tail_penalty * (1.0 - tail / max(1, cfg.min_ms))
            if score > best_any[0]:
                best_any = (score, boundary.t_ms)
            if legal and score > best_legal[0]:
                best_legal = (score, boundary.t_ms)
        # 优先选不会留下非法短尾的切点；实在没有才退而求其次
        return best_legal[1] or best_any[1]

    def _hard_cut(self, timeline: _CharTimeline, start_ms: int, end_limit: int) -> int:
        """窗口内无候选（超长无标点语音）：按目标时长硬切并吸附到边界。"""
        cfg = self._cfg
        ideal = min(start_ms + cfg.target_ms, end_limit)
        ideal = min(ideal, start_ms + cfg.hard_max_ms)

        # 落在静音区内 → 直接取，播放最平滑
        for gap_start, gap_end in timeline.gaps:
            if gap_start <= ideal <= gap_end:
                return ideal

        snapped = timeline.nearest_boundary(ideal, window_ms=max(500, cfg.min_ms // 2))
        if snapped is not None and snapped - start_ms >= max(1_000, cfg.min_clip_ms):
            return min(snapped, end_limit)

        floor_ms = start_ms + max(1_000, cfg.min_clip_ms)
        if ideal <= floor_ms:
            return min(end_limit, floor_ms)
        return ideal

    # ------------------------------------------------------------------ #
    # 物化
    # ------------------------------------------------------------------ #
    def _materialize(
        self,
        video_id: str,
        timeline: _CharTimeline,
        cores: Sequence[tuple[int, int]],
        end_limit: int,
    ) -> list[Clip]:
        cfg = self._cfg
        half_overlap = cfg.overlap_ms // 2
        clips: list[Clip] = []
        for index, (core_start, core_end) in enumerate(cores):
            duration = core_end - core_start
            # 短切片不允许把重叠撑得过大（最多占核心时长的 1/4）
            overlap = min(half_overlap, max(0, duration // 4))
            start = max(0, core_start - overlap) if index > 0 else 0
            end = (
                min(end_limit, core_end + overlap)
                if index < len(cores) - 1
                else end_limit
            )
            start = min(start, core_start)
            end = max(end, core_end)

            text = (
                _clean_text(timeline.text_between(start, end))
                if cfg.clean_text
                else timeline.text_between(start, end)
            )
            core_text = (
                _clean_text(timeline.text_between(core_start, core_end))
                if cfg.clean_text
                else timeline.text_between(core_start, core_end)
            )
            truncated = False
            if len(core_text) > cfg.max_chars:
                core_text = core_text[: cfg.max_chars]
                truncated = True
            if len(text) > cfg.max_chars * 2:
                text = text[: cfg.max_chars * 2]
                truncated = True

            clip_id = self._make_clip_id(
                video_id, index, core_start, core_end, core_text
            )
            clips.append(
                Clip(
                    clip_id=clip_id,
                    video_id=video_id,
                    index=index,
                    start_ms=start,
                    end_ms=end,
                    core_start_ms=core_start,
                    core_end_ms=core_end,
                    text=text,
                    core_text=core_text,
                    segment_indices=timeline.segment_indices_between(
                        core_start, core_end
                    ),
                    sentence_count=sum(
                        1
                        for ch in core_text
                        if ch in _CJK_SENTENCE_ENDS or ch in _LATIN_SENTENCE_ENDS
                    ),
                    char_count=len(core_text),
                    truncated=truncated,
                )
            )
        return clips

    def _make_clip_id(
        self, video_id: str, index: int, start: int, end: int, text: str
    ) -> str:
        if self._cfg.stable_ids:
            digest = hashlib.blake2b(
                f"{video_id}|{start}|{end}|{text}".encode("utf-8"), digest_size=5
            ).hexdigest()
            return f"{video_id}-{digest}"
        try:
            return self._cfg.clip_id_template.format(
                video_id=video_id, index=index, start=start, end=end
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise ChunkError(
                f"clip_id_template 非法: {self._cfg.clip_id_template!r}"
            ) from exc


# --------------------------------------------------------------------------- #
# 文本工具
# --------------------------------------------------------------------------- #
def _trim_words_for_start(seg: "RawSegment", new_start: int) -> tuple["Word", ...]:
    """词级裁剪：只留结束时间晚于新起点的词（与文本裁剪保持同源）。"""
    if not seg.words:
        return ()
    return tuple(w for w in seg.words if w.end_ms > new_start)


def _trim_text_for_start(seg: "RawSegment", new_start: int) -> str:
    """重叠截断后同步裁掉文本开头，避免字符-时间映射整体前移。

    有词级时间戳时按词精确裁；没有时按时间比例近似 —— 近似会有少许偏差，
    但比"整段文本右移"（前几秒的字挂到后面的时间上）好一个量级。
    """
    if seg.words:
        kept = "".join(w.text for w in _trim_words_for_start(seg, new_start)).strip()
        if kept:
            return kept
        return seg.text  # 词全在被截区间但文本非空：宁可疑对齐，不丢内容
    total = max(1, seg.end_ms - seg.start_ms)
    ratio = max(0.0, min(0.95, (new_start - seg.start_ms) / total))
    if ratio <= 0.02:
        return seg.text
    return seg.text[int(len(seg.text) * ratio) :].lstrip()


def _clean_text(text: str) -> str:
    """去零宽字符、统一空白、全角空格转半角、去首尾空白。"""
    if not text:
        return ""
    cleaned = text.translate(_INVISIBLE)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    cleaned = unicodedata.normalize("NFC", cleaned)
    return cleaned.strip()


def _is_sentence_end(chars: Sequence[str], index: int, char: str) -> bool:
    """判断该字符是否为句末，并排除数字小数点与缩写点。"""
    if char in _CJK_SENTENCE_ENDS:
        return True
    if char not in _LATIN_SENTENCE_ENDS:
        return False
    prev = chars[index - 1] if index > 0 else ""
    nxt = chars[index + 1] if index + 1 < len(chars) else ""
    if char == ".":
        # 3.5 / 1.2.3 → 非句末
        if prev.isdigit() and nxt.isdigit():
            return False
        # e.g / U.S / Mr. → 非句末（前后都是字母且无空格）
        if prev.isalpha() and nxt.isalpha():
            return False
        # 连续点（...）只认最后一个
        if prev == ".":
            return False
    return True


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #
def segments_from_whisper(segments: Iterable[Any]) -> list[RawSegment]:
    """直接转换 faster-whisper 的 ``segments``/``info.segments``。"""
    return [RawSegment.from_any(item, index=i) for i, item in enumerate(segments)]


def bind_keyframes(
    clips: Sequence[Clip],
    keyframes: Iterable[tuple[str, int]],
    *,
    tolerance_ms: int = 0,
) -> list[Clip]:
    """把 ``(keyframe_id, timestamp_ms)`` 绑定到覆盖它的切片核心区间。

    落在重叠区（不属于任何核心区间）的关键帧，会归给时间最近且距离 ≤
    ``tolerance_ms`` 的切片；``tolerance_ms`` 为 0 时取最近的一个核心边界。
    """
    frames = [(str(fid), int(ts)) for fid, ts in keyframes]
    if not frames:
        return list(clips)

    mapping: dict[str, list[str]] = {clip.clip_id: [] for clip in clips}
    for frame_id, ts in sorted(frames, key=lambda item: item[1]):
        target: Clip | None = None
        for clip in clips:
            if clip.core_start_ms <= ts < clip.core_end_ms:
                target = clip
                break
        if target is None:
            # 落在重叠区或边界外：取最近的核心区间
            best_distance = math.inf
            for clip in clips:
                if clip.core_start_ms <= ts <= clip.core_end_ms:
                    distance = 0
                else:
                    distance = min(
                        abs(ts - clip.core_start_ms), abs(ts - clip.core_end_ms)
                    )
                if distance < best_distance:
                    best_distance = distance
                    target = clip
            if target is not None and tolerance_ms > 0 and best_distance > tolerance_ms:
                target = None
        if target is not None:
            mapping[target.clip_id].append(frame_id)

    return [clip.with_keyframes(mapping.get(clip.clip_id, ())) for clip in clips]
