"""知影 ZhiYing —— 长视频的章节切分与渲染。

为什么需要单独一层
------------------
一段 1 小时的视频压成「一屏摘要」，等于把 60 分钟的证据塞进 800 字：要么模型
只读了开头（按相关度取前 N 条的通病），要么它把没读到的部分用先验补齐 ——
后者更糟，因为读起来毫无破绽。

章节层把长视频切成若干分钟级区间，每区间**独立**出摘要，各自挂着自己的
``clip_ids``；总摘要拿章节摘要当输入，不直接啃原始证据。这样：

- 任何一分钟的内容都有归属，不会因为"没被检索到"而消失；
- 每段摘要都能点回原始切片，溯源不断；
- 单章预算与视频总长解耦，2 小时视频和 2 分钟视频走同一套代码。

本模块只做**纯函数**：分组、编号、渲染 markdown。调模型在
:class:`backend.pipeline.orchestrator.Orchestrator` 里，便于单测这里的分组逻辑。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from backend.pipeline.context_builder import format_hms

__all__ = [
    "DEFAULT_CHAPTER_SECONDS",
    "MAX_CHAPTERS",
    "Chapter",
    "ChapterPlan",
    "chapters_markdown",
    "plan_chapters",
]

#: 默认章节长度。5 分钟是"一次能读完、又不至于碎"的经验值：再短会让章节数
#: 爆炸（2 小时视频 1 分钟一章 = 120 次模型调用），再长单章就又变回顾全文。
DEFAULT_CHAPTER_SECONDS: Final[int] = 300

#: 章节数上限。超过就说明视频太长或章节切得太碎，这时候应该降级成
#: 「只给章节边界 + 关键帧」而不是硬跑 200 次推理。
MAX_CHAPTERS: Final[int] = 40


@dataclass(frozen=True, slots=True)
class ChapterPlan:
    """一章的时间范围与它覆盖的切片（还没摘，纯结构）。"""

    index: int
    start_ms: int
    end_ms: int
    clip_ids: tuple[str, ...] = ()

    @property
    def time_range(self) -> str:
        return f"{format_hms(self.start_ms)}–{format_hms(self.end_ms)}"

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def clip_count(self) -> int:
        return len(self.clip_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "time_range": self.time_range,
            "clip_ids": list(self.clip_ids),
            "clip_count": self.clip_count,
        }


@dataclass(frozen=True, slots=True)
class Chapter:
    """一章的实际产出。``summary`` 为空表示这一章没跑成功（降级）。"""

    index: int
    start_ms: int
    end_ms: int
    summary: str = ""
    title: str = ""
    clip_ids: tuple[str, ...] = ()
    backend: str = ""
    degraded: bool = False
    notes: tuple[str, ...] = field(default=())

    @property
    def time_range(self) -> str:
        return f"{format_hms(self.start_ms)}–{format_hms(self.end_ms)}"

    @property
    def clip_count(self) -> int:
        return len(self.clip_ids)

    @property
    def heading(self) -> str:
        """``第 3 章 00:10:00.000–00:15:00.000 注意力机制`` 之类的标题行。"""
        label = (
            self.title.strip() or _first_clause(self.summary) or f"第 {self.index} 段"
        )
        return f"第 {self.index} 章 {self.time_range}  {label}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "time_range": self.time_range,
            "title": self.title,
            "summary": self.summary,
            "clip_ids": list(self.clip_ids),
            "clip_count": self.clip_count,
            "backend": self.backend,
            "degraded": self.degraded,
            "notes": list(self.notes),
        }


def plan_chapters(
    clips: Iterable[Any],
    *,
    chapter_seconds: int = DEFAULT_CHAPTER_SECONDS,
    duration_ms: int = 0,
    max_chapters: int = MAX_CHAPTERS,
) -> list[ChapterPlan]:
    """按固定时长把切片分进章节。

    Parameters
    ----------
    clips:
        任何带 ``clip_id`` / ``start_ms`` / ``end_ms`` 的对象或 dict。
    chapter_seconds:
        目标章节长度。<=0 表示不分章（整段算一章）。
    duration_ms:
        视频总时长，用来补齐最后一章的 ``end_ms``（尾部无语音时切片可能提前结束，
        不补的话"最后 3 分钟"在时间线上是空白）。

    Notes
    -----
    以 ``start_ms`` 归章，不按 ``end_ms`` 切：一个 4:58 开始的切片算第三章，
    哪怕它跨到了 5:02。跨章重复计一次比"同一段内容在两章里都出现"好查。
    """
    spans: list[tuple[str, int, int]] = []
    for raw in clips:
        clip_id = _get(raw, "clip_id")
        if not clip_id:
            continue
        start = _as_int(_get(raw, "start_ms"))
        end = max(start, _as_int(_get(raw, "end_ms")))
        spans.append((str(clip_id), start, end))
    if not spans:
        return []
    spans.sort(key=lambda item: (item[1], item[2]))

    span_ms = max(0, int(chapter_seconds)) * 1000
    if span_ms <= 0:
        span_ms = max(1, duration_ms or spans[-1][2])

    buckets: dict[int, list[tuple[str, int, int]]] = {}
    for item in spans:
        buckets.setdefault(item[1] // span_ms, []).append(item)

    plans: list[ChapterPlan] = []
    limit = max(1, int(max_chapters))
    for order, bucket_index in enumerate(sorted(buckets)):
        if order >= limit:
            # 超上限：把剩下的全部并进最后一章，宁可一章长也别丢内容
            tail = [
                item
                for idx in sorted(buckets)
                if idx >= bucket_index
                for item in buckets[idx]
            ]
            last = plans[-1]
            merged = ChapterPlan(
                index=last.index,
                start_ms=last.start_ms,
                end_ms=max(last.end_ms, max(item[2] for item in tail)),
                clip_ids=last.clip_ids + tuple(item[0] for item in tail),
            )
            plans[-1] = merged
            break
        items = buckets[bucket_index]
        plans.append(
            ChapterPlan(
                index=len(plans) + 1,
                start_ms=bucket_index * span_ms,
                end_ms=(bucket_index + 1) * span_ms,
                clip_ids=tuple(item[0] for item in items),
            )
        )

    if duration_ms > 0 and plans:
        last = plans[-1]
        if duration_ms > last.end_ms:
            plans[-1] = ChapterPlan(
                index=last.index,
                start_ms=last.start_ms,
                end_ms=duration_ms,
                clip_ids=last.clip_ids,
            )
        else:
            # 章节边界越过视频末尾：收窄，别让"第 4 章"越过片尾
            plans[-1] = ChapterPlan(
                index=last.index,
                start_ms=last.start_ms,
                end_ms=max(last.start_ms, min(last.end_ms, duration_ms)),
                clip_ids=last.clip_ids,
            )
    return plans


def chapters_markdown(
    title: str,
    chapters: Sequence[Chapter],
    *,
    total_summary: str = "",
    keywords: Sequence[str] = (),
    duration_ms: int = 0,
) -> str:
    """把章节渲染成 markdown（对齐需求文档里的"结构化总结"输出）。"""
    lines: list[str] = [f"# {title or '视频总结'}"]
    meta = []
    if duration_ms > 0:
        meta.append(f"时长 {format_hms(duration_ms)}")
    if chapters:
        meta.append(f"{len(chapters)} 章")
        covered = sum(ch.clip_count for ch in chapters)
        if covered:
            meta.append(f"{covered} 个切片")
    if meta:
        lines.append("")
        lines.append("　·　".join(meta))

    if total_summary.strip():
        lines.append("")
        lines.append("## 总结")
        lines.append("")
        lines.append(total_summary.strip())

    if keywords:
        lines.append("")
        lines.append("## 关键词")
        lines.append("")
        lines.append("、".join(k.strip() for k in keywords if k.strip()))

    for chapter in chapters:
        lines.append("")
        lines.append(f"## {chapter.heading}")
        lines.append("")
        body = chapter.summary.strip()
        if not body:
            body = "（本章未生成摘要）"
            if chapter.notes:
                body += f"　{chapter.notes[0]}"
        lines.append(body)
        # 降级的章节必须自报家门：读者看到一段没标来源的文字，会当成完整
        # 摘要去引用；写清"这是证据摘录不是模型生成"，责任边界才清楚。
        flags: list[str] = []
        if chapter.degraded:
            flags.append("本章降级")
        if chapter.backend and chapter.backend != "local":
            flags.append(f"来源 {chapter.backend}")
        if flags and chapter.clip_ids:
            lines.append("")
            lines.append(
                f"<sub>证据 {chapter.clip_count} 段：{', '.join(chapter.clip_ids)}"
                f"　·　{'　·　'.join(flags)}</sub>"
            )
        elif flags:
            lines.append("")
            lines.append(f"<sub>{'　·　'.join(flags)}</sub>")
        elif chapter.clip_ids:
            lines.append("")
            lines.append(
                f"<sub>证据 {chapter.clip_count} 段：{', '.join(chapter.clip_ids)}</sub>"
            )
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# 内部
# =============================================================================


def _get(obj: Any, name: str, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _first_clause(text: str, limit: int = 18) -> str:
    """从摘要首句里抠一个短语当章节标题（模型没给标题时的兜底）。"""
    stripped = (text or "").strip().replace("\n", " ")
    if not stripped:
        return ""
    for stop in ("。", "，", "；", "！", "？", ".", ",", ";", ":", "："):
        head, sep, _ = stripped.partition(stop)
        if sep and head.strip():
            stripped = head.strip()
            break
    return stripped[:limit]
