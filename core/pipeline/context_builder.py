"""知影 ZhiYing —— 上下文组装（检索结果 → LLM 可读的证据块）。

解决什么问题
------------
需求文档第 4 章遗留问题之一："超长视频上下文组装（token 预算/去重）未定"。
1 小时视频切出 120~200 个切片，检索 Top-20 全塞进去会：

* 超出模型上下文（7B 的 ``n_ctx=8192``，中文约 5000 token 可用）；
* 同一段内容被重叠切片命中三次，白烧预算；
* 证据顺序与模型阅读顺序不一致，导致引用编号错乱。

这一层负责**在预算内挑最有用的证据，并给每条编号**，让 LLM 的输出可以直接
落成 ``[1]``/``[2]`` 形式的时间戳引用。

三条规则
--------
1. **去重**：文本前缀相同的条目视为重复（重叠切片导致的），保留分数高的。
2. **预算**：先按 ``max_tokens`` 折算成字符预算，再逐条分配；单条超长截断，
   而不是整条丢弃 —— 半条证据仍然有用。
3. **编号稳定**：编号按最终输出顺序从 1 开始，``ContextItem.index`` 与文本里的
   ``[n]`` 一一对应，供 :class:`core.pipeline.orchestrator` 回填 ``refs``。

两种排序
--------
* ``build()``：**按相关度**降序（问答场景，最相关的放最前面）。
* ``build_summary()``：**按时间**升序（总结场景，模型需要顺着时间线读）。

用法
----
>>> cb = ContextBuilder(ContextBudget(max_chars=2000))
>>> built = cb.build(items, "切片长度是怎么定的", video_title="演示视频")
>>> built.items[0].index, built.items[0].time_range
(1, '00:00:20.100–00:00:42.300')
>>> built.text.splitlines()[0]
'[1] 《演示视频》 00:00:20.100–00:00:42.300'
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

logger = logging.getLogger("zhiying.pipeline.context")

__all__ = [
    "BuiltContext",
    "ContextBudget",
    "ContextBuilder",
    "ContextItem",
    "EvidenceLike",
    "format_hms",
]

#: 中文平均 1.6 字符 / token（含标点）。保守取 1.5，宁可少放证据。
DEFAULT_CHARS_PER_TOKEN: Final[float] = 1.5


# =============================================================================
# 数据结构
# =============================================================================


class EvidenceLike(Protocol):
    """检索结果的最小接口（``hybrid_search.FusedItem`` 满足它）。"""

    @property
    def clip_id(self) -> str: ...

    @property
    def start_ms(self) -> int: ...

    @property
    def end_ms(self) -> int: ...


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """上下文预算。

    ``max_tokens`` 用于换算字符预算；``max_chars`` 是硬上限，两者取小。
    """

    max_tokens: int = 4000
    max_chars: int = 6000
    max_items: int = 8
    max_chars_per_item: int = 900
    reserve_chars: int = 300
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
    dedupe_prefix_chars: int = 40
    include_keyframes: bool = True
    include_scores: bool = True
    # 时间线组织：编号仍按相关度排（引用不失配），但正文按时间顺序铺，
    # 相邻片段合并成「连续片段」标头 —— 模型读到的才是一条时间线，
    # 而不是一堆按分数打散的孤立片段。
    timeline_order: bool = True
    timeline_gap_ms: int = 3000
    # 字节预算（UTF-8）。字符数在中英混排下不可靠：一个汉字 3 字节、一个
    # ASCII 字符 1 字节，同样 6000 "字符" 的实际体积能差三倍。填 0 表示
    # 退化为 max_chars×3 的保守上限（纯中文时恰好等值）。
    max_bytes: int = 0

    def validate(self) -> None:
        if self.max_tokens < 64:
            raise ValueError(f"max_tokens 过小: {self.max_tokens}")
        if self.max_chars < 100:
            raise ValueError(f"max_chars 过小: {self.max_chars}")
        if self.max_items < 1:
            raise ValueError(f"max_items 必须 >= 1: {self.max_items}")
        if self.max_chars_per_item < 50:
            raise ValueError(f"max_chars_per_item 过小: {self.max_chars_per_item}")
        if self.reserve_chars < 0 or self.reserve_chars >= self.max_chars:
            raise ValueError(f"reserve_chars 非法: {self.reserve_chars}")
        if self.max_bytes < 0:
            raise ValueError(f"max_bytes 非法: {self.max_bytes}")

    @property
    def effective_chars(self) -> int:
        """真正可用于证据正文的字符数。"""
        by_tokens = int(self.max_tokens * self.chars_per_token)
        return max(0, min(self.max_chars, by_tokens) - self.reserve_chars)

    @property
    def effective_bytes(self) -> int:
        """真正可用于证据正文的 UTF-8 字节数。

        这是上下文的**精确**上限：拼出来的文本按 ``len(text.encode())``
        实测比对，不再靠 ``chars_per_token`` 猜 token 数。
        """
        ceiling = self.effective_chars * 3
        if self.max_bytes <= 0:
            return ceiling
        return min(self.max_bytes, ceiling)


@dataclass(slots=True)
class ContextItem:
    """一条写进上下文的证据（编号 + 定位 + 正文 + 关键帧）。"""

    index: int
    clip_id: str
    video_id: str
    start_ms: int
    end_ms: int
    text: str
    score: float = 0.0
    keyframe_paths: tuple[str, ...] = ()
    keyframe_ids: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()
    truncated: bool = False
    source_rank: int = 0

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def time_range(self) -> str:
        return f"{format_hms(self.start_ms)}–{format_hms(self.end_ms)}"

    def header(self, *, video_title: str = "", include_score: bool = True) -> str:
        parts = [f"[{self.index}]"]
        if video_title:
            parts.append(f"《{video_title}》")
        parts.append(self.time_range)
        extras: list[str] = []
        if include_score:
            extras.append(f"score={self.score:.3f}")
        extras.append(f"clip={self.clip_id}")
        if self.channels:
            extras.append("+".join(self.channels))
        if extras:
            parts.append(f"（{'，'.join(extras)}）")
        return " ".join(parts)

    def render(self, *, video_title: str = "", include_score: bool = True,
               include_keyframes: bool = True) -> str:
        lines = [self.header(video_title=video_title, include_score=include_score), self.text]
        if include_keyframes and self.keyframe_paths:
            lines.append(f"关键帧: {', '.join(self.keyframe_paths)}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "clip_id": self.clip_id,
            "video_id": self.video_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "time_range": self.time_range,
            "text": self.text,
            "score": self.score,
            "keyframe_paths": list(self.keyframe_paths),
            "keyframe_ids": list(self.keyframe_ids),
            "channels": list(self.channels),
            "truncated": self.truncated,
        }


@dataclass(slots=True)
class BuiltContext:
    """组装结果：可直接喂给 LLM，也可序列化进报告。"""

    text: str
    items: list[ContextItem] = field(default_factory=list)
    used_chars: int = 0
    budget_chars: int = 0
    dropped: list[dict[str, Any]] = field(default_factory=list)
    query: str = ""
    video_title: str = ""
    degraded: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.items)

    @property
    def refs(self) -> list[dict[str, Any]]:
        """可写进回答的引用列表（编号 → 时间区间 → clip → 原文）。

        ``text`` 必须在这里带上：引用除了"定位到哪一段"，还要能让人当场核对
        结论对应的原话（需求文档：每条结论至少 1 条原始证据）。
        """
        return [
            {
                "index": item.index,
                "clip_id": item.clip_id,
                "video_id": item.video_id,
                "start_ms": item.start_ms,
                "end_ms": item.end_ms,
                "time_range": item.time_range,
                "text": item.text,
                "score": item.score,
                "keyframe_paths": list(item.keyframe_paths),
                "keyframe_ids": list(item.keyframe_ids),
            }
            for item in self.items
        ]

    @property
    def chars_left(self) -> int:
        return max(0, self.budget_chars - self.used_chars)

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "query": self.query,
            "video_title": self.video_title,
            "used_chars": self.used_chars,
            "budget_chars": self.budget_chars,
            "item_count": len(self.items),
            "degraded": self.degraded,
            "notes": list(self.notes),
            "dropped": list(self.dropped),
            "items": [item.to_dict() for item in self.items],
            "refs": self.refs,
        }
        if include_text:
            data["text"] = self.text
        return data


# =============================================================================
# 工具
# =============================================================================


def format_hms(ms: int | float) -> str:
    """毫秒 → ``HH:MM:SS.mmm``（需求文档对外时间格式）。"""
    total = max(0, int(round(float(ms))))
    hours, rem = divmod(total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def utf8_prefix(text: str, byte_limit: int) -> str:
    """按 **UTF-8 字节**截断，绝不切坏多字节字符。

    ``text[:n]`` 按字符切，中文安全；但预算一旦以字节计，就必须按字节切。
    用 ``errors="ignore"`` 让被切在中间的那个汉字整字丢弃，而不是变成 ``?``。

        >>> utf8_prefix("知影", 3)
        '知'
        >>> utf8_prefix("知影", 100)
        '知影'
    """
    if byte_limit <= 0:
        return ""
    blob = (text or "").encode("utf-8")
    if len(blob) <= byte_limit:
        return text or ""
    return blob[:byte_limit].decode("utf-8", errors="ignore")


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(v) for v in value if v is not None and str(v).strip())
    return (str(value),)


def _normalize_for_dedupe(text: str, prefix: int) -> str:
    """去重键：去掉标点与空白后的前 N 个字符。"""
    cleaned = re.sub(r"[\s，。！？；：、,.!?;:\"'“”‘’（）()\[\]]+", "", text or "")
    return cleaned[: max(1, prefix)]


def spread_order(count: int) -> list[int]:
    """预算只够 N 条时，按「首 → 尾 → 中 → 中左 → 中右」铺开访问顺序。

    切片按时间排好后直接取前 N 条，一小时的视频就只总结开头几分钟 ——
    后面 90% 的内容在摘要里完全不存在，而用户看不出摘要被截断了。

    按位置铺开能保证首尾都在、中间均匀。返回**访问顺序**：第一个元素是
    第一个该被选的；调用方取前 N 个再按时间排回去即可。

        >>> spread_order(5)
        [0, 4, 2, 1, 3]
    """
    if count <= 0:
        return []
    chosen: list[int] = []
    pending: list[tuple[int, int]] = [(0, count - 1)]
    while pending:
        low, high = pending.pop(0)
        if low > high:
            continue
        middle = (low + high) // 2
        for index in dict.fromkeys((low, high, middle)):
            if index not in chosen:
                chosen.append(index)
        if low < high:
            pending.append((low, middle - 1))
            pending.append((middle + 1, high))
    return chosen


# =============================================================================
# 组装器
# =============================================================================


class ContextBuilder:
    """把检索结果压进预算并编号。

    Parameters
    ----------
    budget:
        预算配置。默认 4000 token ≈ 6000 字符（中文）。
    prompts:
        提示词表；``render_prompt`` 用它渲染 ``qa_answer`` / ``video_summarize``。
    """

    def __init__(
        self,
        budget: ContextBudget | None = None,
        *,
        prompts: Any | None = None,
        max_context_items: int | None = None,
    ) -> None:
        self._budget = budget or ContextBudget()
        self._budget.validate()
        if max_context_items is not None:
            self._budget = ContextBudget(
                max_tokens=self._budget.max_tokens,
                max_chars=self._budget.max_chars,
                max_items=max(1, int(max_context_items)),
                max_chars_per_item=self._budget.max_chars_per_item,
                reserve_chars=self._budget.reserve_chars,
                chars_per_token=self._budget.chars_per_token,
                dedupe_prefix_chars=self._budget.dedupe_prefix_chars,
                include_keyframes=self._budget.include_keyframes,
                include_scores=self._budget.include_scores,
            )
        self._prompts = prompts

    @property
    def budget(self) -> ContextBudget:
        return self._budget

    # -- 主入口 -----------------------------------------------------------

    def build(
        self,
        items: Sequence[Any],
        query: str,
        *,
        video_title: str = "",
        prompt_key: str = "qa_answer",
        sort_by: str = "score",
    ) -> BuiltContext:
        """组装问答上下文。

        ``sort_by``：``"score"``（默认，相关度优先）或 ``"time"``（时间线优先）。
        """
        candidates = list(items)
        if not candidates:
            return BuiltContext(
                text="",
                query=query,
                video_title=video_title,
                budget_chars=self._budget.effective_chars,
                degraded=True,
                notes=["检索未返回任何切片"],
            )

        ordered = self._order(candidates, sort_by)
        deduped, dropped = self._dedupe(ordered)

        selected: list[ContextItem] = []
        used = 0
        used_bytes = 0
        budget = self._budget.effective_chars
        byte_budget = self._budget.effective_bytes

        for position, raw in enumerate(deduped):
            if len(selected) >= self._budget.max_items:
                dropped.append(
                    {"clip_id": _get(raw, "clip_id", ""), "reason": "超过 max_items"}
                )
                continue

            item = self._make_item(raw, index=len(selected) + 1, source_rank=position)
            allowance = min(self._budget.max_chars_per_item, budget - used)
            room_bytes = byte_budget - used_bytes
            if allowance < 60 or room_bytes < 200:
                dropped.append(
                    {
                        "clip_id": item.clip_id,
                        "reason": f"预算不足（剩余 {budget - used} 字符 / {room_bytes} 字节）",
                    }
                )
                continue
            item.text, item.truncated = self._fit(item.text, allowance)
            rendered = item.render(
                video_title=video_title,
                include_score=self._budget.include_scores,
                include_keyframes=self._budget.include_keyframes,
            )
            # 字节实测：标头（时间区间/分数/帧数）也要算进去，否则"正文没超、
            # 加上标头就超了"。按超出量收缩正文，最多三次；仍超则整条丢弃 ——
            # 半条证据比没有证据更危险，模型会把截断处当成结论。
            overflow = len(rendered.encode("utf-8")) - room_bytes
            if overflow > 0:
                for _ in range(3):
                    keep_bytes = len(item.text.encode("utf-8")) - overflow - 4
                    if keep_bytes < 180:
                        break
                    shrunk = utf8_prefix(item.text, keep_bytes).rstrip() + "…"
                    if shrunk == item.text:
                        break
                    item.text = shrunk
                    item.truncated = True
                    rendered = item.render(
                        video_title=video_title,
                        include_score=self._budget.include_scores,
                        include_keyframes=self._budget.include_keyframes,
                    )
                    overflow = len(rendered.encode("utf-8")) - room_bytes
                    if overflow <= 0:
                        break
            if overflow > 0:
                dropped.append(
                    {"clip_id": item.clip_id, "reason": f"字节预算不足（超 {overflow} 字节）"}
                )
                continue
            used += len(rendered) + 1
            used_bytes += len(rendered.encode("utf-8")) + 1
            selected.append(item)

        if not selected:
            return BuiltContext(
                text="",
                query=query,
                video_title=video_title,
                budget_chars=budget,
                dropped=dropped,
                degraded=True,
                notes=["预算过小，未能放入任何证据；请调大 context.max_chars"],
            )

        body = self._render_timeline(
            selected, video_title=video_title, timeline=self._budget.timeline_order
        )
        # 收尾实测：上面逐条记账容不下「连续片段」标头、编号与连接符，
        # 拼完才是真实体积。超了就从尾部剥证据 —— 尾部是相关度最低的，
        # 而且保证返回的 body 严格不超预算，调用方不必再猜。
        while selected and len(body.encode("utf-8")) > byte_budget:
            tail = selected.pop()
            dropped.append({"clip_id": tail.clip_id, "reason": "时间线标头超出字节预算"})
            body = (
                self._render_timeline(
                    selected, video_title=video_title, timeline=self._budget.timeline_order
                )
                if selected
                else ""
            )
            used_bytes = len(body.encode("utf-8"))
        text = self.render_prompt(prompt_key, context=body, query=query, video_title=video_title or "该视频")

        notes: list[str] = []
        if len(selected) < len(deduped):
            notes.append(f"共 {len(deduped)} 条候选，按预算选入 {len(selected)} 条")
        if dropped:
            notes.append(f"丢弃 {len(dropped)} 条（重复/超预算/超条数）")

        return BuiltContext(
            text=text,
            items=selected,
            used_chars=len(text),
            budget_chars=budget,
            dropped=dropped,
            query=query,
            video_title=video_title,
            degraded=False,
            notes=notes,
        )

    def _render_timeline(
        self,
        items: Sequence[ContextItem],
        *,
        video_title: str,
        timeline: bool,
    ) -> str:
        """按时间线铺证据；相邻片段合并成「连续片段」标头。

        编号 ``[n]`` 保持相关度顺序不变 —— 打乱编号会让 prompt 里的
        引用示例对不上。变的只是**呈现顺序**：模型读到的是
        「00:00 到 00:42 这一段里依次发生了什么」，而不是按分数跳来跳去
        的一堆孤立片段。这是"回答不连贯"的直接原因之一。
        """

        def render_one(item: ContextItem) -> str:
            return item.render(
                video_title=video_title,
                include_score=self._budget.include_scores,
                include_keyframes=self._budget.include_keyframes,
            )

        if not timeline or len(items) <= 1:
            return "\n\n".join(render_one(it) for it in items)

        ordered = sorted(items, key=lambda it: (it.start_ms, it.end_ms))
        groups: list[list[ContextItem]] = []
        for item in ordered:
            if groups:
                tail_end = max(g.end_ms for g in groups[-1])
                if item.start_ms - tail_end <= self._budget.timeline_gap_ms:
                    groups[-1].append(item)
                    continue
            groups.append([item])

        blocks: list[str] = []
        for number, group in enumerate(groups, start=1):
            if len(group) == 1:
                blocks.append(render_one(group[0]))
                continue
            head = (
                f"【连续片段 {number}：{format_hms(group[0].start_ms)} – "
                f"{format_hms(max(g.end_ms for g in group))}】"
                f"（下列 {len(group)} 段首尾相接，按出现顺序读）"
            )
            blocks.append(head + "\n\n" + "\n\n".join(render_one(g) for g in group))
        return "\n\n".join(blocks)

    def build_summary(
        self,
        items: Sequence[Any],
        *,
        video_title: str = "",
        duration_ms: int = 0,
        prompt_key: str = "video_summarize",
        max_items: int | None = None,
    ) -> BuiltContext:
        """组装总结上下文（**按时间升序**，让模型顺着时间线读）。"""
        limit = max_items if max_items is not None else self._budget.max_items
        candidates = list(items)
        if not candidates:
            return BuiltContext(
                text="",
                video_title=video_title,
                budget_chars=self._budget.effective_chars,
                degraded=True,
                notes=["没有可总结的切片"],
            )

        ordered = sorted(candidates, key=lambda x: (int(_get(x, "start_ms", 0) or 0),))
        dropped: list[dict[str, Any]] = []

        # 段数超过预算时按位置铺开，而不是砍掉尾部
        if len(ordered) > limit:
            keep = set(spread_order(len(ordered))[:limit])
            for index, raw in enumerate(ordered):
                if index not in keep:
                    dropped.append(
                        {"clip_id": _get(raw, "clip_id", ""), "reason": "位置铺开后未入选"}
                    )
            ordered = [raw for index, raw in enumerate(ordered) if index in keep]

        selected: list[ContextItem] = []
        used = 0
        budget = self._budget.effective_chars

        for raw in ordered:
            if len(selected) >= limit:
                dropped.append({"clip_id": _get(raw, "clip_id", ""), "reason": "超过 max_items"})
                continue
            item = self._make_item(raw, index=len(selected) + 1, source_rank=len(selected))
            allowance = min(self._budget.max_chars_per_item, budget - used)
            if allowance < 60:
                dropped.append(
                    {"clip_id": item.clip_id, "reason": f"预算不足（剩余 {budget - used} 字符）"}
                )
                continue
            item.text, item.truncated = self._fit(item.text, allowance)
            used += len(item.text) + 40
            selected.append(item)

        if not selected:
            return BuiltContext(
                text="",
                video_title=video_title,
                budget_chars=budget,
                dropped=dropped,
                degraded=True,
                notes=["预算过小，未能放入任何片段"],
            )

        body = "\n\n".join(
            f"{item.time_range}\n{item.text}" for item in selected
        )
        text = self.render_prompt(
            prompt_key,
            context=body,
            title=video_title or "未命名视频",
            duration=format_hms(duration_ms) if duration_ms else "未知",
        )
        return BuiltContext(
            text=text,
            items=selected,
            used_chars=len(text),
            budget_chars=budget,
            dropped=dropped,
            video_title=video_title,
            notes=[
                f"按时间线选入 {len(selected)}/{len(candidates)} 段"
                + ("（位置铺开，首尾均覆盖）" if len(candidates) > limit else ""),
                *([f"丢弃 {len(dropped)} 段"] if dropped else []),
            ],
        )

    # -- 内部 -------------------------------------------------------------

    def _order(self, items: Sequence[Any], sort_by: str) -> list[Any]:
        if sort_by == "time":
            return sorted(items, key=lambda x: int(_get(x, "start_ms", 0) or 0))
        return sorted(
            items,
            key=lambda x: float(_get(x, "normalized_score", None) or _get(x, "score", 0.0) or 0.0),
            reverse=True,
        )

    def _dedupe(self, items: Sequence[Any]) -> tuple[list[Any], list[dict[str, Any]]]:
        """按文本前缀去重，保留分数更高（已排序时即更靠前）的那条。"""
        seen: dict[str, str] = {}
        kept: list[Any] = []
        dropped: list[dict[str, Any]] = []
        prefix = self._budget.dedupe_prefix_chars

        for raw in items:
            text = str(_get(raw, "text", "") or _get(raw, "core_text", "") or "")
            key = _normalize_for_dedupe(text, prefix)
            clip_id = str(_get(raw, "clip_id", ""))
            if key and key in seen:
                dropped.append(
                    {
                        "clip_id": clip_id,
                        "reason": f"内容与 {seen[key]} 重复（前 {prefix} 字相同）",
                    }
                )
                continue
            if key:
                seen[key] = clip_id
            kept.append(raw)
        return kept, dropped

    def _make_item(self, raw: Any, *, index: int, source_rank: int) -> ContextItem:
        clip_id = str(_get(raw, "clip_id", "") or "")
        text = str(_get(raw, "core_text", None) or _get(raw, "text", "") or "").strip()
        keyframe_ids = _as_str_tuple(_get(raw, "keyframe_ids", ()))
        keyframe_paths = _as_str_tuple(_get(raw, "keyframe_paths", ()))
        ranks = _get(raw, "channel_ranks", {}) or {}
        channels = tuple(sorted(ranks)) if isinstance(ranks, dict) else ()
        score = float(_get(raw, "normalized_score", None) or _get(raw, "score", 0.0) or 0.0)
        return ContextItem(
            index=index,
            clip_id=clip_id,
            video_id=str(_get(raw, "video_id", "") or ""),
            start_ms=int(_get(raw, "start_ms", 0) or 0),
            end_ms=int(_get(raw, "end_ms", 0) or 0),
            text=text,
            score=score,
            keyframe_paths=keyframe_paths if self._budget.include_keyframes else (),
            keyframe_ids=keyframe_ids,
            channels=channels,
            source_rank=source_rank,
        )

    @staticmethod
    def _fit(text: str, allowance: int) -> tuple[str, bool]:
        """把正文压进 ``allowance`` 字符：先按句子边界截，再硬截。"""
        raw = (text or "").strip()
        if len(raw) <= allowance:
            return raw, False
        window = raw[:allowance]
        for sep in ("。", "！", "？", "；", "\n", "，", ","):
            pos = window.rfind(sep)
            if pos >= allowance // 2:
                return window[: pos + 1].rstrip() + " …", True
        return raw[: max(1, allowance - 1)].rstrip() + "…", True

    def render_prompt(self, key: str, **variables: Any) -> str:
        """渲染提示词；没有注册表时退化为纯拼接（测试/离线也能量产上下文）。"""
        fallback = self._fallback_prompt(key, variables)
        if self._prompts is None:
            return fallback
        try:
            return self._prompts.render(key, **variables)
        except KeyError:
            logger.debug("提示词 %r 未注册，使用内置拼装", key)
            return fallback
        except Exception as exc:  # noqa: BLE001 - 提示词问题不该阻断问答
            logger.warning("渲染提示词 %r 失败(%s)，使用内置拼装", key, exc)
            return fallback

    @staticmethod
    def _fallback_prompt(key: str, variables: dict[str, Any]) -> str:
        context = str(variables.get("context", ""))
        if key in {"video_summarize", "summary"}:
            return f"【片段摘要】\n{context}\n\n请据此写一份 Markdown 总结。"
        query = str(variables.get("query", ""))
        title = str(variables.get("video_title", "该视频"))
        return (
            f"你是视频内容问答助手。只依据下面证据回答《{title}》的问题，"
            f"每条结论后用 [编号] 标注来源；证据不足时说明视频里没有提到。\n\n"
            f"【证据】\n{context}\n\n【问题】\n{query}"
        )

    # -- 预算测算 ---------------------------------------------------------

    def estimate_tokens(self, text: str) -> int:
        """粗估 token 数（中文按 1.5 字符/token，计入标点）。"""
        if not text:
            return 0
        return int(len(text) / max(0.5, self._budget.chars_per_token)) + 1

    def __repr__(self) -> str:
        b = self._budget
        return (
            f"ContextBuilder(tokens={b.max_tokens}, chars={b.max_chars}, "
            f"items={b.max_items}, per_item={b.max_chars_per_item})"
        )
