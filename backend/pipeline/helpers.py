"""流水线结果转换、证据整理和文件归属校验。"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any, Callable, Sequence


from backend.pipeline.domain import AskResult, Stage

logger = logging.getLogger("zhiying.pipeline")


def _summary_budget(clip_count: int) -> Any:
    """总结用的专用证据预算（与问答的 top_k+2 上限解耦）。

    问答要"挑最相关的"，总结要"覆盖全片"。早期共用 8 条预算，16 片的视频
    有一半内容模型根本没看到 —— 5 分钟的讲话被压成三句话不是模型的错。
    全量片段上限 32：再长的视频靠"按位置铺开"保证首尾，超出预算的部分
    仍由 build_summary 的 spread_order 挑代表。
    """
    from backend.pipeline.context_builder import ContextBudget  # noqa: PLC0415

    count = max(0, int(clip_count or 0))
    return ContextBudget(
        max_tokens=8_000,
        max_chars=max(12_000, 700 * max(1, count)),
        max_items=max(12, min(32, count or 12)),
        max_chars_per_item=1_200,
    )


def _download_progress(
    progress: Callable[[str, float, str], None] | None,
) -> Callable[[Any], None] | None:
    if progress is None:
        return None

    def handler(event: Any) -> None:
        percent = float(getattr(event, "percent", 0.0) or 0.0)
        # 下载占总进度 0~6%
        progress(
            Stage.PROBE.value,
            0.02 + 0.04 * (percent / 100.0),
            getattr(event, "human", "") or "下载中",
        )

    return handler


def _clip_resolver(
    bounds: Sequence[tuple[str, int, int]],
) -> Callable[[int], str | None]:
    """毫秒时间戳 → 命中的 clip_id（用于把关键帧绑到切片）。"""
    ordered = sorted(bounds, key=lambda x: x[1])

    def resolve(timestamp_ms: int) -> str | None:
        for clip_id, start, end in ordered:
            if start <= timestamp_ms <= end:
                return clip_id
        if ordered:
            return ordered[0][0] if timestamp_ms < ordered[0][1] else ordered[-1][0]
        return None

    return resolve


def _segment_payload(segment: Any) -> dict[str, Any]:
    return {
        "video_id": getattr(segment, "video_id", ""),
        "index_no": int(
            getattr(segment, "index_no", None) or getattr(segment, "index", 0) or 0
        ),
        "start_ms": int(getattr(segment, "start_ms", 0) or 0),
        "end_ms": int(getattr(segment, "end_ms", 0) or 0),
        "text": str(getattr(segment, "text", "") or ""),
        "language": str(getattr(segment, "language", "") or ""),
        "confidence": float(
            getattr(segment, "confidence", None)
            or getattr(segment, "avg_logprob", 0.0)
            or 0.0
        ),
        "source": str(getattr(segment, "source", "asr") or "asr"),
    }


def _clip_index(clip: Any) -> int:
    """``Clip`` 的序号字段历史上叫过 ``index_no``，现在是 ``index``。"""
    return int(getattr(clip, "index", None) or getattr(clip, "index_no", 0) or 0)


def _clip_payload(clip: Any) -> dict[str, Any]:
    return {
        "clip_id": str(getattr(clip, "clip_id", "") or ""),
        "video_id": getattr(clip, "video_id", ""),
        "index_no": _clip_index(clip),
        "start_ms": int(getattr(clip, "start_ms", 0) or 0),
        "end_ms": int(getattr(clip, "end_ms", 0) or 0),
        "core_start_ms": int(getattr(clip, "core_start_ms", 0) or 0),
        "core_end_ms": int(getattr(clip, "core_end_ms", 0) or 0),
        "text": str(getattr(clip, "text", "") or ""),
        "core_text": str(getattr(clip, "core_text", "") or ""),
        "truncated": bool(getattr(clip, "truncated", False)),
    }


def _restore_transcript(video_id: str, records: Sequence[Any]) -> Any:
    """库里的 ``SegmentRecord`` → ``Transcript``，供断点续跑复用转写结果。"""
    from backend.models.asr import Transcript, TranscriptSegment  # noqa: PLC0415

    segments = tuple(
        TranscriptSegment(
            index=int(getattr(r, "index_no", 0) or 0),
            start_ms=int(getattr(r, "start_ms", 0) or 0),
            end_ms=int(getattr(r, "end_ms", 0) or 0),
            text=str(getattr(r, "text", "") or ""),
        )
        for r in records
    )
    first = records[0] if records else None
    source = str(getattr(first, "source", "asr") or "asr")
    return Transcript(
        video_id=video_id,
        segments=segments,
        language=str(getattr(first, "language", "") or ""),
        duration_ms=segments[-1].end_ms if segments else 0,
        source="subtitle" if source == "subtitle" else "whisper",
        model_name="(断点恢复)",
        warnings=("转写来自库中已有片段，未重新识别",),
    )


def _restore_clips(records: Sequence[Any]) -> list[Any]:
    """库里的 ``ClipRecord`` → ``Clip``。"""
    from backend.segmentation.chunker import Clip  # noqa: PLC0415

    return [
        Clip(
            clip_id=str(getattr(r, "clip_id", "") or ""),
            video_id=str(getattr(r, "video_id", "") or ""),
            index=int(getattr(r, "index_no", 0) or 0),
            start_ms=int(getattr(r, "start_ms", 0) or 0),
            end_ms=int(getattr(r, "end_ms", 0) or 0),
            core_start_ms=int(getattr(r, "core_start_ms", 0) or 0),
            core_end_ms=int(getattr(r, "core_end_ms", 0) or 0),
            text=str(getattr(r, "text", "") or ""),
            core_text=str(getattr(r, "core_text", "") or ""),
            keyframe_ids=tuple(getattr(r, "keyframe_ids", ()) or ()),
            truncated=bool(getattr(r, "truncated", False)),
        )
        for r in records
    ]


def _restore_keyframes(records: Sequence[Any]) -> list[Any]:
    """库里的 ``KeyframeRecord`` → ``Keyframe``。"""
    from pathlib import Path as _Path  # noqa: PLC0415

    from backend.segmentation.scenes import Keyframe  # noqa: PLC0415

    restored: list[Any] = []
    for r in records:
        path = str(getattr(r, "path", "") or "")
        if not path or not _Path(path).is_file():
            continue
        restored.append(
            Keyframe(
                keyframe_id=str(getattr(r, "keyframe_id", "") or ""),
                video_id=str(getattr(r, "video_id", "") or ""),
                timestamp_ms=int(getattr(r, "timestamp_ms", 0) or 0),
                path=_Path(path),
                width=int(getattr(r, "width", 0) or 0),
                height=int(getattr(r, "height", 0) or 0),
                size_bytes=int(getattr(r, "size_bytes", 0) or 0),
                scene_index=int(getattr(r, "scene_index", 0) or 0),
                clip_id=str(getattr(r, "clip_id", "") or ""),
                sharpness=float(getattr(r, "sharpness", 0.0) or 0.0),
                sharpness_ok=bool(getattr(r, "sharpness_ok", True)),
                quality_ok=bool(getattr(r, "quality_ok", True)),
                requested_ms=int(getattr(r, "timestamp_ms", 0) or 0),
            )
        )
    return restored


def _item_keyframe_ids(item: Any) -> tuple[str, ...]:
    """从 :class:`FusedItem` 或切片 payload dict 里取关键帧 ID。"""
    raw = (
        item.get("keyframe_ids")
        if isinstance(item, Mapping)
        else getattr(item, "keyframe_ids", ())
    )
    if not raw:
        return ()
    if isinstance(raw, str):
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    try:
        return tuple(str(part) for part in raw)
    except TypeError:
        return ()


def _item_text(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item.get("core_text") or item.get("text") or "")
    return str(getattr(item, "text", "") or "")


def _item_with_text(item: Any, text: str) -> Any:
    """返回替换了 text 的副本；对象不可变替换时原样返回。"""
    if isinstance(item, dict):
        clone = dict(item)
        clone["text"] = text
        clone["core_text"] = text
        clone["evidence_text"] = text
        return clone
    try:
        return _dc_replace(item, text=text)
    except TypeError:
        return item


def _ms_label(value: Any) -> str:
    """毫秒 → ``HH:MM:SS.mmm``（溯源铁律要求毫秒级）。"""
    try:
        ms = max(0, int(value))
    except (TypeError, ValueError):
        ms = 0
    seconds, millis = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _kf_ms(keyframe: Any) -> int:
    try:
        return max(0, int(getattr(keyframe, "timestamp_ms", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _kf_scene(keyframe: Any) -> int:
    try:
        return int(getattr(keyframe, "scene_index", -1) or 0)
    except (TypeError, ValueError):
        return 0


def _parse_milestones(
    text: str, window_start_ms: int, window_end_ms: int
) -> list[tuple[int, str]]:
    """解析 LLM 的里程碑行 ``HH:MM:SS 说明``，只收窗口内的时间。

    时间落在章节窗口 ±60s 之外的行直接丢弃 —— 模型偶尔会抄示例里的
    "00:00:00" 或编一个超界时间，宁缺毋滥。两段式时间（MM:SS）也接受。
    """
    out: list[tuple[int, str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("格式"):
            continue
        match = _MILESTONE_RE.match(line)
        if not match:
            continue
        h, m, s, rest = match.groups()
        try:
            hh = int(h)
            mm = int(m)
            ss = int(s or 0)
        except ValueError:
            continue
        # 两种格式：HH:MM:SS（三段）或 M:SS（两段，"时"位其实是分）
        if s is None:
            total_ms = hh * 60_000 + mm * 1_000
        else:
            total_ms = ((hh * 60 + mm) * 60 + ss) * 1000
        if not (window_start_ms - 60_000 <= total_ms <= window_end_ms + 60_000):
            continue
        note = rest.strip().strip("。；;")[:60]
        if note:
            out.append((total_ms, note))
    out.sort(key=lambda item: item[0])
    return out


def _merge_scenes(scenes: Sequence[Any], span_ms: int) -> list[Any]:
    """D3：把相邻场景合并到 ``>= span_ms`` 的窗口（高运动降档用）。

    返回带 index/start_ms/end_ms/duration_ms/midpoint_ms 的轻量对象，
    满足 :meth:`SceneDetector.extract_keyframes` 对 scenes 的全部取用。
    """
    from types import SimpleNamespace  # noqa: PLC0415

    merged: list[list[int]] = []
    cur: list[int] | None = None
    for scene in scenes:
        start = int(getattr(scene, "start_ms", 0) or 0)
        end = int(getattr(scene, "end_ms", 0) or 0)
        if cur is None:
            cur = [start, end]
        else:
            cur[1] = max(cur[1], end)
        if cur[1] - cur[0] >= span_ms:
            merged.append(cur)
            cur = None
    if cur:
        merged.append(cur)

    out: list[Any] = []
    total_end = max((int(getattr(s, "end_ms", 0) or 0) for s in scenes), default=0)
    for i, (start, end) in enumerate(merged):
        out.append(
            SimpleNamespace(
                index=i,
                start_ms=start,
                end_ms=end,
                duration_ms=end - start,
                midpoint_ms=(start + end) // 2,
                detector="merged",
            )
        )
    if out and total_end and out[-1].end_ms < total_end:
        out[-1].end_ms = total_end
        out[-1].duration_ms = total_end - out[-1].start_ms
    return out


def _segment_chunks(scenes: Sequence[Any], max_s: int) -> list[list[int]]:
    """P1：把连续场景聚成 ``<= max_s`` 秒的时间窗，返回场景下标分组。

    供视频段理解使用 —— 段长决定上传体积与模型可看的运动时长，
    超过 API 限制的段会被服务端拒绝，这里在源头剪好。
    """
    limit_ms = max(1, int(max_s)) * 1000
    chunks: list[list[int]] = []
    cur: list[int] = []
    cur_start = 0
    for i, scene in enumerate(scenes):
        start = int(getattr(scene, "start_ms", 0) or 0)
        end = int(getattr(scene, "end_ms", 0) or 0)
        if cur and end - cur_start > limit_ms:
            chunks.append(cur)
            cur = []
            cur_start = start
        if not cur:
            cur_start = start
        cur.append(i)
    if cur:
        chunks.append(cur)
    return chunks


def _group_by_scene(keyframes: Sequence[Any], limit: int) -> list[list[Any]]:
    """按场景切分关键帧，每组最多 ``limit`` 帧，超出后继续分组，不丢帧。
    做法：场景是天然的"镜头边界"，一个场景内抽的 2 帧讲的是同一件事，
    放进一次推理模型能看出"这段时间画面怎么变"。跨场景硬凑在一起反而
    会让它把两个场景讲成一个。
    """
    groups: list[list[Any]] = []
    current: list[Any] = []
    current_scene: int | None = None
    for kf in keyframes:
        sidx = _kf_scene(kf)
        if current_scene is None or sidx == current_scene:
            current.append(kf)
        else:
            groups.append(current)
            current = [kf]
        current_scene = sidx
    if current:
        groups.append(current)
    size = max(1, int(limit))
    return [
        group[start : start + size]
        for group in groups
        for start in range(0, len(group), size)
    ]


def _keyframe_payload(keyframe: Any, description: str, ocr: str) -> dict[str, Any]:
    return {
        "keyframe_id": getattr(keyframe, "keyframe_id", ""),
        "video_id": getattr(keyframe, "video_id", ""),
        "clip_id": getattr(keyframe, "clip_id", None),
        "scene_index": int(getattr(keyframe, "scene_index", 0) or 0),
        "timestamp_ms": int(getattr(keyframe, "timestamp_ms", 0) or 0),
        "path": str(getattr(keyframe, "path", "") or ""),
        "width": int(getattr(keyframe, "width", 0) or 0),
        "height": int(getattr(keyframe, "height", 0) or 0),
        "size_bytes": int(getattr(keyframe, "size_bytes", 0) or 0),
        "sharpness": float(getattr(keyframe, "sharpness", 0.0) or 0.0),
        "sharpness_ok": bool(getattr(keyframe, "sharpness_ok", True)),
        "quality_ok": bool(getattr(keyframe, "quality_ok", True)),
        "description": description,
        "ocr_text": ocr,
    }


def _synthetic_clips(video_id: str, duration_ms: int, span_ms: int) -> list[Any]:
    """无转录时按固定跨度造切片，保证视觉链路有落点。"""
    from backend.segmentation.chunker import Clip  # noqa: PLC0415

    total = max(0, int(duration_ms or 0))
    if total <= 0:
        return []
    span = max(5_000, int(span_ms or 20_000))
    clips: list[Clip] = []
    index = 0
    cursor = 0
    while cursor < total:
        end = min(total, cursor + span)
        clips.append(
            Clip(
                clip_id=f"{video_id}-c{index:04d}",
                video_id=video_id,
                index=index,
                start_ms=cursor,
                end_ms=end,
                core_start_ms=cursor,
                core_end_ms=end,
                text="",
                core_text="",
            )
        )
        index += 1
        cursor = end
    return clips


def _enforce_answer_claims(result: AskResult, built: Any) -> str:
    """校验回答里的引用编号，返回清洗后的正文。

    ``claims.validate_claims`` 只关心"自由文本 + 证据列表"，不碰
    :class:`AskResult` 的结构，所以这里只做搬运：

    - 悬空编号（指向不存在证据的 ``[3]``）撕掉 —— 实测模型会顺着
      "01/02/03" 的惯性编出画面里根本没有的编号；
    - 孤立陈述句补一条最相近的出处（重合度够高才补）；
    - 每条问题都记进 ``notes``，由 API 层原样透出，不吞掉。

    校验本身失败时保留原回答 —— 宁可少一层保险，也不能因为校验崩掉
    让用户拿不到答案。
    """
    if not result.answer.strip():
        return result.answer
    try:
        from backend.pipeline.claims import enforce_claims  # noqa: PLC0415

        cleaned, report = enforce_claims(result.answer, built.refs)
    except Exception as exc:  # noqa: BLE001 - 校验失败不该影响回答
        logger.debug("证据校验跳过（%s）", exc)
        return result.answer

    for issue in report.issues:
        result.notes.append(f"证据校验: {issue}")
    for fix in report.fixes:
        result.notes.append(f"证据校验: {fix}")
    if not report.ok:
        # 只有"没被修掉"的问题才标降级；自动补上的引用不算降级
        result.degraded = True
    return cleaned


def _evidence_only_answer(built: Any) -> str:
    """LLM 不可用时的回答：直接列出证据，不做改写，但保留引用编号。"""
    lines = ["（当前没有可用的语言模型，以下为检索到的原始证据）", ""]
    for item in built.items:
        lines.append(f"[{item.index}] {item.time_range}（{item.clip_id}）")
        lines.append(item.text or "（该片段无转录文本）")
        lines.append("")
    return "\n".join(lines).strip()


def _strip_chapter_refs(text: str) -> str:
    """抹掉章节摘要里的引用编号。

    章节正文里的 ``[1]`` 是 ``build_summary`` 给片段临时编的号，调用方拿不到
    那张映射表 —— 留着它读者以为是可点的引用，点开却是空。顺手清掉清理后
    残留的标点前空格。
    """
    cleaned = _REF_STRIP.sub("", text)
    # 编号删掉后会留下"  "这样的空洞，中文句子里读起来像错字；只收拾中文
    # 标点两侧与连续空格，英文句号后的空格保留。
    cleaned = re.sub(r"([。，；：！？、）」』】\n])[ \t]+", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+([。，；：！？、）」』】])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _flatten_qa_summary(text: str) -> str:
    """把"问答清单"折回叙述。

    本地小模型看到几条并列要点时，常自作主张改写成 ``1. 某问题？`` + 答案。
    那些问句是模型自己编的（视频里没人这么问），留着读者会以为是被采访的
    原文。这里丢掉问句、只留答案，再拼成一段；只有一条内容时原样返回。
    """
    answers: list[str] = []
    dropped = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        body = _QA_PREFIX.sub("", line).strip()
        if not body:
            continue
        if body.endswith(("？", "?")):
            dropped += 1
            continue
        answers.append(body)
    if dropped <= 0 or len(answers) <= 1:
        return text.strip()
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in "".join(answers))
    joiner = "" if has_cjk else " "
    return joiner.join(answers)


def _chapter_fallback(items: Sequence[Any], *, limit: int = 3) -> str:
    """模型不可用时的章节摘要：按时间取前几段原文拼接，不编造。"""
    picked = list(items)[:limit]
    if not picked:
        return ""
    parts: list[str] = []
    for item in picked:
        text = _item_text(item).strip()
        if not text:
            continue
        stamp = _ms_label(_get_field(item, "start_ms"))
        parts.append(f"{stamp} {text[:200]}")
    return "；".join(parts)


def _unlink_all(paths: Sequence[str]) -> int:
    """删掉给定文件，返回成功个数。单个失败只记日志 —— 清理阶段不该被打断。"""
    gone = 0
    for raw in paths:
        try:
            Path(raw).unlink(missing_ok=True)
            gone += 1
        except OSError:
            logger.debug("删除文件失败: %s", raw)
    return gone


def _unlink_owned(path: Path, roots: Sequence[Path]) -> bool:
    """仅当文件位于给定根目录之下才删除。

    用户在界面里拖进来的视频是自己的素材，删记录不能连原件一起删；
    只有下载缓存在 ``data/videos``、``data/temp`` 里的副本才归系统管。
    """
    try:
        if path.is_symlink():
            # 先验证链接本身所在目录，再unlink链接；绝不能resolve后删除原视频。
            if any(path.parent.resolve().is_relative_to(root.resolve()) for root in roots):
                path.unlink(missing_ok=True)
                return True
            return False
        target = path.resolve()
    except OSError:
        return False
    for root in roots:
        try:
            if target.is_relative_to(root.resolve()) and target.is_file():
                target.unlink()
                return True
        except (OSError, ValueError):
            continue
    return False


def _get_field(item: Any, name: str, default: Any = 0) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


_MILESTONE_RE = re.compile(
    r"^\s*[-*•·]*\s*(?:\d+[.、)）]\s*)?"
    r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*[,，、)）]?\s*[—\-：:]?\s*(\S.*)$"
)


_REF_STRIP = re.compile(r"[\[［][0-9][0-9,，、\-\s]*[\]］]")


_QA_PREFIX = re.compile(r"^(?:\d{1,2}[.、)）]|[-*•·]|\((?:\d{1,2})\))\s*")
