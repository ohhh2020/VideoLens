"""任务注册、编排器依赖与网页结果转换。"""

from __future__ import annotations


import json

import logging


import re

import threading


from collections.abc import Sequence

from dataclasses import asdict, is_dataclass


from typing import Any, Final


from fastapi import HTTPException, Request, status


from backend.pipeline.orchestrator import (
    AnalysisReport,
    AskResult,
    Orchestrator,
)

from backend.storage.repository import NotFoundError, StorageError

from backend.web.jobs import TERMINAL_STATUS, get_job_store

from backend.web.schemas import (
    CitationOut,
    AnswerOut,
    StageOut,
    AnalyzeOut,
    VideoOut,
    ClipOut,
    KeyframeOut,
)

logger = logging.getLogger("zhiying.api.routes")


_TERMINAL_JOB_STATUS: Final[frozenset[str]] = TERMINAL_STATUS


_ORCH: Orchestrator | None = None


_ORCH_LOCK = threading.Lock()


def set_orchestrator(orch: Orchestrator | None) -> None:
    """注入编排器（测试用；生产由 ``main.py`` 在启动时装配）。"""
    global _ORCH
    with _ORCH_LOCK:
        _ORCH = orch


def get_job(task_id: str) -> Any | None:
    """内存任务（无则 ``None``；持久化记录用 ``job_snapshot`` 查）。"""
    return get_job_store().get(task_id)


def job_snapshot(task_id: str) -> dict[str, Any] | None:
    """任务快照：内存优先，其次 SQLite（进程重启后仍可查历史任务）。"""
    snapshot = get_job_store().snapshot(task_id)
    if snapshot is not None:
        return snapshot
    with _ORCH_LOCK:
        orch = _ORCH
    if orch is None:
        return None
    try:
        record = orch.repository.get_task(task_id)
    except (NotFoundError, StorageError):
        return None
    return {
        "task_id": record.task_id,
        "video_id": record.video_id,
        # SQLite 里 pending 语义对应接口的 queued
        "status": "queued" if record.status == "pending" else record.status,
        "stage": record.stage,
        "progress": record.progress,
        "message": record.message,
        "error": record.error,
        "created_at": record.created_ts if hasattr(record, "created_ts") else 0.0,
        "updated_at": record.updated_ts,
        "elapsed_ms": 0,
        "report": None,
    }


def shutdown_jobs(*, wait: bool = True) -> None:
    """关停线程池（进程退出 / 测试清理）。"""
    from backend.web.jobs import shutdown_jobs as _shutdown  # noqa: PLC0415

    _shutdown(wait=wait)


def _orchestrator(request: Request) -> Orchestrator:
    """取编排器：优先 app.state（启动时装配），其次模块级注入。"""
    orch: Orchestrator | None = getattr(request.app.state, "orchestrator", None)
    if orch is None:
        orch = _ORCH
    if orch is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="编排器未就绪（模型或依赖尚未加载完成）",
        )
    return orch


def _as_dict(obj: Any) -> dict[str, Any]:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    raise TypeError(f"无法序列化: {type(obj)!r}")


def _video_out(record: Any) -> VideoOut:
    raw = _as_dict(record)
    return VideoOut(
        video_id=str(raw.get("video_id", "")),
        display_name=str(raw.get("display_name", "")),
        title=str(raw.get("title", "")),
        source_kind=str(raw.get("source_kind", "")),
        source_url=str(raw.get("source_url", "")),
        duration_ms=int(raw.get("duration_ms", 0) or 0),
        width=int(raw.get("width", 0) or 0),
        height=int(raw.get("height", 0) or 0),
        fps=float(raw.get("fps", 0.0) or 0.0),
        language=str(raw.get("language", "") or ""),
        status=str(raw.get("status", "")),
        stage=str(raw.get("stage", "") or ""),
        progress=float(raw.get("progress", 0.0) or 0.0),
        error=str(raw.get("error", "") or ""),
        created_at=str(raw.get("created_at", "") or ""),
        updated_at=str(raw.get("updated_at", "") or ""),
    )


def _clip_out(record: Any) -> ClipOut:
    raw = _as_dict(record)
    start = int(raw.get("start_ms", 0) or 0)
    end = int(raw.get("end_ms", 0) or 0)
    ids = raw.get("keyframe_ids", ()) or ()
    if isinstance(ids, str):
        ids = [part for part in ids.split(",") if part]
    return ClipOut(
        clip_id=str(raw.get("clip_id", "")),
        video_id=str(raw.get("video_id", "")),
        index=int(raw.get("index_no", raw.get("index", 0)) or 0),
        start_ms=start,
        end_ms=end,
        core_start_ms=int(raw.get("core_start_ms", start) or start),
        core_end_ms=int(raw.get("core_end_ms", end) or end),
        time_range=f"{_hms(start)}–{_hms(end)}",
        text=str(raw.get("text", "") or ""),
        core_text=str(raw.get("core_text", "") or ""),
        keyframe_ids=[str(x) for x in ids],
    )


def _keyframe_out(record: Any) -> KeyframeOut:
    raw = _as_dict(record)
    ts = int(raw.get("timestamp_ms", 0) or 0)
    return KeyframeOut(
        keyframe_id=str(raw.get("keyframe_id", "")),
        video_id=str(raw.get("video_id", "")),
        clip_id=(str(raw["clip_id"]) if raw.get("clip_id") else None),
        scene_index=int(raw.get("scene_index", 0) or 0),
        timestamp_ms=ts,
        timecode=_hms(ts),
        path=str(raw.get("path", "") or ""),
        width=int(raw.get("width", 0) or 0),
        height=int(raw.get("height", 0) or 0),
        size_bytes=int(raw.get("size_bytes", 0) or 0),
        sharpness=float(raw.get("sharpness", 0.0) or 0.0),
        description=str(raw.get("description", "") or ""),
        ocr_text=str(raw.get("ocr_text", "") or ""),
    )


def _hms(ms: int) -> str:
    """毫秒 → ``HH:MM:SS.mmm``（需求文档规定的统一时间格式）。"""
    total = max(0, int(ms))
    hours, rest = divmod(total, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _srt_ts(ms: int) -> str:
    """毫秒 → SRT 时间戳（``HH:MM:SS,mmm``，逗号是标准要求）。"""
    total = max(0, int(ms))
    hours, rest = divmod(total, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _vtt_ts(ms: int) -> str:
    """毫秒 → WebVTT 时间戳（``HH:MM:SS.mmm``）。"""
    return _srt_ts(ms).replace(",", ".")


def _build_srt(segments: Sequence[Any]) -> str:
    """转写片段 → SRT 文本。空文本段跳过，时间倒挂段至少给 1s。"""
    blocks: list[str] = []
    for seg in segments:
        text = str(getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        start = int(getattr(seg, "start_ms", 0) or 0)
        end = max(int(getattr(seg, "end_ms", 0) or 0), start + 1000)
        blocks.append(f"{len(blocks) + 1}\n{_srt_ts(start)} --> {_srt_ts(end)}\n{text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _build_vtt(segments: Sequence[Any]) -> str:
    """转写片段 → WebVTT 文本（时间戳用点号，块结构直接复用 SRT 生成器）。"""
    srt = _build_srt(segments)
    if not srt:
        return "WEBVTT\n"
    return "WEBVTT\n\n" + re.sub(r"(\d{2}:\d{2}:\d{2}),(\d{3})", r"\1.\2", srt)


def _subtitle_filename(title: str, video_id: str, fmt: str) -> str:
    """导出文件名：优先标题，剔除文件系统非法字符；兜底 video_id。"""
    base = re.sub(r"^https?://(www\.)?", "", title.strip())
    base = "".join(c for c in base if c not in '\\/:*?"<>|').strip()
    base = base or video_id
    return f"{base}.{fmt}"


def _answer_out(result: AskResult) -> AnswerOut:
    return AnswerOut(
        question=result.question,
        answer=result.answer,
        video_id=result.video_id,
        refs=[CitationOut(**c.to_dict()) for c in result.refs],
        backend=result.backend,
        degraded=result.degraded,
        elapsed_ms=result.elapsed_ms,
        context_chars=result.context_chars,
        candidates=result.candidates,
        notes=list(result.notes),
    )


def _analyze_out(report: AnalysisReport) -> AnalyzeOut:
    raw = report.to_dict()
    return AnalyzeOut(
        video_id=str(raw.get("video_id", "")),
        title=str(raw.get("title", "")),
        status=str(raw.get("status", "")),
        ok=bool(raw.get("ok", False)),
        duration_ms=int(raw.get("duration_ms", 0)),
        clip_count=int(raw.get("clip_count", 0)),
        keyframe_count=int(raw.get("keyframe_count", 0)),
        segment_count=int(raw.get("segment_count", 0)),
        vector_count=int(raw.get("vector_count", 0)),
        transcript_source=str(raw.get("transcript_source", "")),
        language=str(raw.get("language", "")),
        summary=str(raw.get("summary", "")),
        summary_backend=str(raw.get("summary_backend", "")),
        degraded=list(raw.get("degraded", [])),
        warnings=list(raw.get("warnings", [])),
        errors=list(raw.get("errors", [])),
        elapsed_ms=int(raw.get("elapsed_ms", 0)),
        stages=[StageOut(**s) for s in raw.get("stages", [])],
    )


def _sse(payload: dict[str, Any]) -> str:
    """SSE 帧：``data: {json}\\n\\n``（JSON 单行，避免多行 payload 拆帧）。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
