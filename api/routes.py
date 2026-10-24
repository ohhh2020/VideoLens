"""知影 HTTP 路由层 —— 对外唯一的业务入口。

设计要点
--------
1. **薄路由**：本模块不做任何业务判断，只负责 ① 校验入参 ② 调用
   :class:`core.pipeline.orchestrator.Orchestrator` ③ 把 dataclass 结果
   转成 Pydantic 响应。所有降级、重试、并发控制都在 core 层完成，路由层
   不写第二遍。
2. **长任务异步化**：一段 1h 视频的分析是分钟级操作（需求文档允许 30min），
   不能占着 HTTP 连接。``POST /videos/analyze`` 立刻返回 ``task_id``，
   真正的执行在线程池里；进度与结果通过
   ``GET /tasks/{task_id}``（轮询）或 ``GET /tasks/{task_id}/events``（SSE）拿。
3. **任务体系单一事实源**：后台执行、内存进度流、SSE 广播与 SQLite 持久化
   恢复记录全部在 :mod:`api.jobs` 的 ``JobStore`` 里，**task_id 只发一次** ——
   HTTP 与 MCP 提交的任务用同一个注册表，``/tasks/{id}`` 查不到内存时回落
   SQLite，进程重启后旧任务仍可查。
4. **鉴权是"默认安全"而非"默认开放"**：配置里给了 ``api.auth_token_env``
   指定的环境变量就读它当 Bearer Token；没有配 token 时**只允许本机回环**
   访问（本地优先原则：默认不给局域网开口子）。MCP / Gradio 复用同一判据。
5. **错误不泄露内部路径**：``PipelineError`` 之外的一律 500 并只回
   ``internal_error``，详细信息进日志；``NotFoundError`` 映射 404。
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Final, Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from pydantic import BaseModel, Field, field_validator

from core.config import AppConfig, get_app_config
from core.pipeline.orchestrator import (
    AnalysisReport,
    AnalyzeOptions,
    AskResult,
    Orchestrator,
    PipelineError,
)
from core.storage.repository import NotFoundError, StorageError

from api.jobs import TERMINAL_STATUS, get_job_store

logger = logging.getLogger("zhiying.api.routes")

__all__ = [
    "AnalyzeRequest",
    "AskRequest",
    "create_router",
    "get_job",
    "job_snapshot",
    "set_orchestrator",
    "shutdown_jobs",
]

#: SSE 流在这些状态下收尾（否则客户端会一直等）。
_TERMINAL_JOB_STATUS: Final[frozenset[str]] = TERMINAL_STATUS


# =============================================================================
# 请求 / 响应模型
# =============================================================================


class AnalyzeRequest(BaseModel):
    """``POST /videos/analyze`` 的入参。"""

    model_config = {"extra": "forbid"}

    source: str = Field(
        ...,
        min_length=1,
        max_length=32_768,
        description="本地路径或视频 URL；URL ≤ limits.max_url_length（默认 2048）字符，本地路径 ≤32768",
        examples=["D:/videos/demo.mp4", "https://www.bilibili.com/video/BV1xx"],
    )
    title: str = Field("", max_length=200, description="自定义标题，留空则自动")
    language: str | None = Field(None, max_length=16, description="强制语言，如 zh / en")
    skip_vision: bool = Field(False, description="跳过视觉理解（只看语音）")
    skip_summary: bool = Field(False, description="跳过整片总结")
    skip_vectors: bool = Field(False, description="跳过向量写入（只留 SQLite）")
    force: bool = Field(False, description="强制重跑，忽略已有结果")
    resume: bool = Field(True, description="断点续跑（跳过已完成阶段）")
    max_keyframes: int | None = Field(None, ge=1, le=5000)

    @field_validator("source")
    @classmethod
    def _strip(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("source 不能为空")
        return text


class AskRequest(BaseModel):
    """``POST /videos/{video_id}/ask`` 的入参。"""

    model_config = {"extra": "forbid"}

    question: str = Field(..., min_length=1, max_length=500, description="问题（≤500 字符）")
    top_k: int | None = Field(None, ge=1, le=50, description="检索条数，默认取配置")
    max_context_tokens: int | None = Field(None, ge=512, le=32_000)

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("question 不能为空")
        return text


class CitationOut(BaseModel):
    index: int
    clip_id: str
    video_id: str
    start_ms: int
    end_ms: int
    time_range: str
    keyframe_paths: list[str] = Field(default_factory=list)
    text: str = ""


class AnswerOut(BaseModel):
    question: str
    answer: str
    video_id: str
    refs: list[CitationOut] = Field(default_factory=list)
    backend: str = ""
    degraded: bool = False
    elapsed_ms: int = 0
    context_chars: int = 0
    candidates: int = 0
    notes: list[str] = Field(default_factory=list)


class StageOut(BaseModel):
    stage: str
    ok: bool
    skipped: bool = False
    degraded: bool = False
    message: str = ""
    error: str = ""
    elapsed_ms: int = 0


class AnalyzeOut(BaseModel):
    video_id: str
    title: str = ""
    status: str
    ok: bool = False
    duration_ms: int = 0
    clip_count: int = 0
    keyframe_count: int = 0
    segment_count: int = 0
    vector_count: int = 0
    transcript_source: str = ""
    language: str = ""
    summary: str = ""
    summary_backend: str = ""
    degraded: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    elapsed_ms: int = 0
    stages: list[StageOut] = Field(default_factory=list)


class VideoOut(BaseModel):
    video_id: str
    display_name: str = ""
    title: str = ""
    source_kind: str = ""
    source_url: str = ""
    duration_ms: int = 0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    language: str = ""
    status: str = ""
    stage: str = ""
    progress: float = 0.0
    error: str = ""
    created_at: str = ""
    updated_at: str = ""


class ClipOut(BaseModel):
    clip_id: str
    video_id: str
    index: int
    start_ms: int
    end_ms: int
    core_start_ms: int
    core_end_ms: int
    time_range: str
    text: str
    core_text: str
    keyframe_ids: list[str] = Field(default_factory=list)


class KeyframeOut(BaseModel):
    keyframe_id: str
    video_id: str
    clip_id: str | None = None
    scene_index: int = 0
    timestamp_ms: int
    timecode: str
    path: str
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    sharpness: float = 0.0
    description: str = ""
    ocr_text: str = ""


class VideoDetailOut(BaseModel):
    video: VideoOut
    clips: list[ClipOut] = Field(default_factory=list)
    keyframes: list[KeyframeOut] = Field(default_factory=list)
    clip_count: int = 0
    keyframe_count: int = 0


class JobOut(BaseModel):
    task_id: str
    video_id: str = ""
    status: Literal["queued", "running", "done", "failed", "cancelled"] = "queued"
    stage: str = ""
    progress: float = 0.0
    message: str = ""
    error: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    elapsed_ms: int = 0
    report: AnalyzeOut | None = None


class HealthOut(BaseModel):
    status: str = "ok"
    version: str = ""
    project_root: str = ""
    orchestrator: bool = False
    device: str = ""
    auth: str = "loopback-only"
    jobs_running: int = 0
    jobs_total: int = 0


class StatsOut(BaseModel):
    videos: int = 0
    clips: int = 0
    transcript_segments: int = 0
    keyframes: int = 0
    tasks_pending: int = 0
    tasks_running: int = 0
    keyframe_bytes: int = 0
    db_bytes: int = 0
    usage: dict[str, float] = Field(default_factory=dict)


class DeleteOut(BaseModel):
    video_id: str
    db: list[str] = Field(default_factory=list)
    vectors: dict[str, int] = Field(default_factory=dict)
    deleted: bool = True


# =============================================================================
# 编排器注入 + 任务体系薄封装（实现在 api.jobs.JobStore）
# =============================================================================

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
    from api.jobs import shutdown_jobs as _shutdown  # noqa: PLC0415

    _shutdown(wait=wait)


# =============================================================================
# 鉴权
# =============================================================================


def _client_is_loopback(request: Request) -> bool:
    host = (request.client.host if request.client else "") or ""
    # ::ffff:127.0.0.1 是 IPv6 套接字上的 IPv4 回环（uvicorn 双栈监听时出现）。
    # 注意：这里绝不放行 "testclient" 之类的测试桩 —— 回环判定只认真实地址，
    # 测试需要带 X-API-Token 走正门（见 tests/test_api.py）。
    return host in {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}


def _expected_token(cfg: AppConfig) -> str:
    return (os.environ.get(cfg.api.auth_token_env) or "").strip()


def require_auth(request: Request) -> None:
    """Token 优先，未配置则只放行回环地址。

    本地优先产品不该默认把端口暴露给局域网 —— 没有 token 就意味着"只有本机
    能用"，而不是"谁都能用"。token 比较用 ``hmac.compare_digest``：非常量
    时间比较的退出时机本身就是一个旁信道。
    """
    cfg: AppConfig = getattr(request.app.state, "config", None) or get_app_config()
    expected = _expected_token(cfg)
    if not expected:
        if _client_is_loopback(request):
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "未配置访问令牌，仅允许本机访问。"
                f"如需远程访问，请设置环境变量 {cfg.api.auth_token_env}"
            ),
        )

    header = request.headers.get("authorization", "")
    token = ""
    if header.lower().startswith("bearer "):
        token = header[7:].strip()
    token = token or request.headers.get("x-api-token", "").strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少访问令牌（Authorization: Bearer <token> 或 X-API-Token）",
        )
    if not hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="访问令牌不正确")


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


# =============================================================================
# dataclass → 响应模型 的小工具
# =============================================================================


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


# =============================================================================
# 路由
# =============================================================================


def create_router(cfg: AppConfig | None = None) -> APIRouter:
    """构造 ``APIRouter``。``cfg`` 为空时每次请求取全局配置。"""
    router = APIRouter()
    auth = Depends(require_auth)

    def _cfg(request: Request) -> AppConfig:
        return cfg or getattr(request.app.state, "config", None) or get_app_config()

    # -- 健康 / 统计 -------------------------------------------------------
    @router.get("/health", response_model=HealthOut, tags=["system"], summary="健康检查")
    def health(request: Request) -> HealthOut:
        conf = _cfg(request)
        orch = getattr(request.app.state, "orchestrator", None) or _ORCH
        return HealthOut(
            status="ok" if orch is not None else "degraded",
            version=str(getattr(conf.app, "version", "")),
            project_root=str(conf.root_dir),
            orchestrator=orch is not None,
            device=str(conf.models.device),
            auth="token" if _expected_token(conf) else "loopback-only",
            jobs_running=get_job_store().running_count(),
            jobs_total=get_job_store().total(),
        )

    @router.get(
        "/stats",
        response_model=StatsOut,
        tags=["system"],
        summary="存储容量与用量（需求文档表3）",
        dependencies=[auth],
    )
    def stats(request: Request) -> StatsOut:
        orch = _orchestrator(request)
        try:
            raw = orch.repository.stats()
            payload = _as_dict(raw)
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=f"读取统计失败: {exc}") from exc
        return StatsOut(
            videos=int(payload.get("videos", 0)),
            clips=int(payload.get("clips", 0)),
            transcript_segments=int(payload.get("transcript_segments", 0)),
            keyframes=int(payload.get("keyframes", 0)),
            tasks_pending=int(payload.get("tasks_pending", 0)),
            tasks_running=int(payload.get("tasks_running", 0)),
            keyframe_bytes=int(payload.get("keyframe_bytes", 0)),
            db_bytes=int(payload.get("db_bytes", 0)),
            usage={k: float(v) for k, v in (payload.get("usage") or {}).items()},
        )

    # -- 分析 --------------------------------------------------------------
    @router.post(
        "/videos/analyze",
        response_model=JobOut,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["videos"],
        summary="提交一个视频分析任务（异步）",
        dependencies=[auth],
    )
    def analyze(
        request: Request, payload: AnalyzeRequest = Body(...)
    ) -> JobOut:
        orch = _orchestrator(request)
        conf = _cfg(request)

        # 入参前置校验：URL 长度上限、文件大小上限由 core 层再校验一次，
        # 这里先做"明显不合理"的拦截，避免无意义的排队。
        source = payload.source
        if len(source) > conf.limits.max_url_length and "://" in source:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"URL 超过 {conf.limits.max_url_length} 字符",
            )

        try:
            job = get_job_store().submit(orch, source, AnalyzeOptions(
                language=payload.language,
                skip_vision=payload.skip_vision,
                skip_summary=payload.skip_summary,
                skip_vectors=payload.skip_vectors,
                force=payload.force,
                resume=payload.resume,
                max_keyframes=payload.max_keyframes,
                title=payload.title,
            ))
        except RuntimeError as exc:  # 线程池已关停
            raise HTTPException(status_code=503, detail="服务正在关闭") from exc
        return JobOut.model_validate(job.snapshot())

    @router.get(
        "/tasks/{task_id}",
        response_model=JobOut,
        tags=["tasks"],
        summary="查询任务进度 / 结果",
        dependencies=[auth],
    )
    def task(task_id: str) -> JobOut:
        snapshot = job_snapshot(task_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
        return JobOut.model_validate(snapshot)

    @router.get(
        "/tasks/{task_id}/events",
        tags=["tasks"],
        summary="任务进度流（SSE）",
        dependencies=[auth],
    )
    async def task_events(request: Request, task_id: str) -> StreamingResponse:
        job = get_job(task_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

        conf = _cfg(request)
        heartbeat = max(1, int(conf.api.sse_heartbeat_s))
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        job.queues.append(queue)
        job.loop = asyncio.get_running_loop()

        async def stream() -> Any:
            try:
                # 首帧先推当前状态，客户端不必等下一次变更。
                snapshot = job.snapshot()
                yield _sse(snapshot)
                # 首帧已是终态（客户端订阅晚了）就没必要继续挂着等消息。
                if snapshot.get("status") in _TERMINAL_JOB_STATUS:
                    return
                while True:
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=heartbeat)
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"  # 心跳注释行，前端 EventSource 会自动忽略
                        continue
                    yield _sse(payload)
                    if payload.get("status") in _TERMINAL_JOB_STATUS:
                        break
            finally:
                try:
                    job.queues.remove(queue)
                except ValueError:
                    pass

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get(
        "/tasks",
        response_model=list[JobOut],
        tags=["tasks"],
        summary="列出内存中的任务",
        dependencies=[auth],
    )
    def list_jobs(limit: int = Query(20, ge=1, le=200)) -> list[JobOut]:
        return [JobOut.model_validate(s) for s in get_job_store().snapshots(limit)]

    # -- 视频 --------------------------------------------------------------
    @router.get(
        "/videos",
        response_model=list[VideoOut],
        tags=["videos"],
        summary="视频列表",
        dependencies=[auth],
    )
    def list_videos(
        request: Request,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        status_filter: str | None = Query(None, alias="status"),
    ) -> list[VideoOut]:
        orch = _orchestrator(request)
        try:
            records = orch.repository.list_videos(limit=limit, offset=offset, status=status_filter)
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=f"读取列表失败: {exc}") from exc
        return [_video_out(item) for item in records]

    @router.get(
        "/videos/{video_id}",
        response_model=VideoDetailOut,
        tags=["videos"],
        summary="视频详情（含切片与关键帧）",
        dependencies=[auth],
    )
    def video_detail(request: Request, video_id: str) -> VideoDetailOut:
        orch = _orchestrator(request)
        try:
            record = orch.repository.find_video(video_id)
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if record is None:
            raise HTTPException(status_code=404, detail=f"视频不存在: {video_id}")
        clips = [_clip_out(c) for c in orch.repository.get_clips(video_id)]
        frames = [_keyframe_out(k) for k in orch.repository.get_keyframes(video_id)]
        return VideoDetailOut(
            video=_video_out(record),
            clips=clips,
            keyframes=frames,
            clip_count=len(clips),
            keyframe_count=len(frames),
        )

    @router.get(
        "/videos/{video_id}/clips",
        response_model=list[ClipOut],
        tags=["videos"],
        summary="切片列表",
        dependencies=[auth],
    )
    def video_clips(request: Request, video_id: str) -> list[ClipOut]:
        orch = _orchestrator(request)
        return [_clip_out(c) for c in orch.repository.get_clips(video_id)]

    @router.get(
        "/videos/{video_id}/keyframes",
        response_model=list[KeyframeOut],
        tags=["videos"],
        summary="关键帧列表",
        dependencies=[auth],
    )
    def video_keyframes(request: Request, video_id: str) -> list[KeyframeOut]:
        orch = _orchestrator(request)
        return [_keyframe_out(k) for k in orch.repository.get_keyframes(video_id)]

    @router.get(
        "/videos/{video_id}/keyframes/{keyframe_id}/image",
        tags=["videos"],
        summary="取关键帧图片",
        dependencies=[auth],
    )
    def keyframe_image(request: Request, video_id: str, keyframe_id: str) -> FileResponse:
        """按 ID 取图，**不接受客户端路径**，因此不存在路径穿越面。"""
        orch = _orchestrator(request)
        try:
            record = orch.repository.get_keyframe(keyframe_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"关键帧不存在: {keyframe_id}") from exc
        if record is None or record.video_id != video_id:
            raise HTTPException(status_code=404, detail=f"关键帧不存在: {keyframe_id}")
        path = Path(record.path)
        if not path.is_file():
            raise HTTPException(status_code=410, detail="关键帧文件已被清理")
        return FileResponse(
            path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @router.delete(
        "/videos/{video_id}",
        response_model=DeleteOut,
        tags=["videos"],
        summary="删除视频及其切片、关键帧、向量",
        dependencies=[auth],
    )
    def delete_video(request: Request, video_id: str) -> DeleteOut:
        orch = _orchestrator(request)
        try:
            removed = orch.delete(video_id)
        except PipelineError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if removed.get("db_error") and "不存在" in str(removed["db_error"]):
            raise HTTPException(status_code=404, detail=f"视频不存在: {video_id}")
        return DeleteOut(
            video_id=video_id,
            db=list(removed.get("db") or []),
            vectors=dict(removed.get("vectors") or {}),
        )

    # -- 问答 --------------------------------------------------------------
    @router.post(
        "/videos/{video_id}/ask",
        response_model=AnswerOut,
        tags=["qa"],
        summary="基于视频内容回答问题（带 [n] 引用与毫秒时间戳）",
        dependencies=[auth],
    )
    def ask(request: Request, video_id: str, payload: AskRequest = Body(...)) -> AnswerOut:
        orch = _orchestrator(request)
        try:
            result = orch.ask(
                video_id,
                payload.question,
                top_k=payload.top_k,
                max_context_tokens=payload.max_context_tokens,
            )
        except PipelineError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not result.answer and any("视频不存在" in note for note in result.notes):
            raise HTTPException(status_code=404, detail=f"视频不存在: {video_id}")
        return _answer_out(result)

    @router.post(
        "/videos/{video_id}/summary",
        response_model=AnalyzeOut,
        tags=["videos"],
        summary="重新生成整片总结",
        dependencies=[auth],
    )
    def resummarize(request: Request, video_id: str) -> AnalyzeOut:
        orch = _orchestrator(request)
        try:
            text = orch.summarize(video_id)
        except PipelineError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        record = orch.repository.find_video(video_id)
        report = AnalysisReport(
            video_id=video_id,
            title=(record.title or record.display_name) if record else "",
            status="ready",
            summary=text,
            clip_count=len(orch.repository.get_clips(video_id)),
            keyframe_count=len(orch.repository.get_keyframes(video_id)),
        )
        return _analyze_out(report)

    @router.get(
        "/videos/{video_id}/chapters",
        tags=["videos"],
        summary="章节化摘要（长视频按时间切片逐章总结）",
        dependencies=[auth],
    )
    def list_chapters(
        request: Request,
        video_id: str,
        chapter_seconds: int = 300,
        max_chapters: int = 40,
        min_seconds: int = 600,
        summarize: bool = True,
    ) -> dict[str, Any]:
        orch = _orchestrator(request)
        chapter_seconds = max(30, min(int(chapter_seconds), 3600))
        max_chapters = max(1, min(int(max_chapters), 200))
        try:
            chapters = orch.chapters(
                video_id,
                chapter_seconds=chapter_seconds,
                max_chapters=max_chapters,
                min_seconds=max(0, int(min_seconds)),
                summarize=bool(summarize),
            )
        except PipelineError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "video_id": video_id,
            "chapter_seconds": chapter_seconds,
            # 空列表意味着视频太短，没必要拆章 —— 客户端据此退回单段总结。
            "needs_chapters": bool(chapters),
            "count": len(chapters),
            "chapters": [c.to_dict() for c in chapters],
        }

    @router.get(
        "/videos/{video_id}/chapters/report",
        response_class=PlainTextResponse,
        tags=["videos"],
        summary="章节化 markdown 报告",
        dependencies=[auth],
    )
    def chapter_report(
        request: Request,
        video_id: str,
        chapter_seconds: int = 300,
        min_seconds: int = 600,
    ) -> PlainTextResponse:
        orch = _orchestrator(request)
        try:
            text = orch.chapter_report(
                video_id,
                chapter_seconds=max(30, min(int(chapter_seconds), 3600)),
                min_seconds=max(0, int(min_seconds)),
            )
        except PipelineError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")

    return router


def _sse(payload: dict[str, Any]) -> str:
    """SSE 帧：``data: {json}\\n\\n``（JSON 单行，避免多行 payload 拆帧）。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
