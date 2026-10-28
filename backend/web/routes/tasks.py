"""网页内部 tasks 路由。"""

from __future__ import annotations

import asyncio


from typing import Any


from fastapi import APIRouter, HTTPException, Query, Request

from fastapi.responses import (
    StreamingResponse,
)


from backend.web.jobs import TERMINAL_STATUS, get_job_store

_TERMINAL_JOB_STATUS = TERMINAL_STATUS

from backend.web.schemas import JobOut
from backend.web.dependencies import get_job, job_snapshot, _sse


def create_router(cfg=None):
    router = APIRouter()

    def _cfg(request):
        return cfg or request.app.state.config

    @router.get(
        "/tasks/{task_id}",
        response_model=JobOut,
        tags=["tasks"],
        summary="查询任务进度 / 结果",
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
    )
    def list_jobs(limit: int = Query(20, ge=1, le=200)) -> list[JobOut]:
        return [JobOut.model_validate(s) for s in get_job_store().snapshots(limit)]

    return router
