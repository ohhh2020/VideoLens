"""网页内部 qa 路由。"""

from __future__ import annotations


import logging
import asyncio
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from backend.logging_setup import log_task


from fastapi import APIRouter, Body, HTTPException, Request

from fastapi.responses import (
    StreamingResponse,
)


from backend.pipeline.orchestrator import (
    PipelineError,
)


from backend.web.schemas import AskRequest, AnswerOut
from backend.web.dependencies import _orchestrator, _answer_out, _sse

logger = logging.getLogger("zhiying.web.qa")


def create_router(cfg=None):
    router = APIRouter()

    def _cfg(request):
        return cfg or request.app.state.config

    @router.post(
        "/videos/{video_id}/ask",
        response_model=AnswerOut,
        tags=["qa"],
        summary="基于视频内容回答问题（带 [n] 引用与毫秒时间戳）",
    )
    def ask(
        request: Request, video_id: str, payload: AskRequest = Body(...)
    ) -> AnswerOut:
        orch = _orchestrator(request)
        try:
            result = orch.ask(
                video_id,
                payload.question,
                top_k=payload.top_k,
                max_context_tokens=payload.max_context_tokens,
                session_id=payload.session_id,
            )
        except PipelineError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not result.answer and any("视频不存在" in note for note in result.notes):
            raise HTTPException(status_code=404, detail=f"视频不存在: {video_id}")
        return _answer_out(result)

    @router.post(
        "/videos/{video_id}/ask/stream",
        tags=["qa"],
        summary="流式问答（SSE：meta → delta* → final；final 为证据校验后的定稿）",
    )
    def ask_stream_sse(
        request: Request, video_id: str, payload: AskRequest = Body(...)
    ) -> StreamingResponse:
        orch = _orchestrator(request)

        async def gen():
            loop = asyncio.get_running_loop()
            queue = asyncio.Queue(maxsize=64)
            stopped = threading.Event()

            def emit(value):
                if stopped.is_set():
                    return False
                future = asyncio.run_coroutine_threadsafe(queue.put(value), loop)
                while not stopped.is_set():
                    try:
                        future.result(timeout=0.2)
                        return True
                    except FutureTimeout:
                        pass
                future.cancel()
                return False

            def produce():
                stream = orch.ask_stream(
                    video_id,
                    payload.question,
                    top_k=payload.top_k,
                    max_context_tokens=payload.max_context_tokens,
                    session_id=payload.session_id,
                )
                try:
                    with log_task(f"qa:{video_id}/{payload.session_id or 'legacy'}"):
                        for kind, data in stream:
                            event = (
                                {"type": "final", "result": data.to_dict()}
                                if kind == "final"
                                else {"type": kind, "data": data}
                            )
                            if not emit(_sse(event)):
                                break
                except Exception as exc:
                    logger.exception("流式问答失败")
                    emit(
                        _sse(
                            {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
                        )
                    )
                finally:
                    stream.close()
                    emit(None)

            worker = asyncio.create_task(asyncio.to_thread(produce))
            active = getattr(request.app.state, "qa_workers", None)
            if active is None:
                active = request.app.state.qa_workers = set()
            active.add(worker)
            worker.add_done_callback(active.discard)
            # 保留任务引用直到线程退出，避免断流时未取出的异常被吞掉。
            worker.add_done_callback(
                lambda task: task.exception() if not task.cancelled() else None
            )
            try:
                yield ": connected\n\n"
                while True:
                    try:
                        value = await asyncio.wait_for(
                            queue.get(), timeout=_cfg(request).api.sse_heartbeat_s
                        )
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            break
                        yield ": heartbeat\n\n"
                        continue
                    if value is None:
                        break
                    yield value
            finally:
                stopped.set()
                # urllib 的在途网络读取由 socket timeout 收尾，不持本地模型锁。

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return router
