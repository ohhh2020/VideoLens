"""网页内部 system 路由。"""

from __future__ import annotations

import os
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict


from backend.storage.repository import StorageError

from backend.web.jobs import get_job_store
from backend.models.devices import active_devices

from backend.web.schemas import HealthOut, StatsOut
from backend.web.dependencies import _orchestrator, _as_dict, _ORCH
from backend.runtime import runtime_path
from backend.web.console import send_console_command


class ServiceAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["restart", "stop"]


def create_router(cfg=None):
    router = APIRouter()

    def _cfg(request):
        return cfg or request.app.state.config

    @router.get(
        "/health", response_model=HealthOut, tags=["system"], summary="健康检查"
    )
    def health(request: Request) -> HealthOut:
        conf = _cfg(request)
        orch = getattr(request.app.state, "orchestrator", None) or _ORCH
        devices = active_devices(orch)
        loaded = set(devices.values())
        return HealthOut(
            pid=os.getpid(),
            status="ok" if orch is not None else "degraded",
            version=str(getattr(conf.app, "version", "")),
            project_root=str(conf.root_dir),
            orchestrator=orch is not None,
            device=next(iter(loaded)) if len(loaded) == 1 else "mixed",
            model_devices=devices,
            auth="local-web",
            jobs_running=get_job_store().running_count(),
            jobs_total=get_job_store().total(),
        )

    @router.post("/service/control", status_code=202, summary="网页服务菜单：正常重启或关闭")
    def service_control(request: Request, payload: ServiceAction):
        conf = _cfg(request)
        server = request.scope.get("server")
        port = server[1] if server else conf.api.port
        try:
            result = send_console_command(payload.action,
                state_path=runtime_path(f"state/console-{port}.json",conf.root_dir))
        except (OSError, ValueError, TimeoutError) as exc:
            raise HTTPException(503,"无法连接服务控制台，请使用项目启动脚本启动服务") from exc
        if not result.get("accepted"):
            raise HTTPException(503,"控制台未接受此操作，请稍后重试")
        # 正常关闭会等待包括本请求在内的响应结束，不会在发送响应前杀掉进程。
        return {"accepted":True,"action":payload.action,"pid":os.getpid()}

    @router.get(
        "/stats",
        response_model=StatsOut,
        tags=["system"],
        summary="存储容量与用量（需求文档表3）",
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

    return router
