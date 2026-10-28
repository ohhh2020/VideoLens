"""网页工作台页面托管与设置读写；前端资源位于 frontend/workbench。"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Literal

from dataclasses import asdict

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

from backend.user_settings import (
    SettingsError,
    current_settings,
    describe_backends,
    save_settings,
    preview_settings,
)
from backend.models.devices import (
    device_capabilities,
    effective_settings,
    active_devices,
)

logger = logging.getLogger("zhiying.api.ui")
from backend.models.switching import apply_local_devices, DeviceSwitchError

__all__ = ["create_ui_router"]

_STATIC_DIR = Path(__file__).resolve().parents[2] / "frontend/workbench"


class RevealKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: Literal["llm.cloud.api_key", "vl.cloud_api_key"]


def _configured_keys(config):
    """与云端适配器一致：配置优先，缺失时使用指定环境变量。"""
    return {
        "llm.cloud.api_key": config.llm.cloud.resolved_key(),
        "vl.cloud_api_key": (config.vl.cloud_api_key or "").strip()
        or (os.environ.get(config.vl.cloud_api_key_env, "") or "").strip(),
    }


def create_ui_router(cfg: Any = None) -> APIRouter:
    router = APIRouter()
    settings_lock = threading.RLock()

    def _orch(request: Request) -> Any:
        orch = getattr(request.app.state, "orchestrator", None)
        if orch is None:
            raise HTTPException(status_code=503, detail="编排器未就绪")
        return orch

    # -- 页面 --------------------------------------------------------------- #
    @router.get("/ui", include_in_schema=False)
    def ui_page() -> FileResponse:
        index = _STATIC_DIR / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="ui 页面文件缺失")
        return FileResponse(index, media_type="text/html; charset=utf-8")

    # -- 设置（读写 data/user_settings.json；密钥打码读、空串=清除）--------- #
    @router.get(
        "/internal/settings", tags=["settings"], summary="当前网页设置（密钥打码）"
    )
    def get_settings(request: Request) -> JSONResponse:
        orch = _orch(request)
        from backend.config import get_app_config, load_app_config

        app_cfg = getattr(request.app.state, "config", None) or get_app_config()
        fresh = load_app_config(root=app_cfg.root_dir)
        caps = device_capabilities()
        return JSONResponse({
            "values": current_settings(app_cfg.root_dir),
            "secret_present": {key: bool(value) for key, value in _configured_keys(fresh).items()},
            "effective": effective_settings(fresh, caps),
            "capabilities": caps,
            "active_devices": active_devices(orch),
            "backends": [asdict(b) for b in describe_backends(app_cfg)],
            "hot_switch_supported": True,
        }, headers={"Cache-Control": "no-store"})

    @router.post("/internal/settings/reveal-key", tags=["settings"], summary="按需显示当前配置的 API Key")
    def reveal_key(request: Request, payload: RevealKeyRequest) -> JSONResponse:
        _orch(request)
        from backend.config import get_app_config, load_app_config

        app_cfg = getattr(request.app.state, "config", None) or get_app_config()
        with settings_lock:
            fresh = load_app_config(root=app_cfg.root_dir)
            value = _configured_keys(fresh)[payload.key]
        # 沿用本机、同源访问保护；明文不进入 URL、日志或普通设置响应。
        return JSONResponse({"value": value}, headers={"Cache-Control": "no-store"})

    @router.post(
        "/internal/settings", tags=["settings"], summary="保存网页设置（白名单校验）"
    )
    def post_settings(
        request: Request, payload: dict[str, Any] = Body(...)
    ) -> JSONResponse:
        orch = _orch(request)
        from backend.config import get_app_config

        app_cfg = getattr(request.app.state, "config", None) or get_app_config()

        with settings_lock:
            try:
                clean, fresh = preview_settings(dict(payload), root=app_cfg.root_dir)
                changed = apply_local_devices(
                    orch, fresh, lambda: save_settings(clean, root=app_cfg.root_dir)
                )
            except (SettingsError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except DeviceSwitchError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"写入设置失败: {exc}") from exc
            runtime_cfg = getattr(orch, "_cfg", None) or app_cfg
            request.app.state.config = runtime_cfg
            from backend.config import set_app_config

            set_app_config(runtime_cfg)
            saved_values = current_settings(app_cfg.root_dir)
            active_snapshot = active_devices(orch)
        caps = device_capabilities()
        restart_required = any(
            getattr(fresh, key) != getattr(runtime_cfg, key)
            for key in ("llm", "vl", "retrieval")
        )
        return JSONResponse({
            "saved": True,
            "values": saved_values,
            "secret_present": {key: bool(value) for key, value in _configured_keys(fresh).items()},
            "effective": effective_settings(fresh, caps),
            "capabilities": caps,
            "active_devices": active_snapshot,
            "backends": [asdict(b) for b in describe_backends(runtime_cfg)],
            "switched_models": changed,
            "hot_switch_supported": True,
            "restart_required": restart_required,
            "note": "已保存；本地模型设备与报告显示模式已生效，无需重启。"
            + ("云端或其他后端设置也有变更，这部分需重启生效。" if restart_required else ""),
        }, headers={"Cache-Control": "no-store"})

    return router
