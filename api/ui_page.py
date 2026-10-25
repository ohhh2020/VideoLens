"""分析工作台（/ui）—— 原生单页，与 /chat 同栈同设计语言。

为什么不再用 Gradio
--------------------
Gradio 6 把界面组件渲染在 iframe 沙箱里：父页 CSS 够不到（主题失效）、
`target="_top"` 导航被沙箱拦截（/chat 入口"接不上"的根因）、theme/css
API 又从 Blocks 挪到了 mount/launch。与其继续和框架搏斗，不如与 /chat
共用同一套原生 HTML/CSS/JS——零 CDN、零构建链，导航天然互通。

后端支撑：全部复用既有 REST（/api/videos/analyze 202 + /api/tasks/{id} 轮询
+ /api/videos 系列 + /api/stats），仅新增设置端点（/api/settings，读写
``data/user_settings.json``，密钥打码——与网页设置层 ``core/user_settings``
同一套白名单与校验）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dataclasses import asdict

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import FileResponse

from core.user_settings import (
    SettingsError,
    current_settings,
    describe_backends,
    save_settings,
)

logger = logging.getLogger("zhiying.api.ui")

__all__ = ["create_ui_router"]

_STATIC_DIR = Path(__file__).resolve().parent / "ui_static"


def create_ui_router(cfg: Any = None) -> APIRouter:
    router = APIRouter()

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
    @router.get("/api/settings", tags=["settings"], summary="当前网页设置（密钥打码）")
    def get_settings(request: Request) -> dict[str, Any]:
        _orch(request)
        from core.config import get_app_config  # noqa: PLC0415

        app_cfg = getattr(request.app.state, "config", None) or get_app_config()
        return {
            "values": current_settings(),
            "backends": [asdict(b) for b in describe_backends(app_cfg)],
        }

    @router.post("/api/settings", tags=["settings"], summary="保存网页设置（白名单校验）")
    def post_settings(request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        _orch(request)
        from core.config import get_app_config, load_app_config  # noqa: PLC0415

        try:
            save_settings(dict(payload))
        except SettingsError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"写入设置失败: {exc}") from exc
        # 重载配置让新设置立即反映到状态页；编排器仍按旧配置运行，
        # 重启服务后全面生效（与原 Gradio 设置页"保存即重建"相比保守，
        # 但语义诚实——见状态页提示）。
        try:
            fresh = load_app_config(force_reload=True)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"保存成功但配置校验失败: {exc}") from exc
        app_cfg = getattr(request.app.state, "config", None) or get_app_config()
        return {
            "saved": True,
            "values": current_settings(),
            "backends": [asdict(b) for b in describe_backends(fresh or app_cfg)],
            "note": "已保存；云端/模型设置的完整生效需要重启服务",
        }

    return router
