"""C 端对话页（/chat）—— 双栏视频问答界面的路由与支撑端点。

设计取舍
--------
* 页面是**单文件原生 HTML/CSS/JS**（零 CDN、零构建链），由本模块托管 ——
  本地优先哲学的延续：断网也能用，Python 团队可直接维护。
* 视频文件走 :meth:`FileResponse`（Starlette 内建 Range 支持，播放器拖动
  进度条才不会整段重下）；按 ID 查库定位本地路径，不接受客户端路径。
* 会话端点是对 ``core.pipeline.chat_sessions`` 的薄封装：历史用于刷新后
  恢复对话流，DELETE 实现"清空对话"。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from core.pipeline.chat_sessions import get_chat_store
from core.storage.repository import StorageError

logger = logging.getLogger("zhiying.api.chat")

__all__ = ["create_chat_router"]

_STATIC_DIR = Path(__file__).resolve().parent / "chat_static"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ms(ms: int) -> str:
    total = max(0, int(ms))
    hours, rem = divmod(total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def create_chat_router(cfg: Any = None) -> APIRouter:
    router = APIRouter()
    auth = Depends(_auth_dep())

    def _orch(request: Request) -> Any:
        orch = getattr(request.app.state, "orchestrator", None)
        if orch is None:
            raise HTTPException(status_code=503, detail="编排器未就绪")
        return orch

    # -- 页面 --------------------------------------------------------------- #
    @router.get("/chat", include_in_schema=False)
    def chat_page() -> FileResponse:
        index = _STATIC_DIR / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="chat 页面文件缺失")
        return FileResponse(index, media_type="text/html; charset=utf-8")

    # -- 会话（SQLite 持久化 + 内存多轮前缀）-------------------------------- #
    def _repo(request: Request) -> Any:
        return _orch(request).repository

    @router.get(
        "/api/videos/{video_id}/chat",
        tags=["chat"],
        summary="对话历史（持久化；session_id 过滤，缺省全部）",
    )
    def chat_history(
        request: Request, video_id: str, session_id: str | None = None
    ) -> dict[str, Any]:
        _orch(request)
        turns = _repo(request).list_qa(video_id, session_id=session_id or None)
        return {"video_id": video_id, "count": len(turns), "turns": turns}

    @router.get(
        "/api/videos/{video_id}/chat/sessions",
        tags=["chat"],
        summary="历史会话列表（按 session_id 聚合）",
    )
    def chat_sessions(request: Request, video_id: str) -> dict[str, Any]:
        _orch(request)
        return {"video_id": video_id, "sessions": _repo(request).list_qa_sessions(video_id)}

    @router.delete(
        "/api/videos/{video_id}/chat",
        tags=["chat"],
        summary="清空该视频的对话历史（可按 session_id）",
        dependencies=[auth],
    )
    def chat_clear(
        request: Request, video_id: str, session_id: str | None = None
    ) -> dict[str, Any]:
        _orch(request)
        get_chat_store().clear(video_id)
        removed = _repo(request).clear_qa(video_id, session_id=session_id or None)
        return {"video_id": video_id, "cleared": True, "removed": removed}

    @router.get(
        "/api/videos/{video_id}/chat/export",
        tags=["chat"],
        summary="导出对话历史为 Markdown（带引用时间戳）",
        dependencies=[auth],
    )
    def chat_export(
        request: Request, video_id: str, session_id: str | None = None
    ) -> FileResponse:
        orch = _orch(request)
        repo = orch.repository
        try:
            video = repo.find_video(video_id)
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if video is None:
            raise HTTPException(status_code=404, detail=f"视频不存在: {video_id}")
        turns = repo.list_qa(video_id, session_id=session_id or None)
        if not turns:
            raise HTTPException(status_code=404, detail="该视频还没有问答历史")
        title = video.title or video.display_name or video_id
        lines = [
            f"# 对话记录 · {title}",
            "",
            f"视频 `video_id`：`{video_id}` · {len(turns)} 轮 · 导出于 {_now_iso()}",
            "",
        ]
        for i, t in enumerate(turns, start=1):
            lines.append(f"## Q{i} {t['question']}")
            lines.append("")
            lines.append(t["answer"])
            lines.append("")
            if t["refs"]:
                lines.append("<details><summary>证据</summary>")
                lines.append("")
                for r in t["refs"]:
                    lo, hi = int(r.get("start_ms", 0)), int(r.get("end_ms", 0))
                    lines.append(
                        f"- [{r.get('index')}] {_ms(lo)}–{_ms(hi)} `{r.get('clip_id', '')}`"
                    )
                lines.append("")
                lines.append("</details>")
                lines.append("")
        name = f"chat-{video_id}{('-' + session_id) if session_id else ''}.md"
        out_dir = Path(cfg.path("exports")) if cfg is not None else Path("data/exports")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / name
        out.write_text("\n".join(lines), encoding="utf-8")
        logger.info("对话记录已导出: %s", out)
        return FileResponse(out, media_type="text/markdown; charset=utf-8", filename=name)

    # -- 视频文件流（播放器用；Range 由 Starlette FileResponse 处理）-------- #
    @router.get(
        "/api/videos/{video_id}/file",
        tags=["chat"],
        summary="视频文件（HTTP Range 流式播放）",
        dependencies=[auth],
    )
    def video_file(request: Request, video_id: str) -> FileResponse:
        orch = _orch(request)
        try:
            record = orch.repository.find_video(video_id)
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if record is None:
            raise HTTPException(status_code=404, detail=f"视频不存在: {video_id}")
        path = Path(str(getattr(record, "local_path", "") or ""))
        if not path.is_file():
            raise HTTPException(status_code=410, detail="视频文件已不在本机（可重新分析恢复）")
        return FileResponse(
            path,
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes", "Cache-Control": "no-store"},
        )

    return router


def _auth_dep():
    """复用 REST 层的鉴权判据（避免循环 import 延迟导入）。"""
    from api.routes import require_auth  # noqa: PLC0415

    return require_auth
