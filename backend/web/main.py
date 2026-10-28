"""FastAPI 本机网页入口，启动前严格加载三个必需本地模型。"""

from __future__ import annotations
import asyncio
import json
import logging
import os
from urllib.parse import urlsplit
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from backend.config import AppConfig, load_app_config, set_app_config
from backend.logging_setup import configure_logging
from backend.pipeline.orchestrator import Orchestrator, PipelineError
from backend.runtime import SOURCE_ROOT, runtime_path
from backend.storage.repository import NotFoundError
from backend.web.dependencies import set_orchestrator, shutdown_jobs
from backend.web.routes import create_router
from backend.web.ui_page import create_ui_router
from backend.web.chat_page import create_chat_router

logger = logging.getLogger("zhiying.web")


def create_app(cfg: AppConfig | None = None, *, orchestrator=None) -> FastAPI:
    config = cfg or load_app_config(root=SOURCE_ROOT)

    @asynccontextmanager
    async def lifespan(application):
        set_app_config(config)
        configure_logging(config)
        from backend.media.ffmpeg_utils import resolve_binary

        resolve_binary("ffmpeg")
        resolve_binary("ffprobe")
        orch = orchestrator or await asyncio.to_thread(Orchestrator.from_config, config)
        application.state.orchestrator = orch
        set_orchestrator(orch)
        if getattr(orch,"_memory",None) is not None and not getattr(orch, "_cloud_setup_required", ()):
            orch._memory.notify()
        state = runtime_path("state/server.json", config.root_dir)
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "status": "running",
                    "executable": __import__("sys").executable,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        try:
            yield
        finally:
            await asyncio.to_thread(shutdown_jobs, wait=True)
            await asyncio.gather(
                *getattr(application.state, "qa_workers", ()), return_exceptions=True
            )
            orch.close()
            application.state.orchestrator = None
            set_orchestrator(None)
            state.unlink(missing_ok=True)

    application = FastAPI(
        title="知影网页工作台",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.state.config = config
    application.state.orchestrator = orchestrator

    @application.middleware("http")
    async def local_web_only(request: Request, call_next):
        host = request.url.hostname
        client = request.client.host if request.client else ""
        if host not in {
            "127.0.0.1",
            "localhost",
            "::1",
            "testserver",
        } or client not in {
            "127.0.0.1",
            "::1",
            "testclient",
        }:
            return JSONResponse({"detail": "仅允许本机网页访问"}, status_code=403)
        origin = request.headers.get("origin")
        if origin:
            actual, supplied = urlsplit(str(request.base_url)), urlsplit(origin)
            if (actual.scheme, actual.hostname, actual.port) != (
                supplied.scheme,
                supplied.hostname,
                supplied.port,
            ):
                return JSONResponse({"detail": "拒绝跨域网页请求"}, status_code=403)
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "拒绝跨站请求"}, status_code=403)
        return await call_next(request)

    application.include_router(create_router(config), prefix="/internal")
    application.include_router(create_ui_router(config))
    application.include_router(create_chat_router(config))
    application.mount(
        "/assets", StaticFiles(directory=SOURCE_ROOT / "frontend"), name="frontend"
    )

    @application.get("/")
    def index():
        return RedirectResponse("/ui")

    @application.exception_handler(NotFoundError)
    async def not_found(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @application.exception_handler(PipelineError)
    async def pipeline_error(request, exc):
        return JSONResponse({"detail": str(exc), "kind": "pipeline"}, status_code=400)

    @application.exception_handler(Exception)
    async def internal_error(request, exc):
        logger.exception("网页请求失败")
        return JSONResponse(
            {"detail": "处理失败，请查看运行目录中的日志", "kind": "internal"},
            status_code=500,
        )

    return application


app = create_app()


def main(argv=None):
    import uvicorn
    from backend.web.console import CONTROL_ENV, main as console_main, web_parser, start_control_reader

    if os.environ.get(CONTROL_ENV) != "1":
        return console_main(argv)
    args = web_parser().parse_args(argv)
    configure_logging(app.state.config)
    config = uvicorn.Config(
        "backend.web.main:app",
        log_config=None,
        host=args.host,
        port=args.port,
    )
    server = uvicorn.Server(config)
    start_control_reader(server)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
