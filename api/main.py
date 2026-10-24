"""知影 HTTP 服务入口 —— FastAPI 应用装配。

职责边界
--------
``main.py`` 只做三件事：**装配**（配置 → 编排器）、**挂载**（路由 / CORS /
异常处理）、**生命周期**（启动预热、关闭释放）。业务逻辑一律不在这里。

启动策略：**装配失败不阻止启动**。理由是本地优先工具的常见故障是"模型没下完"
或"Chroma 目录被占用"，这时用户更需要一个能打开的界面去触发
``scripts/download_models.py``，而不是一个连不上的端口。失败原因写进
``/api/health`` 的 ``orchestrator`` 字段与日志，业务接口返回 503。

用法::

    python -m api.main                       # 用 configs/app_config.yaml
    uvicorn api.main:app --port 8000         # 同上（模块级 app 已装配）
    ZH_YING__API__PORT=9000 python -m api.main

环境变量覆盖规则见 :mod:`core.config`（``ZH_YING__<段>__<字段>``）。
"""

from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from api.mcp_server import create_http_router
from api.routes import create_router, set_orchestrator, shutdown_jobs
from core.config import AppConfig, load_app_config
from core.pipeline.orchestrator import Orchestrator, PipelineError
from core.storage.repository import NotFoundError, RepositoryError, StorageError

logger = logging.getLogger("zhiying.api")

__all__ = ["create_app", "app", "configure_logging", "main"]


# =============================================================================
# 日志
# =============================================================================


def configure_logging(cfg: AppConfig) -> None:
    """按配置初始化根 logger（幂等：重复调用不会叠加 handler）。

    需求文档要求"日志不得记录密钥与用户隐私"：这里只控制输出位置与级别，
    脱敏责任在 ``_redact`` 与各模块"不打日志原文/密钥"的纪律上。
    """
    root = logging.getLogger()
    if getattr(root, "_zhiying_configured", False):
        root.setLevel(getattr(logging, cfg.logging.level.upper(), logging.INFO))
        return

    level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    fmt = logging.Formatter(cfg.logging.format, datefmt=cfg.logging.date_format)
    root.setLevel(level)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    try:
        # 注意：`paths.logs` 是**目录**，日志文件是 `logging.file`（默认
        # logs/zhiying.log）。早先这里误把目录当文件传给 FileHandler，结果
        # 建出一个叫 `logs` 的普通文件，把目录位置占了，初始化目录直接失败。
        log_path = Path(cfg.logging.file)
        if not log_path.is_absolute():
            log_path = Path(cfg.root_dir) / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=cfg.logging.max_bytes,
            backupCount=cfg.logging.backup_count,
            encoding="utf-8",
        )
        rotating.setFormatter(fmt)
        root.addHandler(rotating)
    except OSError as exc:  # 只读目录 / 权限不足：退化为仅控制台
        logger.warning("日志文件不可写，仅输出到控制台: %s", exc)

    # 第三方库降噪：uvicorn 的 access 日志保留，httpx/chromadb 只留警告。
    for noisy in ("httpx", "urllib3", "chromadb", "modelscope", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root._zhiying_configured = True  # type: ignore[attr-defined]


# =============================================================================
# 应用工厂
# =============================================================================


def create_app(
    cfg: AppConfig | None = None,
    *,
    orchestrator: Orchestrator | None = None,
    load_models: bool = True,
    serve_ui: bool | None = None,
) -> FastAPI:
    """构造 FastAPI 应用。

    Args:
        cfg: 配置对象；``None`` 时用 :func:`core.config.load_app_config`。
        orchestrator: 直接注入编排器（测试用，跳过重量级装配）。
        load_models: 为 ``False`` 时只装配轻量组件（不下载/加载权重），
            供无模型环境做接口联调。
        serve_ui: 是否把 Gradio 界面挂到 ``/ui``。``None`` 时读
            ``api.serve_ui``。挂载会 import gradio（连带 transformers 等），
            因此模块级 ``app`` 显式关闭，只有 ``python -m api.main`` 打开。
    """
    config = cfg or load_app_config()
    configure_logging(config)
    with_ui = config.api.serve_ui if serve_ui is None else serve_ui

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> Any:
        """启动装配 / 关闭释放。"""
        orch = orchestrator
        if orch is None:
            try:
                # 装配会触达模型目录与 Chroma，属于阻塞 IO；丢到线程里跑，
                # 让事件循环在启动阶段仍能响应 /api/health。
                orch = await asyncio.to_thread(
                    _build_orchestrator, config, load_models
                )
            except Exception as exc:  # noqa: BLE001 - 装配失败降级启动
                logger.exception("编排器装配失败，服务以降级模式启动: %s", exc)
                application.state.assembly_error = f"{type(exc).__name__}: {exc}"
                orch = None
            else:
                application.state.assembly_error = ""
        application.state.orchestrator = orch
        set_orchestrator(orch)
        if orch is not None:
            logger.info("编排器就绪: %r", orch)
        try:
            yield
        finally:
            logger.info("服务关闭，释放资源")
            shutdown_jobs(wait=False)
            if application.state.orchestrator is not None:
                try:
                    application.state.orchestrator.close()
                except Exception:  # noqa: BLE001 - 关闭阶段的异常不再上抛
                    logger.warning("释放编排器资源时出错", exc_info=True)
            application.state.orchestrator = None
            set_orchestrator(None)

    application = FastAPI(
        title="知影 ZhiYing",
        description=(
            "本地优先的视频智能分析助手：转写 → 切片 → 关键帧 → 多模态检索 → 带引用的问答。\n\n"
            "**所有返回的时间戳均为毫秒精度，并携带 video_id / clip_id 溯源信息。**"
        ),
        version=str(config.app.version),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    application.state.config = config
    application.state.orchestrator = orchestrator
    application.state.assembly_error = ""

    if config.api.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.api.cors_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-API-Token"],
        )

    _install_error_handlers(application)
    application.include_router(create_router(config), prefix="/api")
    # MCP over HTTP：与 REST 共用鉴权与编排器，路径 /api/mcp（Obsidian/远程客户端用）。
    application.include_router(create_http_router(), prefix="/api")

    @application.get("/", include_in_schema=False)
    def index() -> Any:
        """根路径：挂了界面就直接进界面，否则给一页入口导航。"""
        target = getattr(application.state, "ui_path", "")
        if target:
            return RedirectResponse(url=target)
        return HTMLResponse(_LANDING_HTML)

    @application.get("/health", include_in_schema=False)
    def root_health() -> dict[str, Any]:
        """不带 ``/api`` 前缀的探活（容器/K8s 习惯放根路径）。"""
        return {
            "status": "ok" if application.state.orchestrator is not None else "degraded",
            "version": str(config.app.version),
        }

    application.state.ui_path = _mount_ui(application, config) if with_ui else ""

    return application


def _build_orchestrator(cfg: AppConfig, load_models: bool) -> Orchestrator:
    """在启动线程里装配编排器；把异常原样抛给调用方决定降级。"""
    return Orchestrator.from_config(cfg, load_models=load_models)


# =============================================================================
# 内嵌 Web 界面
# =============================================================================

_LANDING_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>知影 ZhiYing</title>
<style>
  body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
       background:#0f1115;color:#e6e8ee;font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
  .card{max-width:640px;padding:36px 40px;background:#171a21;border:1px solid #262b36;border-radius:14px}
  h1{margin:0 0 6px;font-size:22px;font-weight:600}
  p.sub{margin:0 0 22px;color:#98a2b3;font-size:13px}
  a{display:block;padding:11px 14px;margin:8px 0;border:1px solid #2e3542;border-radius:9px;
    color:#e6e8ee;text-decoration:none;transition:.15s}
  a:hover{border-color:#4f7cff;background:#1d2330}
  code{color:#9db4ff;font-family:ui-monospace,Consolas,monospace}
  .warn{margin-top:18px;padding:10px 12px;background:#2a2113;border:1px solid #4a3a17;
        border-radius:8px;color:#e8c97a;font-size:13px}
</style></head><body>
<div class="card">
  <h1>知影 ZhiYing</h1>
  <p class="sub">本地优先的视频智能分析服务</p>
  <a href="http://127.0.0.1:7860">→ 打开独立 Web 界面（7860，需另起 python -m clients.gradio_ui）</a>
  <a href="/docs">→ REST API 文档（Swagger）</a>
  <a href="/api/health">→ 服务健康检查</a>
  <div class="warn">Web 界面未挂载。用 <code>python -m api.main</code> 或 <code>start.bat web</code>
  启动，界面会出现在本地址；也可单独运行 <code>python -m clients.gradio_ui</code>（7860 端口）。</div>
</div></body></html>
"""


class _OrchestratorProxy:
    """把界面侧的编排器调用转发到 lifespan 里才装配好的实例。

    界面必须在 ``create_app`` 时就建好（Gradio 的路由要一次性挂载），
    而真正的编排器在启动钩子里才装配 —— 中间这段时间用代理顶住，
    调用到达时才取实例，避免出现"界面在、模型没装配"的假成功。
    """

    def __init__(self, application: FastAPI) -> None:
        self._app = application

    @property
    def config(self) -> AppConfig:
        return self._app.state.config

    def _target(self) -> Any:
        orch = getattr(self._app.state, "orchestrator", None)
        if orch is None:
            reason = getattr(self._app.state, "assembly_error", "") or "正在启动"
            raise RuntimeError(f"分析引擎尚未就绪：{reason}")
        return orch

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target(), name)


def _mount_ui(application: FastAPI, config: AppConfig) -> str:
    """把 Gradio 界面挂到 ``/ui``，返回挂载路径；失败返回空串（不影响 API）。"""
    try:
        import gradio as gr  # noqa: PLC0415
        from clients.gradio_ui import build_ui  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - 缺 gradio 也要能起 API
        logger.warning("Gradio 不可用，跳过内嵌界面: %s", exc)
        return ""
    try:
        demo = build_ui(orchestrator=_OrchestratorProxy(application), load_models=False)
        gr.mount_gradio_app(application, demo, path="/ui")
    except Exception as exc:  # noqa: BLE001
        logger.warning("内嵌 Web 界面挂载失败: %s", exc)
        return ""
    logger.info("Web 界面已挂载: /ui（根路径会重定向到它）")
    return "/ui"


# =============================================================================
# 异常处理
# =============================================================================


def _install_error_handlers(application: FastAPI) -> None:
    """统一错误信封：``{"detail": str, "kind": str}``。

    500 一律不回内部路径或堆栈 —— 只写日志。
    """

    @application.exception_handler(NotFoundError)
    async def _not_found(_request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc), "kind": "not_found"},
        )

    @application.exception_handler(PipelineError)
    async def _pipeline_error(_request: Request, exc: PipelineError) -> JSONResponse:
        logger.warning("流水线错误: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc), "kind": "pipeline"},
        )

    @application.exception_handler(StorageError)
    async def _storage_error(_request: Request, exc: StorageError) -> JSONResponse:
        logger.error("存储错误: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "存储层错误，请查看服务日志", "kind": "storage"},
        )

    @application.exception_handler(RepositoryError)
    async def _repo_error(_request: Request, exc: RepositoryError) -> JSONResponse:
        logger.error("仓储错误: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc), "kind": "repository"},
        )

    @application.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("未处理异常")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "服务内部错误，请查看日志",
                "kind": "internal",
                "type": type(exc).__name__,
            },
        )


# =============================================================================
# 模块级应用 + CLI 入口
# =============================================================================

#: ``uvicorn api.main:app`` / ``python -m api.main`` 共用。
#: 注意：这里的装配发生在 import 时，因此**不做**重量级模型加载 ——
#: 真正的加载在 lifespan 里（``create_app`` 的 ``orchestrator=None`` 路径）。
#: 界面显式关掉：import 时拉 gradio 会连带加载 transformers，让
#: ``pytest``/脚本引用本模块变慢十几秒。``python -m api.main`` 会重开。
app = create_app(serve_ui=False)

#: 延迟装配开关：``ZHIYING_NO_MODEL_LOAD=1`` 时只跑接口不碰权重。
_LOAD_MODELS = os.environ.get("ZHIYING_NO_MODEL_LOAD", "").strip() not in {"1", "true", "yes"}


def main(argv: list[str] | None = None) -> int:
    """命令行入口：``python -m api.main [--host H] [--port P] [--reload]``。"""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="api.main", description="启动知影 HTTP 服务")
    parser.add_argument("--host", default=None, help="监听地址（默认取配置 api.host）")
    parser.add_argument("--port", type=int, default=None, help="端口（默认取配置 api.port）")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    parser.add_argument(
        "--no-models", action="store_true", help="不加载模型（仅接口联调）"
    )
    args = parser.parse_args(argv)

    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        print("缺少依赖：pip install 'uvicorn[standard]'", file=sys.stderr)
        return 2

    cfg = load_app_config()
    host = args.host or cfg.api.host
    port = args.port or cfg.api.port
    token_env = cfg.api.auth_token_env
    if not os.environ.get(token_env):
        logger.warning(
            "未设置 %s：仅允许本机访问。局域网/远程访问请先设置该环境变量。", token_env
        )

    if args.reload:
        # reload 模式由 uvicorn 自己起子进程导入 app，模型加载交给 lifespan。
        # 走 factory 而不是字符串 app：模块级 app 关了界面（import 更快）。
        uvicorn.run(
            "api.main:_dev_app",
            host=host,
            port=port,
            reload=True,
            factory=True,
            log_config=None,
        )
        return 0

    global _LOAD_MODELS
    if args.no_models:
        _LOAD_MODELS = False
    application = create_app(cfg, load_models=_LOAD_MODELS, serve_ui=True)
    uvicorn.run(application, host=host, port=port, log_config=None)
    return 0


def _dev_app() -> FastAPI:
    """``uvicorn --reload`` 用的应用工厂（带界面）。"""
    return create_app(load_models=_LOAD_MODELS, serve_ui=True)


def _project_root_marker() -> Path:
    return Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(main())
