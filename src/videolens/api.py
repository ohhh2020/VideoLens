"""HTTP 接入层：只负责鉴权、输入输出和流式传输，业务交给 VideoService。

新手阅读顺序：create_app → require_token → upload_video → task_events。
API 令牌仅用于内部调用；本机网页无需账号密码，媒体通过本机代理访问。
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
import re
import tempfile
import time
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import load_settings
from .errors import VideoLensError

LOGGER = logging.getLogger(__name__)
TERMINAL_STATES = {"succeeded", "completed", "success", "partial", "partial_success", "failed", "cancelled", "canceled", "paused"}


class InputModel(BaseModel):
    """拒绝拼错的参数，避免用户以为某项设置已经生效。"""

    model_config = ConfigDict(extra="forbid")


class LocalImport(InputModel):
    path: str = Field(min_length=1)
    subtitle: str | None = None


class UrlImport(InputModel):
    url: str = Field(min_length=1, max_length=2048)


class TaskInput(InputModel):
    kind: str = Field(default="analyze", pattern="^(analyze|reindex)$")


class SearchInput(InputModel):
    query: str = Field(min_length=1, max_length=10000)


class AskInput(InputModel):
    question: str = Field(min_length=1, max_length=20000)
    stream: bool = False


class KeyInput(InputModel):
    key: str = Field(min_length=1, max_length=8192)


class ModelTransfer(InputModel):
    path: str = ""
    url: str = ""
    name: str = Field(default="", max_length=160)
    sha256: str = ""


class ModelActivation(InputModel):
    gpu_layers: int = Field(default=0, ge=0, le=999)
    context: int = Field(default=8192, ge=1024, le=131072)


def sse_event(event: str, data: Any, event_id: int | None = None) -> str:
    """把一个完整事件编码成 SSE；JSON 会转义换行，防止破坏事件边界。"""
    safe_event = event.replace("\n", "").replace("\r", "")
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return f"{prefix}event: {safe_event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _loopback(host: str | None) -> bool:
    """本地路径导入会读取服务所在机器的文件，因此只允许回环地址。"""
    if host in {"localhost", "testclient"}:
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def create_app(settings: Any = None, service: Any = None, with_ui: bool = True) -> FastAPI:
    """构建应用；注入 service 使测试可以验证协议而不用下载模型。"""
    settings = settings or load_settings()
    from .logging_setup import configure_logging
    from .paths import configure_paths
    configure_logging(configure_paths())
    if service is None:
        from .service import VideoService
        service = VideoService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # SQLite 恢复和后台队列由业务服务统一管理，Web 重启不会创建第二套队列。
        await run_in_threadpool(service.start)
        try:
            yield
        finally:
            await run_in_threadpool(service.close)
            if hasattr(app.state, "local_models"):
                await run_in_threadpool(app.state.local_models.close)

    app = FastAPI(title="知影 VideoLens 本地 API", version="1.0.0", lifespan=lifespan,
                  description="视频分析、检索、可追溯问答。除健康检查外使用 Bearer 令牌。",
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.service = service
    app.state.settings = settings
    requests_by_client: dict[str, deque[float]] = defaultdict(deque)

    @app.exception_handler(VideoLensError)
    async def service_error(request: Request, exc: VideoLensError):
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse({"error": {"code": f"http_{exc.status_code}", "message": str(exc.detail)}},
                            status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request: Request, exc: RequestValidationError):
        # Pydantic 原始错误包含用户输入；密钥写入出错时也不应把密钥回显。
        fields = [".".join(str(part) for part in error["loc"]) for error in exc.errors()]
        return JSONResponse({"error": {"code": "invalid_input", "message": "请求参数不符合要求", "fields": fields}},
                            status_code=422)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        # 外部服务的异常可能包含 URL、密钥、原文或绝对路径，公开响应不携带它们。
        LOGGER.error("接口执行失败，异常类型：%s", type(exc).__name__)
        return JSONResponse({"error": {"code": "internal_error", "message": "操作失败，请查看任务状态或运行环境诊断。"}}, status_code=500)

    bearer = HTTPBearer(auto_error=False, scheme_name="本地访问令牌")

    async def require_token(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        scheme, token = (credentials.scheme, credentials.credentials) if credentials else ("", "")
        expected = str(settings.api_token)
        if scheme.lower() != "bearer" or not token or not hmac.compare_digest(token, expected):
            raise HTTPException(401, "需要有效的本地访问令牌", headers={"WWW-Authenticate": "Bearer"})
        # 进度轮询、媒体播放不计入写操作限流，避免正常观看和轮询耗尽额度。
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            host = request.client.host if request.client else "unknown"
            queue = requests_by_client[host]
            now = time.monotonic()
            while queue and queue[0] <= now - 60:
                queue.popleft()
            if len(queue) >= int(getattr(settings, "rate_limit_per_minute", 60)):
                raise HTTPException(429, "操作过于频繁，请稍后重试", headers={"Retry-After": "60"})
            queue.append(now)

    @app.get("/health", tags=["健康检查"])
    def health():
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def index():
        return RedirectResponse("/app" if with_ui else "/health")

    router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_token)])

    @router.get("/openapi.json", tags=["文档"])
    def openapi_schema():
        return app.openapi()

    @router.get("/videos", tags=["视频"])
    def list_videos():
        return service.list_videos()

    @router.post("/videos/import", status_code=201, tags=["视频"])
    def import_video(body: LocalImport, request: Request):
        if not _loopback(request.client.host if request.client else None):
            raise HTTPException(403, "通过本地路径导入只允许本机调用；远程请上传文件")
        return service.import_file(Path(body.path), Path(body.subtitle) if body.subtitle else None)

    async def save_upload(upload: UploadFile, folder: Path, limit: int) -> Path:
        # 每次只读 1 MiB；4 GB 视频也不会同时占用 4 GB 内存。
        # 使用独立临时目录解决重名，同时保留中文原文件名供视频库作为标题。
        name = Path((upload.filename or "upload.bin").replace("\\", "/")).name
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .") or "upload.bin"
        suffix = Path(name).suffix[:12]
        stem = Path(name).stem[:160]
        if stem.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
            stem = "upload-" + stem
        directory = Path(tempfile.mkdtemp(prefix="upload-", dir=folder))
        target = directory / (stem + suffix)
        with target.open("xb") as handle:
            size = 0
            try:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, "上传文件超过配置的大小限制")
                    await run_in_threadpool(handle.write, chunk)
                if not size:
                    raise HTTPException(422, "上传文件不能为空")
            except BaseException:
                handle.close()
                target.unlink(missing_ok=True)
                directory.rmdir()
                raise
            finally:
                await upload.close()
        return target

    def remove_upload(path: Path) -> None:
        """仅删除本次产生的文件和空目录，不递归删除任何用户路径。"""
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass

    @router.post("/videos/upload", status_code=201, tags=["视频"])
    async def upload_video(file: UploadFile = File(...), subtitle: UploadFile | None = File(None)):
        folder = Path(settings.data_dir) / "tmp" / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        try:
            path = await save_upload(file, folder, int(getattr(settings, "max_video_bytes", 4_000_000_000)))
            paths.append(path)
            subtitle_path = None
            if subtitle and subtitle.filename:
                subtitle_path = await save_upload(subtitle, folder, 20 * 1024 * 1024)
                paths.append(subtitle_path)
            # import_file 必须先复制到受管理目录，再允许这里清除上传临时文件。
            return await run_in_threadpool(service.import_file, path, subtitle_path)
        finally:
            for path in paths:
                remove_upload(path)

    @router.post("/videos/url", status_code=201, tags=["视频"])
    def import_url(body: UrlImport):
        return service.import_url(body.url)

    @router.get("/videos/{video_id}", tags=["视频"])
    def get_video(video_id: str):
        return service.get_video(video_id)

    @router.delete("/videos/{video_id}", tags=["视频"])
    def delete_video(video_id: str):
        return service.delete_video(video_id)

    @router.get("/videos/{video_id}/media", tags=["媒体"])
    def video_media(video_id: str):
        return FileResponse(service.media_path(video_id))

    @router.get("/videos/{video_id}/frames/{frame_id}", tags=["媒体"])
    def frame_media(video_id: str, frame_id: str):
        return FileResponse(service.media_path(video_id, frame_id))

    @router.get("/videos/{video_id}/export", tags=["导出"])
    def export_video(video_id: str, format: str = "markdown"):
        if format not in {"markdown", "json"}:
            raise HTTPException(422, "导出格式只支持 markdown 或 json")
        content = service.export(video_id, format)
        extension = "json" if format == "json" else "md"
        return Response(content, media_type="application/json" if format == "json" else "text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="videolens-report.{extension}"'})

    @router.post("/videos/{video_id}/tasks", status_code=202, tags=["任务"])
    def create_task(video_id: str, body: TaskInput):
        return service.create_task(video_id, body.kind)

    @router.get("/tasks", tags=["任务"])
    def list_tasks():
        return service.list_tasks()

    @router.get("/tasks/{task_id}", tags=["任务"])
    def get_task(task_id: str):
        return service.get_task(task_id)

    @router.post("/tasks/{task_id}/{action}", tags=["任务"])
    def control_task(task_id: str, action: str):
        if action not in {"pause", "resume", "cancel"}:
            raise HTTPException(422, "任务操作只支持 pause、resume、cancel")
        return service.control_task(task_id, action)

    @router.get("/tasks/{task_id}/events", tags=["任务"])
    async def task_events(task_id: str, request: Request, after: int = 0):
        await run_in_threadpool(service.get_task, task_id)
        try:
            cursor = max(after, int(request.headers.get("last-event-id", "0")))
        except ValueError as exc:
            raise HTTPException(422, "Last-Event-ID 必须是整数") from exc

        async def events():
            nonlocal cursor
            last_ping = time.monotonic()
            while not await request.is_disconnected():
                rows = await run_in_threadpool(service.task_events, task_id, cursor)
                for row in rows:
                    event_id = int(row["id"])
                    if event_id <= cursor:
                        continue
                    cursor = event_id
                    yield sse_event(str(row.get("event", "progress")), row.get("data", {}), event_id)
                task = await run_in_threadpool(service.get_task, task_id)
                if task.get("status", task.get("state")) in TERMINAL_STATES:
                    yield sse_event("done", task)
                    break
                if time.monotonic() - last_ping >= 15:
                    yield ": 保持连接\n\n"
                    last_ping = time.monotonic()
                await asyncio.sleep(0.25)

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @router.post("/videos/{video_id}/search", tags=["检索"])
    def search(video_id: str, body: SearchInput):
        return service.search(video_id, body.query)

    @router.post("/videos/{video_id}/search-image", tags=["检索"])
    async def search_image(video_id: str, image: UploadFile = File(...), query: str = Form("")):
        folder = Path(settings.data_dir) / "tmp" / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        path = await save_upload(image, folder, 20 * 1024 * 1024)
        try:
            return await run_in_threadpool(service.search, video_id, query, path)
        finally:
            remove_upload(path)

    @router.post("/videos/{video_id}/sessions", status_code=201, tags=["对话"])
    def create_session(video_id: str):
        return service.create_session(video_id)

    @router.get("/videos/{video_id}/sessions", tags=["对话"])
    def list_sessions(video_id: str):
        return service.list_sessions(video_id)

    @router.get("/sessions/{session_id}/messages", tags=["对话"])
    def session_messages(session_id: str):
        return service.session_messages(session_id)

    @router.post("/sessions/{session_id}/ask", tags=["对话"])
    async def ask(session_id: str, body: AskInput, request: Request):
        # 在发送 HTTP 200 之前确认会话存在；流开始后只能通过 error 事件传递失败。
        await run_in_threadpool(service.session_messages, session_id)
        if not body.stream:
            return await run_in_threadpool(service.ask, session_id, body.question)

        async def answer_events():
            # 业务生成器在 yield 之间持有 RLock。通用线程池会让不同 next() 落在
            # 不同线程，导致结束时“不能释放未持有的锁”。每条流固定一个工作线程，
            # 连同 close() 也交给该线程，保证断流后资源按正确的线程所有权释放。
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="videolens-answer")
            iterator = None
            loop = asyncio.get_running_loop()

            def advance(generator):
                try:
                    return True, next(generator)
                except StopIteration:
                    return False, None

            try:
                iterator = service.ask_stream(session_id, body.question)
                while True:
                    available, item = await loop.run_in_executor(executor, advance, iterator)
                    if not available:
                        break
                    if await request.is_disconnected():
                        break
                    yield sse_event(item.get("event", "message"), item.get("data", {}))
            except VideoLensError as exc:
                yield sse_event("error", {"code": exc.code, "message": exc.message})
            except Exception as exc:
                LOGGER.error("问答流失败，异常类型：%s", type(exc).__name__)
                yield sse_event("error", {"code": "stream_failed", "message": "回答生成中断，请重试或检查模型配置。"})
            finally:
                if iterator is not None and hasattr(iterator, "close"):
                    # 提交后再关闭执行器，未完成的 next() 会先完成，随后在原线程关闭。
                    cleanup = executor.submit(iterator.close)
                    try:
                        await asyncio.shield(asyncio.wrap_future(cleanup))
                    finally:
                        executor.shutdown(wait=False)
                else:
                    executor.shutdown(wait=False)

        return StreamingResponse(answer_events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @router.get("/settings", tags=["设置"])
    def read_settings():
        return service.public_settings()

    @router.patch("/settings", tags=["设置"])
    def update_settings(body: dict[str, Any]):
        return service.update_settings(body)

    @router.put("/keys/{reference}", tags=["设置"])
    def store_key(reference: str, body: KeyInput):
        service.store_key(reference, body.key)
        return {"saved": True, "reference": reference}

    @router.post("/providers/{kind}/check", tags=["设置"])
    def check_provider(kind: str):
        if kind not in {"text", "vision", "local"}:
            raise HTTPException(422, "模型用途只支持 text、vision、local")
        return service.check_provider(kind)

    @router.get("/doctor", tags=["维护"])
    def doctor():
        return service.doctor()

    @router.post("/cleanup", tags=["维护"])
    def cleanup():
        return service.cleanup()

    # 延迟初始化，构建网页/协议测试时不会自动下载或启动模型。
    manager_lock = threading.Lock()

    def model_manager():
        with manager_lock:
            if not hasattr(app.state, "local_models"):
                from .local_models import LocalModels
                app.state.local_models = LocalModels(service)
            return app.state.local_models

    @router.get("/local-models", tags=["本地模型"])
    def local_models():
        return model_manager().listing()

    @router.post("/local-models/import", tags=["本地模型"], status_code=202)
    def import_model(body: ModelTransfer, request: Request):
        if not _loopback(request.client.host if request.client else None):
            raise HTTPException(403, "只能从本机导入模型")
        return model_manager().submit("import", **body.model_dump())

    @router.post("/local-models/download", tags=["本地模型"], status_code=202)
    def download_model(body: ModelTransfer):
        return model_manager().submit("download", **body.model_dump())

    @router.post("/local-models/runtime/{variant}", tags=["本地模型"], status_code=202)
    def deploy_runtime(variant: str):
        if variant not in {"cpu", "cuda"}:
            raise HTTPException(422, "仅支持 cpu 或 cuda")
        return model_manager().submit("runtime", variant=variant)

    @router.post("/local-models/stop", tags=["本地模型"])
    def stop_model():
        return model_manager().stop()

    @router.post("/local-models/{model_id}/start", tags=["本地模型"])
    def activate_model(model_id: str, body: ModelActivation):
        return model_manager().activate(model_id, body.gpu_layers, body.context)

    @router.delete("/local-models/{model_id}", tags=["本地模型"])
    def delete_model(model_id: str):
        return model_manager().delete(model_id)

    app.include_router(router)
    if with_ui:
        from .ui import mount_ui
        app = mount_ui(app, settings, service)
    return app
