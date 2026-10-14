"""网页本地模型仓库：流式复制/下载，校验后发布，按 ID 操作受管文件。"""
from __future__ import annotations

import hashlib
import logging
import re
import shutil
import threading
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import httpx

from .bootstrap import managed_process, read_json, start_local, stop_local
from .config import Settings, atomic_json
from .errors import VideoLensError

LOGGER = logging.getLogger(__name__)


class LocalModels:
    def __init__(self, service):
        self.service = service
        self.root = Path(service.settings.models_dir).resolve() / "gguf"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self.thread: threading.Thread | None = None
        self.owner: str | None = None

    def close(self):
        """只回收本网页启动的模型，不停止别的启动脚本拥有的进程。"""
        if self.owner:
            stop_local(self.service.settings, owner=self.owner)

    def _entries(self) -> dict:
        return read_json(self.root / "library.json")

    def _path(self, model_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", model_id) or model_id not in self._entries():
            raise VideoLensError("MODEL_NOT_FOUND", "模型不存在", 404)
        path = (self.root / (model_id + ".gguf")).resolve()
        if path.parent != self.root or not path.is_file():
            raise VideoLensError("MODEL_NOT_FOUND", "受管模型文件不存在", 404)
        return path

    def listing(self) -> dict:
        with self.lock:
            settings = self.service.settings
            active = Path(settings.llama_model_path).resolve() if settings.llama_model_path else None
            rows = [{"id": key, **value, "selected": active == self.root / (key + ".gguf"), "deletable": True}
                    for key, value in self._entries().items() if (self.root / (key + ".gguf")).is_file()]
            if active and active.is_file() and not any(row["selected"] for row in rows):
                rows.insert(0, {"id": "installed", "name": settings.local_model + "（预装快照）", "bytes": active.stat().st_size,
                                "selected": True, "deletable": False})
            record = read_json(Path(settings.runtime_dir) / "llama-process.json")
            return {"models": rows, "running": bool(record and managed_process(record)), "jobs": list(self.jobs.values())[-10:]}

    def submit(self, kind: str, **values) -> dict:
        """一次只处理一个模型文件，避免多 GB 下载叠加；失败不会发布半成品。"""
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise VideoLensError("MODEL_BUSY", "已有模型操作正在进行，请等待完成", 409)
            identity = uuid4().hex
            job = {"id": identity, "status": "running", "message": "准备处理模型", "bytes": 0, "total": 0}
            self.jobs[identity] = job
            self.thread = threading.Thread(target=self._work, args=(job, kind, values), daemon=True, name="模型管理")
            self.thread.start()
            return dict(job)

    def _progress(self, job, **patch):
        with self.lock:
            job.update(patch)

    def _work(self, job: dict, kind: str, values: dict):
        from .logging_setup import task_context
        token = task_context.set(job["id"])
        temporary = self.root / (job["id"] + ".part")
        try:
            if kind == "runtime":
                from .bootstrap import install_runtime
                # 部署也会更新配置；与推理/分析互斥，以免替换正在运行的程序。
                with self._idle():
                    if self.listing()["running"]:
                        raise VideoLensError("MODEL_RUNNING", "请先停止本地模型，再部署运行时", 409)
                    install_runtime(self.service.settings, values["variant"], None, None, None)
                    saved = read_json(Path(self.service.settings.config_path))
                    self._apply({key: saved[key] for key in ("llama_server_path", "llama_gpu_layers") if key in saved})
                self._progress(job, status="succeeded", message="llama.cpp 已部署，可选择模型并启动")
                return
            name = values.get("name", "").strip()
            digest = hashlib.sha256()
            written = 0
            limit = 32 * 1024 ** 3

            def write_chunks(chunks, total):
                nonlocal written
                if total > limit or shutil.disk_usage(self.root).free < max(total, 1024 ** 3):
                    raise ValueError("模型过大或磁盘空间不足（单文件最多 32 GiB）")
                self._progress(job, total=total, message="正在复制/下载与校验")
                with temporary.open("wb") as output:
                    for chunk in chunks:
                        written += len(chunk)
                        if written > limit:
                            raise ValueError("模型超过 32 GiB 限制")
                        output.write(chunk)
                        digest.update(chunk)
                        self._progress(job, bytes=written)
                if total and written != total:
                    raise ValueError("文件接收不完整")

            if kind == "import":
                supplied = Path(values["path"])
                source = supplied.resolve()
                if not source.is_file() or supplied.suffix.lower() != ".gguf":
                    raise ValueError("请选择本机已有的单文件 .gguf 模型")
                name = name or supplied.name
                with source.open("rb") as stream:
                    write_chunks(iter(lambda: stream.read(8 * 1024 ** 2), b""), source.stat().st_size)
            elif kind == "download":
                url = values["url"].strip()
                expected = values.get("sha256", "").strip().lower()
                if urlparse(url).scheme != "https" or not re.fullmatch(r"[a-f0-9]{64}", expected):
                    raise ValueError("请使用 HTTPS 直链，并填写发布者提供的 SHA-256")
                name = name or Path(urlparse(url).path).name
                with httpx.Client(timeout=60, follow_redirects=True) as client:
                    with client.stream("GET", url) as response:
                        response.raise_for_status()
                        write_chunks(response.iter_bytes(8 * 1024 ** 2), int(response.headers.get("Content-Length", 0)))
                if digest.hexdigest() != expected:
                    raise ValueError("SHA-256 不匹配，文件已拒绝")
            else:
                raise ValueError("未知模型操作")
            with temporary.open("rb") as stream:
                header = stream.read(24)
            if len(header) < 24 or header[:4] != b"GGUF" or int.from_bytes(header[4:8], "little") not in {2, 3}:
                raise ValueError("不是支持的 GGUF 文件，不能将网页/压缩包作为模型导入")
            with self.lock:
                entries = self._entries()
                # 相同内容不重复占用磁盘。运行时兼容性由启动健康检查进一步确认。
                duplicate = next((key for key, item in entries.items() if item["sha256"] == digest.hexdigest()), None)
                model_id = duplicate or job["id"]
                if not duplicate:
                    temporary.replace(self.root / (model_id + ".gguf"))
                    entries[model_id] = {"name": Path(name).name[:160], "bytes": written, "sha256": digest.hexdigest()}
                    atomic_json(self.root / "library.json", entries)
                self._progress(job, status="succeeded", model_id=model_id, message="模型已就绪，请选择后启动")
            LOGGER.info("模型文件处理成功")
        except Exception as exc:
            # 网络异常可能含带签名下载地址，不把它保存或回显。
            message = str(exc) if isinstance(exc, (ValueError, VideoLensError)) else "模型操作失败，请检查网络、磁盘与日志"
            self._progress(job, status="failed", message=message)
            LOGGER.error("模型处理失败：%s", type(exc).__name__)
        finally:
            temporary.unlink(missing_ok=True)
            task_context.reset(token)

    def _idle(self):
        """先锁任务提交，再尝试推理锁；立即报告忙碌，不让网页无限等待。"""
        from contextlib import contextmanager

        @contextmanager
        def scope():
            from .service import ACTIVE
            with self.service._guard:
                if any(j["status"] in ACTIVE for j in self.service.store.list_jobs()):
                    raise VideoLensError("TASK_ACTIVE", "请等待视频分析完成后操作运行中的模型", 409)
                acquired = self.service._inference.acquire(blocking=False)
                if not acquired:
                    raise VideoLensError("MODEL_BUSY", "正在生成回答，请稍后重试", 409)
                try:
                    yield
                finally:
                    self.service._inference.release()
        return scope()

    def _apply(self, patch):
        # 与普通设置保存共用持久化，再将本地模型字段即时应用，无须重启网页。
        self.service.update_settings(patch)
        for key, value in patch.items():
            setattr(self.service.settings, key, value)
        pending = self.service._pending_settings
        if pending and pending.model_dump() == self.service.settings.model_dump():
            self.service._pending_settings = None

    def activate(self, model_id: str, gpu_layers: int, context: int) -> dict:
        with self._idle(), self.lock:
            path = Path(self.service.settings.llama_model_path) if model_id == "installed" else self._path(model_id)
            if not path.is_file():
                raise VideoLensError("MODEL_NOT_FOUND", "请先下载或导入 GGUF 模型", 404)
            alias = self.service.settings.local_model if model_id == "installed" else Path(self._entries()[model_id]["name"]).stem
            patch = {"llama_model_path": str(path.resolve()), "local_model": alias,
                     "llama_gpu_layers": gpu_layers, "local_context_tokens": context, "local_enabled": True}
            Settings(**{**self.service.settings.model_dump(), **patch})
            stop_local(self.service.settings)
            self.service.gateway.release()
            self._apply(patch)
            try:
                self.owner = "web-" + uuid4().hex
                return start_local(self.service.settings, owner=self.owner)
            except Exception as exc:
                raise VideoLensError("MODEL_START_FAILED", "模型启动失败；请检查 logs/llama-server.log，尝试 GPU 层数 0 或较小上下文", 503) from exc

    def stop(self) -> dict:
        with self._idle(), self.lock:
            return stop_local(self.service.settings)

    def delete(self, model_id: str) -> dict:
        with self._idle(), self.lock:
            if model_id == "installed":
                raise VideoLensError("SHARED_MODEL", "此模型属于原始共享快照。网页仅删除通过此页面下载或导入的独立模型副本", 409)
            path = self._path(model_id)
            record = read_json(Path(self.service.settings.runtime_dir) / "llama-process.json")
            if record and managed_process(record) and Path(record["model"]).resolve() == path:
                raise VideoLensError("MODEL_RUNNING", "请先停止此模型，再删除", 409)
            if self.service.settings.llama_model_path and Path(self.service.settings.llama_model_path).resolve() == path:
                self._apply({"llama_model_path": "", "local_enabled": False})
            path.unlink()
            entries = self._entries()
            del entries[model_id]
            atomic_json(self.root / "library.json", entries)
            return {"message": "已删除项目内模型副本，原始导入文件不受影响"}
