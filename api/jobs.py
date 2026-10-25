"""知影后台任务体系 —— HTTP 与 MCP 共用的唯一任务入口。

为什么必须有这一层
------------------
历史上 HTTP 路由层自造 ``t<uuid>`` 任务号（只活在内存），编排器又在 SQLite 里
另起一个任务号 —— 两套 ID 从未打通：HTTP 返回的 task_id 用 MCP 的 ``get_task``
查不到，重启后内存表清零、"用 task_id 查进度"直接 404。

现在三件事统一在这里：

1. **task_id 只发一次**：``JobStore.submit`` 生成 ID，经
   ``AnalyzeOptions.task_id`` 传进编排器，由 ``_start_task`` 用**同一个 ID**
   写 SQLite ``tasks`` 表 —— 内存进度流与持久化恢复记录自此一一对应。
2. **心跳落库**：编排器的进度回调在更新内存的同时，节流写
   ``repo.update_task``（``updated_ts`` 就是心跳）。此前大视频单阶段跑 40 分钟
   不落库，恢复器会把仍在跑的任务误判成"中断"。
3. **SSE 广播**：``JobState.queues`` 挂订阅者队列；慢消费者把队列塞满时
   **丢最旧帧**而不是丢整条消息 —— 终态帧必须能进队，否则客户端会挂着
   等心跳到天荒地老。

线程模型：``ThreadPoolExecutor(max_workers=2)`` 全进程一份；真正的重模型段
由编排器的 ``_inference_slot`` 串行化，这里只管"两路流水线交错"。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Final

from core.pipeline.orchestrator import AnalysisReport, AnalyzeOptions, Orchestrator, PipelineError
from core.storage.repository import NotFoundError, StorageError

logger = logging.getLogger("zhiying.api.jobs")

__all__ = [
    "JobState",
    "JobStore",
    "get_job_store",
    "reset_job_store",
    "shutdown_jobs",
    "TERMINAL_STATUS",
]

#: 任务终态（SSE 流与等待循环都以此收尾）。
TERMINAL_STATUS: Final[frozenset[str]] = frozenset({"done", "failed", "cancelled"})

#: 内存任务表上限。超限先淘汰**终态**任务；全是运行中时才动最旧的
#: （运行中的任务一旦被淘汰，SSE/轮询就查不到它了，所以它是最后手段）。
MAX_JOBS: Final[int] = 200

#: 每完成多少个任务做一次 SQLite 终态任务裁剪（簿记表不设硬闸，靠这里兜底）。
PRUNE_EVERY: Final[int] = 25
PRUNE_KEEP: Final[int] = 500

#: 心跳落库的最小间隔（秒）。阶段边界强制写，同阶段内的高频进度节流。
HEARTBEAT_INTERVAL_S: Final[float] = 0.5


class JobState:
    """一次后台分析的进度状态。线程安全（字段读写都在 ``_lock`` 内）。"""

    __slots__ = (
        "created_at", "elapsed_ms", "error", "loop", "message", "progress",
        "queues", "report", "stage", "status", "task_id", "updated_at",
        "video_id", "_last_db_write", "_lock",
    )

    def __init__(self, task_id: str, video_id: str = "") -> None:
        self.task_id = task_id
        self.video_id = video_id
        self.status: str = "queued"
        self.stage: str = ""
        self.progress: float = 0.0
        self.message: str = ""
        self.error: str = ""
        self.report: AnalysisReport | None = None
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.elapsed_ms = 0
        self._last_db_write = 0.0
        self._lock = threading.Lock()
        #: 订阅者队列（SSE）；只在事件循环线程里读写。
        self.queues: list[asyncio.Queue[dict[str, Any]]] = []
        self.loop: asyncio.AbstractEventLoop | None = None

    # -- 写入（任意线程）---------------------------------------------------
    def update(
        self,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: float | None = None,
        message: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            if status is not None:
                self.status = status
            if stage is not None:
                self.stage = stage
            if progress is not None:
                self.progress = max(0.0, min(1.0, float(progress)))
            if message is not None:
                self.message = message
            if error is not None:
                self.error = error
            self.updated_at = time.time()
            self.elapsed_ms = int((self.updated_at - self.created_at) * 1000)
            snapshot = self._snapshot_locked()
        self._publish(snapshot)

    def finish(self, report: AnalysisReport | None, error: str = "") -> None:
        with self._lock:
            self.status = "failed" if error else "done"
            self.progress = 1.0 if not error else self.progress
            self.error = error
            self.report = report
            if report is not None:
                self.video_id = report.video_id or self.video_id
                self.stage = report.status
                self.message = report.title or self.message
            self.updated_at = time.time()
            self.elapsed_ms = int((self.updated_at - self.created_at) * 1000)
            snapshot = self._snapshot_locked()
        self._publish(snapshot)

    # -- 读取 --------------------------------------------------------------
    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "video_id": self.video_id,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 4),
            "message": self.message,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "elapsed_ms": self.elapsed_ms,
            "report": self.report.to_dict() if self.report is not None else None,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    # -- SSE 广播 ----------------------------------------------------------
    def _publish(self, payload: dict[str, Any]) -> None:
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        for queue in list(self.queues):
            def _put(q: asyncio.Queue[dict[str, Any]] = queue) -> None:
                while True:
                    try:
                        q.put_nowait(payload)
                        return
                    except asyncio.QueueFull:
                        # 慢消费者：丢最旧的一帧腾位。终态帧必须能进队，
                        # 否则该客户端会一直挂在心跳上等不到收尾。
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:  # pragma: no cover - 竞态窗口极小
                            return
            try:
                loop.call_soon_threadsafe(_put)
            except RuntimeError:  # 事件循环已关闭
                continue


class JobStore:
    """任务注册表 + 线程池 + SQLite 心跳/终态落库。"""

    def __init__(
        self,
        *,
        max_workers: int = 2,
        max_jobs: int = MAX_JOBS,
    ) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers), thread_name_prefix="zhiying-analyze"
        )
        self._finished_count = 0
        self._max_jobs = max(10, int(max_jobs))
        #: 规范化 source → 运行中 task_id。同一视频重复点"分析"时直接回传
        #: 既有任务，而不是再排一条全量流水线（编排器层还有一道同视频锁兜底）。
        self._inflight: dict[str, str] = {}

    # -- 提交 --------------------------------------------------------------- #
    def submit(self, orch: Orchestrator, source: str, options: AnalyzeOptions) -> JobState:
        """登记任务并投递到线程池。抛 ``RuntimeError`` 表示线程池已关停。

        同一 source 已有排队/运行中的任务时，**返回那个任务**（task_id 不变）——
        调用方拿到的 202 与进度流都指向同一份执行，重复点击不会翻倍计算。
        """
        key = _source_key(source)
        with self._lock:
            existing_id = self._inflight.get(key)
            if existing_id:
                existing = self._jobs.get(existing_id)
                if existing is not None and existing.status in {"queued", "running"}:
                    return existing
            task_id = f"t{uuid.uuid4().hex[:12]}"
            job = JobState(task_id)
            self._evict_locked()
            self._jobs[task_id] = job
            self._inflight[key] = task_id

        # 同一个 task_id 传进编排器：SQLite tasks 表里的恢复记录与这里的
        # 进度流自此共用一个 ID，MCP get_task / HTTP /tasks/{id} 都能查到。
        options.task_id = task_id

        def progress(stage: str, ratio: float, message: str) -> None:
            job.update(
                status="running" if job.status == "queued" else None,
                stage=stage,
                progress=ratio,
                message=message,
            )
            self._heartbeat(orch, job, stage, ratio, message)

        options.progress = progress

        def work() -> None:
            job.update(status="running", progress=0.0, message="开始分析")
            try:
                report = orch.analyze(source, options)
            except PipelineError as exc:
                logger.warning("分析失败: %s", exc)
                job.finish(None, error=str(exc))
            except Exception as exc:  # noqa: BLE001 - 后台线程不能把异常抛给谁
                logger.exception("分析异常")
                job.finish(None, error=f"{type(exc).__name__}: {exc}")
            else:
                job.finish(report)
            self._persist_finish(orch, job, source_key=key)

        self._executor.submit(work)  # shutdown 后 submit 抛 RuntimeError，交给调用方转 503
        return job

    # -- 查询 --------------------------------------------------------------- #
    def get(self, task_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(task_id)

    def snapshot(self, task_id: str) -> dict[str, Any] | None:
        job = self.get(task_id)
        return job.snapshot() if job is not None else None

    def snapshots(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [job.snapshot() for job in jobs[: max(1, limit)]]

    def running_count(self) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if job.status in {"queued", "running"})

    def total(self) -> int:
        with self._lock:
            return len(self._jobs)

    def wait(self, job: JobState, *, timeout_s: float) -> dict[str, Any]:
        """等到终态或超时，返回最新快照（超时不取消任务，只返回当下状态）。"""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            snapshot = job.snapshot()
            if snapshot.get("status") in TERMINAL_STATUS:
                return snapshot
            time.sleep(0.2)
        return job.snapshot()

    # -- 生命周期 ----------------------------------------------------------- #
    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    # -- 内部 --------------------------------------------------------------- #
    def _evict_locked(self) -> None:
        if len(self._jobs) < self._max_jobs:
            return
        surplus = len(self._jobs) - self._max_jobs + 1
        terminal = sorted(
            (key for key, job in self._jobs.items() if job.status in TERMINAL_STATUS),
            key=lambda key: self._jobs[key].created_at,
        )
        for key in terminal[:surplus]:
            self._jobs.pop(key, None)
            surplus -= 1
        if surplus > 0:
            # 全在跑：动最旧的并记录 —— 被淘汰的运行中任务只能靠 SQLite 行查历史。
            oldest = sorted(self._jobs, key=lambda key: self._jobs[key].created_at)[:surplus]
            for key in oldest:
                logger.warning("任务表已满且无终态可淘汰，移除最旧任务 %s", key)
                self._jobs.pop(key, None)

    def _heartbeat(
        self, orch: Orchestrator, job: JobState, stage: str, ratio: float, message: str
    ) -> None:
        """进度节流落库：updated_ts 就是恢复器的心跳判据。"""
        now = time.monotonic()
        last = getattr(job, "_last_db_write", 0.0)
        if now - last < HEARTBEAT_INTERVAL_S:
            return
        job._last_db_write = now
        try:
            orch.repository.update_task(
                job.task_id, stage=stage, progress=ratio, message=message
            )
        except NotFoundError:
            pass  # 任务行由编排器 _start_task 在 PARSE 阶段创建，早于它的心跳忽略
        except StorageError:
            logger.debug("任务心跳落库失败 %s", job.task_id, exc_info=True)

    def _persist_finish(self, orch: Orchestrator, job: JobState, source_key: str | None = None) -> None:
        snapshot = job.snapshot()
        status = "done" if snapshot["status"] == "done" else "failed"
        if source_key:
            with self._lock:
                # 只在仍指向本任务时清掉（防极小概率的 ABA 覆盖）
                if self._inflight.get(source_key) == job.task_id:
                    self._inflight.pop(source_key, None)
        try:
            orch.repository.upsert_task(
                snapshot.get("video_id") or "",
                task_id=job.task_id,
                kind="ingest",
                status=status,  # type: ignore[arg-type]
                stage=snapshot.get("stage") or "",
                progress=float(snapshot.get("progress") or 0.0),
                message=snapshot.get("message") or "",
            )
            if snapshot.get("error"):
                orch.repository.update_task(job.task_id, error=str(snapshot["error"])[:2000])
        except Exception:  # noqa: BLE001 - 落库失败不该影响已完成的内存状态
            logger.debug("任务终态落库失败 %s", job.task_id, exc_info=True)
        with self._lock:
            self._finished_count += 1
            due = self._finished_count % PRUNE_EVERY == 0
        if due:
            try:
                orch.repository.prune_finished_tasks(keep=PRUNE_KEEP)
            except Exception:  # noqa: BLE001
                logger.debug("任务表裁剪失败", exc_info=True)


def _source_key(source: str) -> str:
    """同源判定键：去空白 + 小写。URL 参数顺序不同视为不同源（宁可多跑不误判）。"""
    return " ".join(str(source or "").split()).lower()


# --------------------------------------------------------------------------- #
# 进程级单例：HTTP 路由与 MCP（含 HTTP 模式的 /api/mcp）共用同一份注册表，
# 这样 MCP 提交的任务用 HTTP /tasks/{id} 也能查，反之亦然。
# --------------------------------------------------------------------------- #
_STORE: JobStore | None = None
_STORE_LOCK = threading.Lock()


def get_job_store() -> JobStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = JobStore()
        return _STORE


def reset_job_store(*, shutdown: bool = True) -> None:
    """丢弃单例（测试清理用）。默认先关停线程池。"""
    global _STORE
    with _STORE_LOCK:
        store, _STORE = _STORE, None
    if store is not None and shutdown:
        store.shutdown(wait=False)


def shutdown_jobs(*, wait: bool = True) -> None:
    """关停线程池（进程退出时调用；保持与旧 routes API 兼容的名字）。"""
    with _STORE_LOCK:
        store = _STORE
    if store is not None:
        store.shutdown(wait=wait)
