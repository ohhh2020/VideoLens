"""知影 ZhiYing —— SQLite 元数据库。
职责边界
--------
本模块**只**管结构化元数据（视频 / 切片 / 关键帧 / 转写片段 / 任务队列），
与之并列的两块存储各司其职：

* 向量（文本切片向量、关键帧图像向量）→ ``core/rag/vector_store.py``（ChromaDB）
* 二进制文件（视频、音频、关键帧 JPG、导出报告）→ 磁盘，由 ``paths`` 配置决定

这样做的好处是：向量库可以整库重建（re-embed），而 ``video_id`` / ``clip_id``
这些**溯源锚点**始终稳定存在 SQLite 里。任何对外输出的证据链都从这里出发。
设计要点
--------
1. **WAL + 短事务**：读写并发下不互相阻塞；每次写操作一个 ``BEGIN IMMEDIATE``
   事务，失败整体回滚（对应需求文档「向量/元数据写入失败 → 事务回滚」）。
2. **线程局部连接**：``sqlite3`` 连接不是线程安全的。这里给每个线程一条连接，
   并在 ``close()`` 时统一释放，供 FastAPI 线程池 / 抽帧线程池使用。
3. **schema 版本迁移**：用 ``PRAGMA user_version`` 记录，逐版升级，不猜。
4. **容量红线**：写入前按 ``storage.limits`` 校验总量，超限抛
   :class:`CapacityExceededError` 而不是让库撑爆（需求文档表 3）。
5. **毫秒级时间戳**：所有时间字段统一 ``*_ms`` 整数，不做浮点秒运算。
典型用法
--------
>>> from backend.config import get_app_config
>>> from backend.storage.repository import SqliteRepository
>>> repo = SqliteRepository.from_config(get_app_config())
>>> vid = repo.add_video(source_url="https://example.com/a.mp4", duration_ms=612_345)
>>> repo.add_clips(vid, [{"clip_id": f"{vid}-c0000", "start_ms": 0, "end_ms": 20_100,
...                       "text": "大家好"}])
>>> repo.get_clip(f"{vid}-c0000").time_range
'00:00:00.000-00:00:20.100'
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

from backend.config import AppConfig, get_app_config
from backend.storage.conversations import DDL_V6, DDL_V7
from backend.analysis.runs import DDL_V8

logger = logging.getLogger("zhiying.storage.repository")

__all__ = [
    "SCHEMA_VERSION",
    "CapacityExceededError",
    "ClipRecord",
    "KeyframeRecord",
    "NotFoundError",
    "RepositoryError",
    "SegmentRecord",
    "SqliteRepository",
    "StorageError",
    "StorageStats",
    "TaskRecord",
    "VideoRecord",
    "VideoStatus",
    "new_video_id",
]


# =============================================================================
# 异常体系
# =============================================================================


class StorageError(RuntimeError):
    """存储层错误基类。"""


class RepositoryError(StorageError):
    """仓库操作失败（SQL 执行、事务冲突、数据不合法）。"""


class NotFoundError(RepositoryError, KeyError):
    """按主键找不到记录。
    同时继承 ``KeyError``，这样 ``repo.get_video("x")`` 能被 ``except KeyError``
    捕获 —— 调用方往往就是按"字典缺键"的语义来处理的。
    """

    def __init__(self, entity: str, key: str) -> None:
        super().__init__(f"{entity} 不存在: {key}")
        self.entity = entity
        self.key = key

    def __str__(self) -> str:
        # KeyError.__str__ 会给消息加引号，这里绕开
        return f"{self.entity} 不存在: {self.key}"


class CapacityExceededError(StorageError):
    """触及容量红线（需求文档表 3）。"""

    def __init__(self, table: str, limit: int, current: int, incoming: int) -> None:
        super().__init__(
            f"{table} 容量超限：上限 {limit}，当前 {current}，本次写入 {incoming}"
        )
        self.table = table
        self.limit = limit
        self.current = current
        self.incoming = incoming


# =============================================================================
# 枚举与字面量
# =============================================================================

VideoStatus = Literal[
    "pending",  # 已登记，未开始
    "downloading",  # 远程拉取中
    "probing",  # ffprobe 探测中
    "transcribing",  # 转写中
    "detecting",  # 场景检测 + 抽帧中
    "indexing",  # 向量化写入中
    "ready",  # 全流程完成，可使用
    "failed",  # 失败（error 字段有原因）
    "cancelled",  # 用户取消
]

TaskStatus = Literal["pending", "running", "done", "failed", "cancelled"]

SCHEMA_VERSION: Final[int] = 8

#: 表名 → ``storage.limits`` 里对应的上限字段
_LIMIT_MAP: Final[dict[str, str]] = {
    "videos": "max_videos",
    "clips": "max_clips",
    "transcript_segments": "max_transcript_segments",
    "keyframes": "max_keyframes",
    # tasks 不设硬闸：分析不能被簿记行挡住，改由 prune_finished_tasks 裁剪
    "artifacts": "max_artifacts",
}


def new_video_id() -> str:
    """生成稳定的 ``video_id``。
    只含小写字母数字，长度固定 13 —— 它会进 ``clip_id``
    （``{video_id}-c0000``）和关键帧文件名，不能出现路径分隔符或空格。
    """
    return "v" + uuid.uuid4().hex[:12]


def _now_iso() -> str:
    """UTC 毫秒精度 ISO 时间串，便于人读与排序。"""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _now_epoch() -> float:
    return time.time()


# =============================================================================
# 记录对象
# =============================================================================


@dataclass(frozen=True, slots=True)
class VideoRecord:
    """一条视频元数据。"""

    video_id: str
    display_name: str = ""
    source_kind: Literal["local", "url"] = "url"
    source_url: str = ""
    local_path: str = ""
    title: str = ""
    duration_ms: int = 0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = False
    size_bytes: int = 0
    language: str = ""
    status: VideoStatus = "pending"
    stage: str = ""
    progress: float = 0.0
    error: str = ""
    fingerprint: str = ""
    vector_generation: str = ""
    created_at: str = ""
    updated_at: str = ""
    finished_at: str = ""

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "display_name": self.display_name,
            "source_kind": self.source_kind,
            "source_url": self.source_url,
            "local_path": self.local_path,
            "title": self.title,
            "duration_ms": self.duration_ms,
            "duration_s": round(self.duration_s, 3),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "has_audio": self.has_audio,
            "size_bytes": self.size_bytes,
            "language": self.language,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "error": self.error,
            "fingerprint": self.fingerprint,
            "vector_generation": self.vector_generation,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
        }


@dataclass(frozen=True, slots=True)
class ClipRecord:
    """一段带音切片（检索与引用的最小单位）。
    ``start_ms`` / ``end_ms`` 是**含 1.5s 重叠边界**的对外区间；
    ``core_start_ms`` / ``core_end_ms`` 是去掉重叠的"正文区间"。
    回答里给用户看 ``start_ms``～``end_ms``，做 RRF 去重时按 ``core_*`` 判同一片。
    """

    clip_id: str
    video_id: str
    index_no: int = 0
    start_ms: int = 0
    end_ms: int = 0
    core_start_ms: int = 0
    core_end_ms: int = 0
    text: str = ""
    core_text: str = ""
    keyframe_ids: tuple[str, ...] = ()
    truncated: bool = False
    created_at: str = ""

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def time_range(self) -> str:
        return f"{ms_to_timecode(self.start_ms)}-{ms_to_timecode(self.end_ms)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "video_id": self.video_id,
            "index_no": self.index_no,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "core_start_ms": self.core_start_ms,
            "core_end_ms": self.core_end_ms,
            "duration_ms": self.duration_ms,
            "start_time": ms_to_timecode(self.start_ms),
            "end_time": ms_to_timecode(self.end_ms),
            "text": self.text,
            "core_text": self.core_text,
            "keyframe_ids": list(self.keyframe_ids),
            "truncated": self.truncated,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class KeyframeRecord:
    """一张关键帧。``path`` 是绝对或相对项目根的磁盘路径。"""

    keyframe_id: str
    video_id: str
    clip_id: str | None = None
    scene_index: int = 0
    timestamp_ms: int = 0
    path: str = ""
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    sharpness: float = 0.0
    sharpness_ok: bool = True
    quality_ok: bool = True
    description: str = ""
    ocr_text: str = ""
    created_at: str = ""

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_ms / 1000.0

    @property
    def timecode(self) -> str:
        return ms_to_timecode(self.timestamp_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyframe_id": self.keyframe_id,
            "video_id": self.video_id,
            "clip_id": self.clip_id,
            "scene_index": self.scene_index,
            "timestamp_ms": self.timestamp_ms,
            "timecode": self.timecode,
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "size_bytes": self.size_bytes,
            "sharpness": round(self.sharpness, 2),
            "sharpness_ok": self.sharpness_ok,
            "quality_ok": self.quality_ok,
            "description": self.description,
            "ocr_text": self.ocr_text,
        }


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    """一条转写片段（ASR 原子输出，毫秒时间戳）。"""

    segment_id: str
    video_id: str
    index_no: int = 0
    start_ms: int = 0
    end_ms: int = 0
    text: str = ""
    language: str = ""
    confidence: float = 0.0
    source: Literal["asr", "subtitle"] = "asr"

    @property
    def time_range(self) -> str:
        return f"{ms_to_timecode(self.start_ms)}-{ms_to_timecode(self.end_ms)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "video_id": self.video_id,
            "index_no": self.index_no,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "start_time": ms_to_timecode(self.start_ms),
            "end_time": ms_to_timecode(self.end_ms),
            "text": self.text,
            "language": self.language,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """持久化任务，用于崩溃后恢复（需求文档表 4 最后一行）。"""

    task_id: str
    video_id: str
    kind: str = "ingest"
    status: TaskStatus = "pending"
    stage: str = ""
    progress: float = 0.0
    message: str = ""
    error: str = ""
    attempts: int = 0
    payload: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    updated_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "video_id": self.video_id,
            "kind": self.kind,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "attempts": self.attempts,
            "payload": dict(self.payload),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class StorageStats:
    """容量水位，用于 UI 展示与写入前校验。"""

    videos: int = 0
    clips: int = 0
    transcript_segments: int = 0
    keyframes: int = 0
    tasks_pending: int = 0
    tasks_running: int = 0
    keyframe_bytes: int = 0
    db_bytes: int = 0
    limits: Mapping[str, int] = field(default_factory=dict)

    def usage(self, table: str) -> float:
        """返回该表的占用率 0.0~1.0（不设上限时返回 0.0）。"""
        key = _LIMIT_MAP.get(table)
        if not key:
            return 0.0
        limit = int(self.limits.get(key, 0) or 0)
        if limit <= 0:
            return 0.0
        current = int(getattr(self, table, 0) or 0)
        return min(1.0, current / limit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "videos": self.videos,
            "clips": self.clips,
            "transcript_segments": self.transcript_segments,
            "keyframes": self.keyframes,
            "tasks_pending": self.tasks_pending,
            "tasks_running": self.tasks_running,
            "keyframe_bytes": self.keyframe_bytes,
            "db_bytes": self.db_bytes,
            "limits": dict(self.limits),
            "usage": {
                "videos": round(self.usage("videos"), 4),
                "clips": round(self.usage("clips"), 4),
                "transcript_segments": round(self.usage("transcript_segments"), 4),
                "keyframes": round(self.usage("keyframes"), 4),
            },
        }


def ms_to_timecode(ms: int) -> str:
    """毫秒 → ``HH:MM:SS.mmm``（需求文档统一输出格式）。"""
    total = max(0, int(ms))
    hours, rem = divmod(total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


# =============================================================================
# DDL
# =============================================================================

_DDL_V1: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS videos (
        video_id     TEXT PRIMARY KEY,
        display_name TEXT NOT NULL DEFAULT '',
        source_kind  TEXT NOT NULL DEFAULT 'url',
        source_url   TEXT NOT NULL DEFAULT '',
        local_path   TEXT NOT NULL DEFAULT '',
        title        TEXT NOT NULL DEFAULT '',
        duration_ms  INTEGER NOT NULL DEFAULT 0,
        width        INTEGER NOT NULL DEFAULT 0,
        height       INTEGER NOT NULL DEFAULT 0,
        fps          REAL NOT NULL DEFAULT 0,
        has_audio    INTEGER NOT NULL DEFAULT 0,
        size_bytes   INTEGER NOT NULL DEFAULT 0,
        language     TEXT NOT NULL DEFAULT '',
        status       TEXT NOT NULL DEFAULT 'pending',
        stage        TEXT NOT NULL DEFAULT '',
        progress     REAL NOT NULL DEFAULT 0,
        error        TEXT NOT NULL DEFAULT '',
        created_at   TEXT NOT NULL DEFAULT '',
        updated_at   TEXT NOT NULL DEFAULT '',
        finished_at  TEXT NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status, updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_videos_created ON videos(created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS clips (
        clip_id        TEXT PRIMARY KEY,
        video_id       TEXT NOT NULL,
        index_no       INTEGER NOT NULL DEFAULT 0,
        start_ms       INTEGER NOT NULL DEFAULT 0,
        end_ms         INTEGER NOT NULL DEFAULT 0,
        core_start_ms  INTEGER NOT NULL DEFAULT 0,
        core_end_ms    INTEGER NOT NULL DEFAULT 0,
        text           TEXT NOT NULL DEFAULT '',
        core_text      TEXT NOT NULL DEFAULT '',
        keyframe_ids   TEXT NOT NULL DEFAULT '[]',
        truncated      INTEGER NOT NULL DEFAULT 0,
        created_at     TEXT NOT NULL DEFAULT '',
        FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_clips_video ON clips(video_id, index_no)",
    "CREATE INDEX IF NOT EXISTS idx_clips_time ON clips(video_id, start_ms)",
    """
    CREATE TABLE IF NOT EXISTS transcript_segments (
        segment_id  TEXT PRIMARY KEY,
        video_id    TEXT NOT NULL,
        index_no    INTEGER NOT NULL DEFAULT 0,
        start_ms    INTEGER NOT NULL DEFAULT 0,
        end_ms      INTEGER NOT NULL DEFAULT 0,
        text        TEXT NOT NULL DEFAULT '',
        language    TEXT NOT NULL DEFAULT '',
        confidence  REAL NOT NULL DEFAULT 0,
        source      TEXT NOT NULL DEFAULT 'asr',
        FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_segments_video ON transcript_segments(video_id, index_no)",
    """
    CREATE TABLE IF NOT EXISTS keyframes (
        keyframe_id  TEXT PRIMARY KEY,
        video_id     TEXT NOT NULL,
        clip_id      TEXT,
        scene_index  INTEGER NOT NULL DEFAULT 0,
        timestamp_ms INTEGER NOT NULL DEFAULT 0,
        path         TEXT NOT NULL DEFAULT '',
        width        INTEGER NOT NULL DEFAULT 0,
        height       INTEGER NOT NULL DEFAULT 0,
        size_bytes   INTEGER NOT NULL DEFAULT 0,
        sharpness    REAL NOT NULL DEFAULT 0,
        sharpness_ok INTEGER NOT NULL DEFAULT 1,
        quality_ok   INTEGER NOT NULL DEFAULT 1,
        description  TEXT NOT NULL DEFAULT '',
        ocr_text     TEXT NOT NULL DEFAULT '',
        created_at   TEXT NOT NULL DEFAULT '',
        FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_kf_video ON keyframes(video_id, timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_kf_clip ON keyframes(clip_id)",
    """
    CREATE TABLE IF NOT EXISTS tasks (
        task_id    TEXT PRIMARY KEY,
        video_id   TEXT NOT NULL,
        kind       TEXT NOT NULL DEFAULT 'ingest',
        status     TEXT NOT NULL DEFAULT 'pending',
        stage      TEXT NOT NULL DEFAULT '',
        progress   REAL NOT NULL DEFAULT 0,
        message    TEXT NOT NULL DEFAULT '',
        error      TEXT NOT NULL DEFAULT '',
        attempts   INTEGER NOT NULL DEFAULT 0,
        payload    TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT '',
        updated_at TEXT NOT NULL DEFAULT '',
        updated_ts REAL NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, updated_ts)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_video ON tasks(video_id)",
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY,
        video_id    TEXT NOT NULL,
        kind        TEXT NOT NULL DEFAULT 'report',
        path        TEXT NOT NULL DEFAULT '',
        size_bytes  INTEGER NOT NULL DEFAULT 0,
        created_at  TEXT NOT NULL DEFAULT '',
        FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_artifacts_video ON artifacts(video_id, kind)",
)

# ===== v2：分析指纹 =====
#: 换 ASR 模型、换向量模型、换关键帧分辨率、换切分时长后重跑，旧向量和
#: 摘要**不能直接复用** —— 它们不是一套参数算出来的。混在一起不会报错，
#: 只是让检索结果里出现"同一条证据两个来源、分数不可比"，很难查。
#: 记下指纹，重跑时对不上就把该阶段之后的所有产物作废。
_DDL_V2: Final[tuple[str, ...]] = (
    "ALTER TABLE videos ADD COLUMN fingerprint TEXT NOT NULL DEFAULT ''",
)

# ===== v3：向量索引代 =====
#: 重建索引时先把向量写进 ``clip_text_<代号>`` 这一代，全部成功才把指针切
#: 过来。中途模型或网络故障只丢半成品，上一代索引仍然完整可用 —— 固定集合
#: 名直连 upsert 的做法一旦重建失败，旧索引跟着毁，检索直接空手。
_DDL_V3: Final[tuple[str, ...]] = (
    "ALTER TABLE videos ADD COLUMN vector_generation TEXT NOT NULL DEFAULT ''",
)

# ===== v4：多轮问答历史（持久化，重启不丢）=====
_DDL_V4: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS qa_history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      video_id TEXT NOT NULL,
      question TEXT NOT NULL,
      answer TEXT NOT NULL,
      refs TEXT NOT NULL DEFAULT '[]',
      rewritten TEXT NOT NULL DEFAULT '',
      elapsed_ms INTEGER NOT NULL DEFAULT 0,
      session_id TEXT NOT NULL DEFAULT '',
      created_at TEXT NOT NULL,
      created_ts REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_qa_video ON qa_history(video_id, created_ts)",
    "CREATE INDEX IF NOT EXISTS idx_qa_session ON qa_history(session_id)",
)

# ===== v5：分析报告持久化（重启后 /ui 仍能查看上次分析结果）=====
_DDL_V5: Final[tuple[str, ...]] = (
    "ALTER TABLE videos ADD COLUMN report_json TEXT NOT NULL DEFAULT ''",
)

_MIGRATIONS: Final[dict[int, tuple[str, ...]]] = {
    1: _DDL_V1,
    2: _DDL_V2,
    3: _DDL_V3,
    4: _DDL_V4,
    5: _DDL_V5,
    6: DDL_V6,
    7: DDL_V7,
    8: DDL_V8,
}

# =============================================================================
# 仓库
# =============================================================================


class SqliteRepository:
    """SQLite 元数据库（线程安全，按声明周期复用单实例）。
    Parameters
    ----------
    db_path:
        数据库文件路径。父目录会自动创建。
    journal_mode:
        ``WAL``（默认）/ ``DELETE``。WAL 是读写并发的关键。
    busy_timeout_ms:
        写锁等待上限。抽帧线程池 + HTTP 请求并发写入时避免 ``database is locked``。
    limits:
        ``{"max_videos": 200, ...}``，来自 ``storage.limits``。空表示不校验。
    check_capacity:
        是否在批量写入前校验容量。
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        journal_mode: str = "WAL",
        busy_timeout_ms: int = 5_000,
        limits: Mapping[str, int] | None = None,
        check_capacity: bool = True,
    ) -> None:
        self._path = Path(db_path).expanduser()
        if not self._path.name:
            raise RepositoryError(f"非法数据库路径: {db_path!r}")
        self._journal_mode = str(journal_mode or "WAL").upper()
        if self._journal_mode not in {
            "WAL",
            "DELETE",
            "TRUNCATE",
            "PERSIST",
            "MEMORY",
            "OFF",
        }:
            raise RepositoryError(f"不支持的 journal_mode: {journal_mode!r}")
        self._busy_timeout_ms = max(0, int(busy_timeout_ms))
        self._limits: dict[str, int] = {
            str(k): int(v) for k, v in (limits or {}).items()
        }
        self._check_capacity = bool(check_capacity)

        self._local = threading.local()
        self._all_conns: list[sqlite3.Connection] = []
        self._conn_lock = threading.Lock()

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    # -- 构造 ---------------------------------------------------------------

    @classmethod
    def from_config(cls, config: AppConfig | None = None) -> SqliteRepository:
        """按 ``configs/app_config.yaml`` 的 ``paths.sqlite`` / ``storage`` 构造。"""
        cfg = config or get_app_config()
        db_path = cfg.path("sqlite")
        limits = {
            "max_videos": cfg.storage.limits.max_videos,
            "max_clips": cfg.storage.limits.max_clips,
            "max_transcript_segments": cfg.storage.limits.max_transcript_segments,
            "max_keyframes": cfg.storage.limits.max_keyframes,
        }
        return cls(
            db_path,
            journal_mode=cfg.storage.sqlite.journal_mode,
            busy_timeout_ms=cfg.storage.sqlite.busy_timeout_ms,
            limits=limits,
        )

    @property
    def path(self) -> Path:
        return self._path

    # -- 连接管理 ----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._path),
            timeout=max(1.0, self._busy_timeout_ms / 1000.0),
            isolation_level=None,  # 手动控制事务
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA journal_mode={self._journal_mode}")
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def connection(self) -> sqlite3.Connection:
        """取当前线程的连接（惰性创建）。"""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
            with self._conn_lock:
                self._all_conns.append(conn)
        return conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """写事务：``BEGIN IMMEDIATE`` → 提交/回滚。"""
        conn = self.connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:  # pragma: no cover - 极端并发
            raise RepositoryError(f"开启事务失败: {exc}") from exc
        try:
            yield conn
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:  # pragma: no cover
                logger.warning("回滚失败，连接可能已失效", exc_info=True)
            raise
        else:
            conn.execute("COMMIT")

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        try:
            cur = self.connection().execute(sql, tuple(params))
        except sqlite3.Error as exc:
            raise RepositoryError(f"查询失败: {exc}\nSQL: {sql}") from exc
        return cur.fetchall()

    def _query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        rows = self._query(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        """关闭本实例创建的所有连接（跨线程）。"""
        with self._conn_lock:
            conns, self._all_conns = self._all_conns, []
        for conn in conns:
            try:
                conn.close()
            except sqlite3.Error:  # pragma: no cover
                logger.debug("关闭连接时忽略错误", exc_info=True)
        self._local = threading.local()

    def __enter__(self) -> SqliteRepository:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"SqliteRepository(path={str(self._path)!r})"

    # -- schema ------------------------------------------------------------

    def _initialize(self) -> None:
        conn = self.connection()
        current = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
        if current > SCHEMA_VERSION:
            raise RepositoryError(
                f"数据库 schema 版本 {current} 高于本程序支持的 {SCHEMA_VERSION}，"
                f"请升级知影或换个 db 文件（当前: {self._path}）"
            )
        if current == SCHEMA_VERSION:
            return
        if 0 < current < 8:
            backup = self._path.with_name(self._path.name + ".before-v8.bak")
            if not backup.exists():
                destination = sqlite3.connect(backup)
                try:
                    conn.backup(destination)
                finally:
                    destination.close()
        logger.info(
            "初始化数据库 schema: %s → %s (%s)", current, SCHEMA_VERSION, self._path
        )
        try:
            conn.execute("BEGIN IMMEDIATE")
            for version in range(current + 1, SCHEMA_VERSION + 1):
                for stmt in _MIGRATIONS[version]:
                    conn.execute(stmt)
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            conn.execute("COMMIT")
        except sqlite3.Error as exc:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:  # pragma: no cover
                pass
            raise RepositoryError(f"数据库初始化失败: {exc}") from exc

    def schema_version(self) -> int:
        row = self._query_one("PRAGMA user_version")
        return int(row[0]) if row else 0

    def table_names(self) -> tuple[str, ...]:
        rows = self._query(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return tuple(str(r["name"]) for r in rows)

    # -- 容量 --------------------------------------------------------------

    def _count(self, conn: sqlite3.Connection, table: str) -> int:
        row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()  # noqa: S608 - 表名来自白名单
        return int(row["n"] if row else 0)

    def _assert_capacity(
        self, conn: sqlite3.Connection, table: str, incoming: int
    ) -> None:
        if not self._check_capacity or incoming <= 0:
            return
        limit_key = _LIMIT_MAP.get(table)
        if not limit_key:
            return
        limit = self._limits.get(limit_key, 0)
        if limit <= 0:
            return
        current = self._count(conn, table)
        if current + incoming > limit:
            raise CapacityExceededError(table, limit, current, incoming)

    def stats(self) -> StorageStats:
        counts: dict[str, int] = {}
        for table in ("videos", "clips", "transcript_segments", "keyframes"):
            row = self._query_one(f"SELECT COUNT(*) AS n FROM {table}")  # noqa: S608 - 白名单
            counts[table] = int(row["n"]) if row else 0
        pending = self._query_one(
            "SELECT COUNT(*) AS n FROM tasks WHERE status='pending'"
        )
        running = self._query_one(
            "SELECT COUNT(*) AS n FROM tasks WHERE status='running'"
        )
        kf_bytes = self._query_one(
            "SELECT COALESCE(SUM(size_bytes),0) AS n FROM keyframes"
        )
        db_bytes = self._path.stat().st_size if self._path.exists() else 0
        return StorageStats(
            videos=counts["videos"],
            clips=counts["clips"],
            transcript_segments=counts["transcript_segments"],
            keyframes=counts["keyframes"],
            tasks_pending=int(pending["n"]) if pending else 0,
            tasks_running=int(running["n"]) if running else 0,
            keyframe_bytes=int(kf_bytes["n"]) if kf_bytes else 0,
            db_bytes=db_bytes,
            limits=dict(self._limits),
        )

    # -- videos ------------------------------------------------------------

    def add_video(
        self,
        *,
        video_id: str | None = None,
        display_name: str = "",
        source_kind: Literal["local", "url"] = "url",
        source_url: str = "",
        local_path: str = "",
        title: str = "",
        duration_ms: int = 0,
        width: int = 0,
        height: int = 0,
        fps: float = 0.0,
        has_audio: bool = False,
        size_bytes: int = 0,
        language: str = "",
        status: VideoStatus = "pending",
        replace: bool = False,
    ) -> str:
        """登记一条视频，返回 ``video_id``。
        ``replace=True`` 时对同 id 做 **"补齐式"覆盖**：派生字段（时长/宽高/fps/
        有音轨/文件大小/标题/语言/本地路径）只在本次传入非空时才更新，否则保留旧值；
        进度与错误状态始终重置，``created_at`` 保留首次登记时间。
        这样"同一视频换了源地址重新入库"不会把已探测到的元数据抹成 0，
        而"重新跑一遍流程"仍能从干净的进度状态开始。
        """
        vid = str(video_id or new_video_id()).strip()
        if not vid:
            raise RepositoryError("video_id 不能为空")
        now = _now_iso()
        with self._tx() as conn:
            exists = conn.execute(
                "SELECT 1 FROM videos WHERE video_id=?", (vid,)
            ).fetchone()
            if exists and not replace:
                raise RepositoryError(f"video_id 已存在: {vid}")
            if not exists:
                self._assert_capacity(conn, "videos", 1)
            conn.execute(
                """
                INSERT INTO videos (
                    video_id, display_name, source_kind, source_url, local_path, title,
                    duration_ms, width, height, fps, has_audio, size_bytes, language,
                    status, stage, progress, error, created_at, updated_at, finished_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'',0,'',?,?,'')
                ON CONFLICT(video_id) DO UPDATE SET
                    display_name = CASE WHEN excluded.display_name <> ''
                                        THEN excluded.display_name ELSE videos.display_name END,
                    source_kind  = excluded.source_kind,
                    source_url   = CASE WHEN excluded.source_url <> ''
                                        THEN excluded.source_url ELSE videos.source_url END,
                    local_path   = CASE WHEN excluded.local_path <> ''
                                        THEN excluded.local_path ELSE videos.local_path END,
                    title        = CASE WHEN excluded.title <> ''
                                        THEN excluded.title ELSE videos.title END,
                    duration_ms  = CASE WHEN excluded.duration_ms > 0
                                        THEN excluded.duration_ms ELSE videos.duration_ms END,
                    width        = CASE WHEN excluded.width > 0
                                        THEN excluded.width ELSE videos.width END,
                    height       = CASE WHEN excluded.height > 0
                                        THEN excluded.height ELSE videos.height END,
                    fps          = CASE WHEN excluded.fps > 0
                                        THEN excluded.fps ELSE videos.fps END,
                    has_audio    = MAX(videos.has_audio, excluded.has_audio),
                    size_bytes   = CASE WHEN excluded.size_bytes > 0
                                        THEN excluded.size_bytes ELSE videos.size_bytes END,
                    language     = CASE WHEN excluded.language <> ''
                                        THEN excluded.language ELSE videos.language END,
                    status       = excluded.status,
                    stage        = '',
                    progress     = 0,
                    error        = '',
                    created_at   = videos.created_at,
                    updated_at   = excluded.updated_at,
                    finished_at  = ''
                """,
                (
                    vid,
                    display_name,
                    source_kind,
                    source_url,
                    local_path,
                    title,
                    int(duration_ms),
                    int(width),
                    int(height),
                    float(fps),
                    int(bool(has_audio)),
                    int(size_bytes),
                    language,
                    status,
                    now,
                    now,
                ),
            )
        logger.debug("登记视频 %s (%s)", vid, display_name or source_url)
        return vid

    def get_video(self, video_id: str) -> VideoRecord:
        row = self._query_one("SELECT * FROM videos WHERE video_id=?", (video_id,))
        if row is None:
            raise NotFoundError("video", video_id)
        return _row_to_video(row)

    def find_video(self, video_id: str) -> VideoRecord | None:
        row = self._query_one("SELECT * FROM videos WHERE video_id=?", (video_id,))
        return _row_to_video(row) if row else None

    def list_videos(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> list[VideoRecord]:
        sql = "SELECT * FROM videos"
        params: list[Any] = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([max(0, int(limit)), max(0, int(offset))])
        return [_row_to_video(r) for r in self._query(sql, params)]

    def expired_videos(
        self, *, finished_before: str, limit: int = 500
    ) -> list[VideoRecord]:
        """分析完成时间早于 ``finished_before`` 的视频，最久远的排前面。

        只挑 ``status='ready'``：进行中的任务归崩溃恢复管，TTL 清理不该
        动正在写的数据。``finished_before`` 用 ``_now_iso()`` 同款格式
        （UTC + 毫秒），字符串序即时间序。
        """
        if not finished_before:
            return []
        sql = (
            "SELECT * FROM videos WHERE status='ready' AND finished_at<>'' "
            "AND finished_at < ? ORDER BY finished_at ASC LIMIT ?"
        )
        rows = self._query(sql, [finished_before, max(1, int(limit))])
        return [_row_to_video(r) for r in rows]

    def count_videos(self, *, status: str | None = None) -> int:
        if status:
            row = self._query_one(
                "SELECT COUNT(*) AS n FROM videos WHERE status=?", (status,)
            )
        else:
            row = self._query_one("SELECT COUNT(*) AS n FROM videos")
        return int(row["n"]) if row else 0

    def update_video_status(
        self,
        video_id: str,
        status: VideoStatus,
        *,
        stage: str | None = None,
        progress: float | None = None,
        error: str | None = None,
        finished: bool | None = None,
    ) -> None:
        """推进状态机。``finished=None`` 时按 ``status`` 自动推断终态。"""
        if finished is None:
            finished = status in ("ready", "failed", "cancelled")
        sets = ["status=?", "updated_at=?"]
        params: list[Any] = [status, _now_iso()]
        if stage is not None:
            sets.append("stage=?")
            params.append(stage)
        if progress is not None:
            sets.append("progress=?")
            params.append(max(0.0, min(1.0, float(progress))))
        if error is not None:
            sets.append("error=?")
            params.append(error)
        if finished:
            sets.append("finished_at=?")
            params.append(_now_iso())
        params.append(video_id)
        with self._tx() as conn:
            cur = conn.execute(
                f"UPDATE videos SET {', '.join(sets)} WHERE video_id=?", params
            )  # noqa: S608
            if cur.rowcount == 0:
                raise NotFoundError("video", video_id)

    def update_video_meta(
        self,
        video_id: str,
        *,
        duration_ms: int | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        has_audio: bool | None = None,
        size_bytes: int | None = None,
        title: str | None = None,
        language: str | None = None,
        local_path: str | None = None,
        fingerprint: str | None = None,
        vector_generation: str | None = None,
    ) -> None:
        pairs: list[tuple[str, Any]] = []
        for column, value in (
            ("duration_ms", duration_ms),
            ("width", width),
            ("height", height),
            ("fps", fps),
            ("has_audio", None if has_audio is None else int(bool(has_audio))),
            ("size_bytes", size_bytes),
            ("title", title),
            ("language", language),
            ("local_path", local_path),
            # 分析指纹：模型/切分/关键帧规格任一变化都要让它失效，
            # 否则把旧向量和旧摘要混进新模型的结果里，检索会出现同一条证据
            # 两个来源、分数不可比 —— 这种错误不会报错，只会悄悄答错。
            ("fingerprint", fingerprint),
            # 向量索引代：新索引整代写完之后才允许切过来。
            ("vector_generation", vector_generation),
        ):
            if value is not None:
                pairs.append((column, value))
        if not pairs:
            return
        sets = ", ".join(f"{c}=?" for c, _ in pairs)
        params = [v for _, v in pairs] + [_now_iso(), video_id]
        with self._tx() as conn:
            cur = conn.execute(
                f"UPDATE videos SET {sets}, updated_at=? WHERE video_id=?", params
            )  # noqa: S608
            if cur.rowcount == 0:
                raise NotFoundError("video", video_id)

    def clear_keyframes(self, video_id: str) -> int:
        """删掉一个视频的全部关键帧记录（重跑抽帧前调用，防孤行残留）。"""
        with self._tx() as conn:
            cursor = conn.execute("DELETE FROM keyframes WHERE video_id=?", (video_id,))
            removed = int(cursor.rowcount or 0)
        if removed:
            logger.debug("清空视频 %s 的 %d 条关键帧记录", video_id, removed)
        return removed

    def reset_outputs(self, video_id: str) -> list[str]:
        """清空一个视频的派生结果（切片/转写片段/关键帧），保留视频行本身。

        换 ASR 模型、换切分参数、换关键帧规格之后重跑，必须先清：
        旧向量和旧摘要混进新模型的结果里**不会报错**，只会让检索出现
        "同一条证据两个来源、分数不可比"，答出来看着通顺，引用却是错的。
        返回**需要调用方清理的**关键帧文件路径（同 ``delete_video``）。
        """
        kf_rows = self._query(
            "SELECT path FROM keyframes WHERE video_id=?", (video_id,)
        )
        paths = [str(r["path"]) for r in kf_rows if r["path"]]
        with self._tx() as conn:
            for table in ("clips", "transcript_segments", "keyframes"):
                conn.execute(f"DELETE FROM {table} WHERE video_id=?", (video_id,))  # noqa: S608
            conn.execute(
                "UPDATE videos SET status='pending', stage='', progress=0, error='',"
                " finished_at=?, updated_at=? WHERE video_id=?",
                ("", _now_iso(), video_id),
            )
        logger.info(
            "清空视频 %s 的派生结果：%d 个关键帧文件待清理", video_id, len(paths)
        )
        return paths

    def ensure_video_idle(self, video_id: str, conn=None) -> None:
        """任一关联会话在处理，或该视频正在分析时，拒绝联动删除。"""
        conn = conn or self.connection()
        if conn.execute("SELECT 1 FROM analysis_runs WHERE video_id=? AND status='running'", (video_id,)).fetchone():
            raise RepositoryError('该视频正在分析，请等待当前任务完成后删除')
        if conn.execute("SELECT 1 FROM tasks WHERE video_id=? AND status IN ('queued','running')", (video_id,)).fetchone():
            raise RepositoryError('该视频有运行中的任务，请等待任务完成后删除')
        if conn.execute("SELECT 1 FROM conversation_operations o JOIN conversations c ON c.id=o.conversation_id "
                        "WHERE c.video_id=? AND o.status IN ('queued','running')", (video_id,)).fetchone():
            raise RepositoryError('该视频的关联会话正在处理，请等待分析或问答完成后删除')

    def library_conversations(self, video_ids):
        """批量取最近活动的已命名会话，避免视频列表逐条查询。"""
        ids=list(dict.fromkeys(video_ids))
        if not ids:
            return {}
        rows=self._query('SELECT id,video_id,title,title_source FROM conversations WHERE video_id IN ('
            + ','.join('?' for _ in ids)+") ORDER BY (title_source='pending'),updated_ts DESC,id DESC",ids)
        result={}
        for row in rows:
            result.setdefault(row['video_id'],dict(row))
        return result

    def delete_video(self, video_id: str, *, missing_ok=False) -> list[str]:
        """删除视频及其级联记录，返回**需要调用方清理的**关键帧文件路径。
        数据库层不动文件系统 —— 删除是破坏性操作，交给上层决定
        （例如 ``--keep-frames`` 场景下只删库不删图）。
        """
        kf_rows = self._query(
            "SELECT path FROM keyframes WHERE video_id=?", (video_id,)
        )
        paths = [str(r["path"]) for r in kf_rows if r["path"]]
        from backend.storage.lifecycle import video_guard
        with video_guard(video_id), self._tx() as conn:
            self.ensure_video_idle(video_id, conn)
            exists = conn.execute('SELECT 1 FROM videos WHERE video_id=?', (video_id,)).fetchone()
            if not exists and not missing_ok:
                raise NotFoundError('video', video_id)
            sessions = conn.execute('SELECT session_id FROM conversations WHERE video_id=?', (video_id,)).fetchall()
            for sid in ['', *(row['session_id'] for row in sessions)]:
                conn.execute('INSERT OR IGNORE INTO conversation_imports VALUES (?)', (f'legacy:{video_id}:{sid}',))
            # 会话没有视频外键：必须与视频删除放在同一事务，子表由FK级联。
            conn.execute('DELETE FROM conversations WHERE video_id=?', (video_id,))
            # 旧版问答表没有视频外键，不能依赖ON DELETE CASCADE。
            conn.execute('DELETE FROM qa_history WHERE video_id=?', (video_id,))
            conn.execute('DELETE FROM videos WHERE video_id=?', (video_id,))
        logger.info("删除视频 %s，含 %d 个关键帧记录", video_id, len(paths))
        return paths

    # -- clips -------------------------------------------------------------

    def add_clips(self, video_id: str, clips: Iterable[Mapping[str, Any]]) -> int:
        """批量写入切片，返回写入条数。
        每个 ``clip`` 至少要有 ``clip_id``；缺 ``core_start_ms``/``core_end_ms``
        时回退到 ``start_ms``/``end_ms``。
        """
        payload: list[tuple[Any, ...]] = []
        now = _now_iso()
        for item in clips:
            clip_id = str(item.get("clip_id") or "").strip()
            if not clip_id:
                raise RepositoryError("clip 缺少 clip_id")
            start = int(item.get("start_ms") or 0)
            end = int(item.get("end_ms") or 0)
            if end < start:
                raise RepositoryError(
                    f"clip {clip_id} 的 end_ms({end}) < start_ms({start})"
                )
            core_start = int(item.get("core_start_ms", start))
            core_end = int(item.get("core_end_ms", end))
            kf_ids = item.get("keyframe_ids") or ()
            if isinstance(kf_ids, str):
                kf_json = json.dumps([kf_ids], ensure_ascii=False)
            else:
                kf_json = json.dumps([str(x) for x in kf_ids], ensure_ascii=False)
            payload.append(
                (
                    clip_id,
                    video_id,
                    int(item.get("index_no") or len(payload)),
                    start,
                    end,
                    core_start,
                    core_end,
                    str(item.get("text") or ""),
                    str(item.get("core_text") or ""),
                    kf_json,
                    int(bool(item.get("truncated", False))),
                    now,
                )
            )
        if not payload:
            return 0
        with self._tx() as conn:
            self._assert_capacity(conn, "clips", len(payload))
            conn.executemany(
                """
                INSERT INTO clips (
                    clip_id, video_id, index_no, start_ms, end_ms,
                    core_start_ms, core_end_ms, text, core_text,
                    keyframe_ids, truncated, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(clip_id) DO UPDATE SET
                    video_id=excluded.video_id,
                    index_no=excluded.index_no,
                    start_ms=excluded.start_ms,
                    end_ms=excluded.end_ms,
                    core_start_ms=excluded.core_start_ms,
                    core_end_ms=excluded.core_end_ms,
                    text=excluded.text,
                    core_text=excluded.core_text,
                    keyframe_ids=excluded.keyframe_ids,
                    truncated=excluded.truncated
                """,
                payload,
            )
        return len(payload)

    def get_clip(self, clip_id: str) -> ClipRecord:
        row = self._query_one("SELECT * FROM clips WHERE clip_id=?", (clip_id,))
        if row is None:
            raise NotFoundError("clip", clip_id)
        return _row_to_clip(row)

    def find_clip(self, clip_id: str) -> ClipRecord | None:
        row = self._query_one("SELECT * FROM clips WHERE clip_id=?", (clip_id,))
        return _row_to_clip(row) if row else None

    def get_clips(self, video_id: str) -> list[ClipRecord]:
        rows = self._query(
            "SELECT * FROM clips WHERE video_id=? ORDER BY index_no", (video_id,)
        )
        return [_row_to_clip(r) for r in rows]

    def iter_clip_payloads(self, video_id: str) -> Iterator[dict[str, Any]]:
        """流式产出切片字典，供向量库 / BM25 建索引，避免一次性占满内存。"""
        for row in self._query(
            "SELECT * FROM clips WHERE video_id=? ORDER BY index_no", (video_id,)
        ):
            yield _row_to_clip(row).to_dict()

    def clip_map(self, video_id: str) -> dict[str, dict[str, Any]]:
        """``clip_id → payload``，直接喂给 ``HybridSearcher(clip_index=...)``。"""
        return {c.clip_id: c.to_dict() for c in self.get_clips(video_id)}

    def set_clip_keyframes(self, clip_id: str, keyframe_ids: Sequence[str]) -> None:
        """回填关键帧绑定（抽帧在切片之后完成，需要二次写）。"""
        payload = json.dumps([str(x) for x in keyframe_ids], ensure_ascii=False)
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE clips SET keyframe_ids=? WHERE clip_id=?", (payload, clip_id)
            )
            if cur.rowcount == 0:
                raise NotFoundError("clip", clip_id)

    def delete_clips(self, video_id: str) -> int:
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM clips WHERE video_id=?", (video_id,))
            return cur.rowcount

    # -- segments ----------------------------------------------------------

    def add_segments(
        self,
        video_id: str,
        segments: Iterable[Mapping[str, Any]],
        *,
        replace: bool = True,
    ) -> int:
        """批量写入转写片段。``replace=True`` 先清空该视频的旧片段（幂等重跑）。"""
        payload: list[tuple[Any, ...]] = []
        for idx, item in enumerate(segments):
            seg_id = str(item.get("segment_id") or f"{video_id}-s{idx:05d}")
            start = int(item.get("start_ms") or 0)
            end = int(item.get("end_ms") or 0)
            if end < start:
                raise RepositoryError(f"segment {seg_id} 的 end_ms < start_ms")
            payload.append(
                (
                    seg_id,
                    video_id,
                    int(item.get("index_no", idx)),
                    start,
                    end,
                    str(item.get("text") or ""),
                    str(item.get("language") or ""),
                    float(item.get("confidence") or 0.0),
                    str(item.get("source") or "asr"),
                )
            )
        with self._tx() as conn:
            if replace:
                conn.execute(
                    "DELETE FROM transcript_segments WHERE video_id=?", (video_id,)
                )
            if payload:
                self._assert_capacity(conn, "transcript_segments", len(payload))
                conn.executemany(
                    """
                    INSERT INTO transcript_segments (
                        segment_id, video_id, index_no, start_ms, end_ms,
                        text, language, confidence, source
                    ) VALUES (?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(segment_id) DO UPDATE SET
                        index_no=excluded.index_no,
                        start_ms=excluded.start_ms,
                        end_ms=excluded.end_ms,
                        text=excluded.text,
                        language=excluded.language,
                        confidence=excluded.confidence,
                        source=excluded.source
                    """,
                    payload,
                )
        return len(payload)

    def get_segments(self, video_id: str) -> list[SegmentRecord]:
        rows = self._query(
            "SELECT * FROM transcript_segments WHERE video_id=? ORDER BY index_no",
            (video_id,),
        )
        return [_row_to_segment(r) for r in rows]

    # -- 分析报告持久化（v5） ----------------------------------------------- #

    def set_video_report(self, video_id: str, report: Mapping[str, Any]) -> None:
        """把一次成功分析的完整报告（``AnalysisReport.to_dict()``）存进视频行。

        存 JSON 文本而非拆列：报告结构跟着产品演进，读的时候原样吐回，
        老报告遇到新代码多一个少一个字段都不致命。
        """
        payload = json.dumps(dict(report), ensure_ascii=False)
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE videos SET report_json=? WHERE video_id=?", (payload, video_id)
            )
            if cur.rowcount == 0:
                raise NotFoundError("video", video_id)

    def get_video_report(self, video_id: str) -> dict[str, Any] | None:
        """取上次分析落库的报告；从未分析成功过则返回 ``None``。"""
        rows = self._query(
            "SELECT report_json FROM videos WHERE video_id=?", (video_id,)
        )
        if not rows:
            raise NotFoundError("video", video_id)
        raw = str(rows[0]["report_json"] or "")
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("视频 %s 的持久化报告损坏，忽略", video_id)
            return None
        return data if isinstance(data, dict) else None

    # -- keyframes ---------------------------------------------------------

    def add_keyframes(
        self,
        video_id: str,
        keyframes: Iterable[Mapping[str, Any]],
        *,
        replace: bool = True,
    ) -> int:
        """批量写入关键帧记录。"""
        payload: list[tuple[Any, ...]] = []
        now = _now_iso()
        for item in keyframes:
            kf_id = str(item.get("keyframe_id") or "").strip()
            if not kf_id:
                raise RepositoryError("keyframe 缺少 keyframe_id")
            clip_id = item.get("clip_id")
            payload.append(
                (
                    kf_id,
                    video_id,
                    str(clip_id) if clip_id else None,
                    int(item.get("scene_index") or 0),
                    int(item.get("timestamp_ms") or 0),
                    str(item.get("path") or ""),
                    int(item.get("width") or 0),
                    int(item.get("height") or 0),
                    int(item.get("size_bytes") or 0),
                    float(item.get("sharpness") or 0.0),
                    int(bool(item.get("sharpness_ok", True))),
                    int(bool(item.get("quality_ok", True))),
                    str(item.get("description") or ""),
                    str(item.get("ocr_text") or ""),
                    now,
                )
            )
        with self._tx() as conn:
            if replace:
                conn.execute("DELETE FROM keyframes WHERE video_id=?", (video_id,))
            if payload:
                self._assert_capacity(conn, "keyframes", len(payload))
                conn.executemany(
                    """
                    INSERT INTO keyframes (
                        keyframe_id, video_id, clip_id, scene_index, timestamp_ms, path,
                        width, height, size_bytes, sharpness, sharpness_ok, quality_ok,
                        description, ocr_text, created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(keyframe_id) DO UPDATE SET
                        clip_id=excluded.clip_id,
                        scene_index=excluded.scene_index,
                        timestamp_ms=excluded.timestamp_ms,
                        path=excluded.path,
                        width=excluded.width,
                        height=excluded.height,
                        size_bytes=excluded.size_bytes,
                        sharpness=excluded.sharpness,
                        sharpness_ok=excluded.sharpness_ok,
                        quality_ok=excluded.quality_ok,
                        description=excluded.description,
                        ocr_text=excluded.ocr_text
                    """,
                    payload,
                )
        return len(payload)

    def get_keyframe(self, keyframe_id: str) -> KeyframeRecord:
        row = self._query_one(
            "SELECT * FROM keyframes WHERE keyframe_id=?", (keyframe_id,)
        )
        if row is None:
            raise NotFoundError("keyframe", keyframe_id)
        return _row_to_keyframe(row)

    def get_keyframes(self, video_id: str) -> list[KeyframeRecord]:
        rows = self._query(
            "SELECT * FROM keyframes WHERE video_id=? ORDER BY timestamp_ms",
            (video_id,),
        )
        return [_row_to_keyframe(r) for r in rows]

    def keyframe_paths(self, video_id: str) -> dict[str, str]:
        """``keyframe_id → path``，喂给 ``HybridSearcher(keyframe_resolver=...)``。"""
        rows = self._query(
            "SELECT keyframe_id, path FROM keyframes WHERE video_id=? ORDER BY timestamp_ms",
            (video_id,),
        )
        return {str(r["keyframe_id"]): str(r["path"]) for r in rows if r["path"]}

    def set_keyframe_binding(self, video_id: str) -> int:
        """按时间包含关系把关键帧回绑到切片，返回更新条数。
        规则：关键帧时间戳落在 ``[start_ms, end_ms]`` 内即归属该片；
        一张帧只绑一个 ``clip_id``（多处命中取时间最近者）。
        """
        kfs = self.get_keyframes(video_id)
        clips = self.get_clips(video_id)
        if not kfs or not clips:
            return 0
        mapping: dict[str, list[str]] = {c.clip_id: [] for c in clips}
        updated = 0
        with self._tx() as conn:
            for kf in kfs:
                best: ClipRecord | None = None
                best_dist = 1 << 62
                for clip in clips:
                    if clip.start_ms <= kf.timestamp_ms <= clip.end_ms:
                        dist = abs(
                            kf.timestamp_ms
                            - (clip.core_start_ms + clip.core_end_ms) // 2
                        )
                        if dist < best_dist:
                            best, best_dist = clip, dist
                target = best.clip_id if best else None
                conn.execute(
                    "UPDATE keyframes SET clip_id=? WHERE keyframe_id=?",
                    (target, kf.keyframe_id),
                )
                if target:
                    mapping[target].append(kf.keyframe_id)
                    updated += 1
            for clip_id, ids in mapping.items():
                conn.execute(
                    "UPDATE clips SET keyframe_ids=? WHERE clip_id=?",
                    (json.dumps(ids, ensure_ascii=False), clip_id),
                )
        return updated

    # -- tasks -------------------------------------------------------------

    def upsert_task(
        self,
        video_id: str,
        *,
        task_id: str | None = None,
        kind: str = "ingest",
        status: TaskStatus = "pending",
        stage: str = "",
        progress: float = 0.0,
        message: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> str:
        tid = str(task_id or f"t{uuid.uuid4().hex[:12]}")
        now = _now_iso()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    task_id, video_id, kind, status, stage, progress, message,
                    error, attempts, payload, created_at, updated_at, updated_ts
                ) VALUES (?,?,?,?,?,?,'','',0,?,?,?,?)
                ON CONFLICT(task_id) DO UPDATE SET
                    video_id=excluded.video_id,
                    status=excluded.status,
                    stage=excluded.stage,
                    progress=excluded.progress,
                    message=excluded.message,
                    payload=excluded.payload,
                    -- 重新投递即新一轮执行：旧错误清空；每次转 running 计一次尝试
                    error='',
                    attempts=CASE
                        WHEN excluded.status='running' AND tasks.status<>'running'
                        THEN tasks.attempts + 1 ELSE tasks.attempts END,
                    updated_at=excluded.updated_at,
                    updated_ts=excluded.updated_ts
                """,
                (
                    tid,
                    video_id,
                    kind,
                    status,
                    stage,
                    float(progress),
                    json.dumps(dict(payload or {}), ensure_ascii=False, default=str),
                    now,
                    now,
                    _now_epoch(),
                ),
            )
        return tid

    def update_task(
        self,
        task_id: str,
        *,
        status: TaskStatus | None = None,
        stage: str | None = None,
        progress: float | None = None,
        message: str | None = None,
        error: str | None = None,
        bump_attempts: bool = False,
    ) -> None:
        sets = ["updated_at=?", "updated_ts=?"]
        params: list[Any] = [_now_iso(), _now_epoch()]
        for column, value in (
            ("status", status),
            ("stage", stage),
            ("message", message),
            ("error", error),
        ):
            if value is not None:
                sets.append(f"{column}=?")
                params.append(value)
        if progress is not None:
            sets.append("progress=?")
            params.append(max(0.0, min(1.0, float(progress))))
        if bump_attempts:
            sets.append("attempts=attempts+1")
        params.append(task_id)
        with self._tx() as conn:
            cur = conn.execute(
                f"UPDATE tasks SET {', '.join(sets)} WHERE task_id=?", params
            )  # noqa: S608
            if cur.rowcount == 0:
                raise NotFoundError("task", task_id)

    def get_task(self, task_id: str) -> TaskRecord:
        row = self._query_one("SELECT * FROM tasks WHERE task_id=?", (task_id,))
        if row is None:
            raise NotFoundError("task", task_id)
        return _row_to_task(row)

    def list_tasks(
        self,
        *,
        status: str | None = None,
        video_id: str | None = None,
        limit: int = 100,
    ) -> list[TaskRecord]:
        sql = "SELECT * FROM tasks"
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status=?")
            params.append(status)
        if video_id:
            clauses.append("video_id=?")
            params.append(video_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_ts DESC LIMIT ?"
        params.append(max(1, int(limit)))
        return [_row_to_task(r) for r in self._query(sql, params)]

    def recover_stale_tasks(
        self, *, stale_after_s: float = 3600.0, max_attempts: int = 5
    ) -> int:
        """把"卡在 running 太久"的任务退回 ``pending``，供进程重启后恢复。
        需求文档表 4：崩溃 → 任务队列持久化恢复。这里不猜"哪些该重跑"，
        只按心跳超时判定，恢复动作由调度器决定。

        ``stale_after_s`` 默认 1 小时：CPU 上 1 小时视频的转写本身就要 40 分钟，
        阶段内不落心跳是常态，判得过短会把仍在跑的任务误杀成"中断"。
        ``max_attempts`` 是恢复次数上限：无上限时，一个每次都让进程崩溃的
        视频会形成无限恢复循环。
        """
        cutoff = _now_epoch() - max(1.0, float(stale_after_s))
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE tasks SET status='pending', updated_at=?, updated_ts=?, "
                "stage='', error='恢复：上次执行中断', attempts=attempts+1 "
                "WHERE status='running' AND updated_ts < ? AND attempts < ?",
                (_now_iso(), _now_epoch(), cutoff, max(1, int(max_attempts))),
            )
            n = cur.rowcount
        if n:
            logger.warning("恢复 %d 个中断任务（running 超过 %.0fs）", n, stale_after_s)
        return n

    def prune_finished_tasks(self, *, keep: int = 500) -> int:
        """删除最旧的已完成/已失败任务记录，返回删除条数。

        tasks 是簿记表，不该占用容量红线；但无限累积同样不可接受 ——
        这里按"保留最近 ``keep`` 条终态记录"裁剪，运行中/等待中的任务永不删。
        """
        keep_n = max(0, int(keep))
        with self._tx() as conn:
            cur = conn.execute(
                "DELETE FROM tasks WHERE task_id IN ("
                "  SELECT task_id FROM tasks WHERE status IN ('done','failed','cancelled') "
                "  ORDER BY updated_ts DESC LIMIT -1 OFFSET ?"
                ")",
                (keep_n,),
            )
            return cur.rowcount

    def pending_tasks(self, limit: int = 50) -> list[TaskRecord]:
        return self.list_tasks(status="pending", limit=limit)

    # -- qa_history（多轮问答持久化）----------------------------------------

    def add_qa(
        self,
        video_id: str,
        *,
        question: str,
        answer: str,
        refs: Sequence[Mapping[str, Any]] | None = None,
        rewritten: str = "",
        elapsed_ms: int = 0,
        session_id: str = "",
        keep_per_video: int = 20,
    ) -> int:
        """记录一轮问答，返回行 id。同视频超过 ``keep_per_video`` 轮时裁最旧。"""
        import json as _json

        now_iso = _now_iso()
        with self._tx() as conn:
            cur = conn.execute(
                "INSERT INTO qa_history (video_id, question, answer, refs, rewritten,"
                " elapsed_ms, session_id, created_at, created_ts)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    video_id,
                    question,
                    answer,
                    _json.dumps(list(refs or []), ensure_ascii=False, default=str),
                    rewritten,
                    max(0, int(elapsed_ms)),
                    session_id,
                    now_iso,
                    _now_epoch(),
                ),
            )
            row_id = int(cur.lastrowid or 0)
            conn.execute(
                "DELETE FROM qa_history WHERE video_id=? AND id NOT IN ("
                "  SELECT id FROM qa_history WHERE video_id=?"
                "  ORDER BY created_ts DESC LIMIT ?"
                ")",
                (video_id, video_id, max(1, int(keep_per_video))),
            )
        return row_id

    def list_qa(
        self,
        video_id: str | None = None,
        *,
        session_id: str | None = None,
        limit: int = 200,
        newest: bool = False,
    ) -> list[dict[str, Any]]:
        """按时间升序列出问答历史（refs 自动反序列化）。"""
        import json as _json

        sql = "SELECT * FROM qa_history"
        clauses: list[str] = []
        params: list[Any] = []
        if video_id:
            clauses.append("video_id=?")
            params.append(video_id)
        if session_id is not None:
            clauses.append("session_id=?")
            params.append(session_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_ts DESC, id DESC LIMIT ?" if newest else " ORDER BY created_ts ASC LIMIT ?"
        params.append(max(1, int(limit)))
        out: list[dict[str, Any]] = []
        for row in self._query(sql, params):
            try:
                refs = _json.loads(row["refs"] or "[]")
            except Exception:  # noqa: BLE001
                refs = []
            out.append(
                {
                    "id": int(row["id"]),
                    "video_id": row["video_id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "refs": refs,
                    "rewritten": row["rewritten"],
                    "elapsed_ms": int(row["elapsed_ms"]),
                    "session_id": row["session_id"],
                    "created_at": row["created_at"],
                    "created_ts": row["created_ts"],
                }
            )
        return list(reversed(out)) if newest else out

    def list_qa_sessions(
        self, video_id: str | None = None, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        """按会话聚合（session_id 为空的历史归入 ``legacy`` 一组）。"""
        sql = (
            "SELECT video_id, session_id, COUNT(*) AS turns, MAX(created_ts) AS last_ts"
            " FROM qa_history"
        )
        clauses: list[str] = []
        params: list[Any] = []
        if video_id:
            clauses.append("video_id=?")
            params.append(video_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " GROUP BY video_id, session_id ORDER BY last_ts DESC LIMIT ?"
        params.append(max(1, int(limit)))
        return [
            {
                "video_id": r["video_id"],
                "session_id": r["session_id"] or "legacy",
                "turns": int(r["turns"]),
                "last_ts": float(r["last_ts"]),
            }
            for r in self._query(sql, params)
        ]

    def clear_qa(
        self, video_id: str | None = None, *, session_id: str | None = None
    ) -> int:
        """清空问答历史（全库 / 指定视频 / 指定会话），返回删除条数。"""
        sql = "DELETE FROM qa_history"
        clauses: list[str] = []
        params: list[Any] = []
        if video_id:
            clauses.append("video_id=?")
            params.append(video_id)
        if session_id:
            clauses.append("session_id=?")
            params.append(session_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self._tx() as conn:
            cur = conn.execute(sql, params)
            return cur.rowcount

    # -- artifacts ---------------------------------------------------------

    def add_artifact(
        self, video_id: str, *, kind: str, path: str, size_bytes: int = 0
    ) -> str:
        aid = f"a{uuid.uuid4().hex[:12]}"
        with self._tx() as conn:
            self._assert_capacity(conn, "artifacts", 1)
            conn.execute(
                "INSERT INTO artifacts (artifact_id, video_id, kind, path, size_bytes, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (aid, video_id, kind, path, int(size_bytes), _now_iso()),
            )
        return aid

    def get_artifacts(
        self, video_id: str, *, kind: str | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM artifacts WHERE video_id=?"
        params: list[Any] = [video_id]
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        sql += " ORDER BY created_at DESC"
        return [dict(r) for r in self._query(sql, params)]

    # -- 维护 --------------------------------------------------------------

    def orphan_keyframe_paths(self) -> list[str]:
        """库里已无对应记录、但磁盘上可能还留着的关键帧（交给调用方清理）。"""
        return [
            str(r["path"])
            for r in self._query("SELECT path FROM keyframes WHERE path <> ''")
            if r["path"] and not Path(str(r["path"])).exists()
        ]

    def vacuum(self) -> None:
        """整理碎片。仅在无并发写入时调用（例如 CLI 维护命令）。"""
        self.connection().execute("VACUUM")
        logger.info("VACUUM 完成: %s", self._path)

    def checkpoint(self) -> None:
        """把 WAL 内容刷回主库，便于备份/拷贝 db 文件。"""
        self.connection().execute("PRAGMA wal_checkpoint(TRUNCATE)")


# =============================================================================
# 表 → 记录
# =============================================================================


def _row_to_video(row: sqlite3.Row) -> VideoRecord:
    return VideoRecord(
        video_id=str(row["video_id"]),
        display_name=str(row["display_name"] or ""),
        source_kind="local" if str(row["source_kind"]) == "local" else "url",
        source_url=str(row["source_url"] or ""),
        local_path=str(row["local_path"] or ""),
        title=str(row["title"] or ""),
        duration_ms=int(row["duration_ms"] or 0),
        width=int(row["width"] or 0),
        height=int(row["height"] or 0),
        fps=float(row["fps"] or 0.0),
        has_audio=bool(row["has_audio"]),
        size_bytes=int(row["size_bytes"] or 0),
        language=str(row["language"] or ""),
        status=str(row["status"] or "pending"),  # type: ignore[arg-type]
        stage=str(row["stage"] or ""),
        progress=float(row["progress"] or 0.0),
        error=str(row["error"] or ""),
        fingerprint=str(row["fingerprint"] or "")
        if "fingerprint" in row.keys()
        else "",
        vector_generation=(
            str(row["vector_generation"] or "")
            if "vector_generation" in row.keys()
            else ""
        ),
        created_at=str(row["created_at"] or ""),
        updated_at=str(row["updated_at"] or ""),
        finished_at=str(row["finished_at"] or ""),
    )


def _row_to_clip(row: sqlite3.Row) -> ClipRecord:
    raw = str(row["keyframe_ids"] or "[]")
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        logger.debug("clip %s 的 keyframe_ids 不是合法 JSON: %r", row["clip_id"], raw)
        parsed = []
    kf_ids = tuple(str(x) for x in parsed) if isinstance(parsed, list) else ()
    return ClipRecord(
        clip_id=str(row["clip_id"]),
        video_id=str(row["video_id"]),
        index_no=int(row["index_no"] or 0),
        start_ms=int(row["start_ms"] or 0),
        end_ms=int(row["end_ms"] or 0),
        core_start_ms=int(row["core_start_ms"] or 0),
        core_end_ms=int(row["core_end_ms"] or 0),
        text=str(row["text"] or ""),
        core_text=str(row["core_text"] or ""),
        keyframe_ids=kf_ids,
        truncated=bool(row["truncated"]),
        created_at=str(row["created_at"] or ""),
    )


def _row_to_keyframe(row: sqlite3.Row) -> KeyframeRecord:
    clip_id = row["clip_id"]
    return KeyframeRecord(
        keyframe_id=str(row["keyframe_id"]),
        video_id=str(row["video_id"]),
        clip_id=str(clip_id) if clip_id else None,
        scene_index=int(row["scene_index"] or 0),
        timestamp_ms=int(row["timestamp_ms"] or 0),
        path=str(row["path"] or ""),
        width=int(row["width"] or 0),
        height=int(row["height"] or 0),
        size_bytes=int(row["size_bytes"] or 0),
        sharpness=float(row["sharpness"] or 0.0),
        sharpness_ok=bool(row["sharpness_ok"]),
        quality_ok=bool(row["quality_ok"]),
        description=str(row["description"] or ""),
        ocr_text=str(row["ocr_text"] or ""),
        created_at=str(row["created_at"] or ""),
    )


def _row_to_segment(row: sqlite3.Row) -> SegmentRecord:
    source = str(row["source"] or "asr")
    return SegmentRecord(
        segment_id=str(row["segment_id"]),
        video_id=str(row["video_id"]),
        index_no=int(row["index_no"] or 0),
        start_ms=int(row["start_ms"] or 0),
        end_ms=int(row["end_ms"] or 0),
        text=str(row["text"] or ""),
        language=str(row["language"] or ""),
        confidence=float(row["confidence"] or 0.0),
        source="subtitle" if source == "subtitle" else "asr",
    )


def _row_to_task(row: sqlite3.Row) -> TaskRecord:
    raw = str(row["payload"] or "{}")
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        parsed = {}
    return TaskRecord(
        task_id=str(row["task_id"]),
        video_id=str(row["video_id"]),
        kind=str(row["kind"] or "ingest"),
        status=str(row["status"] or "pending"),  # type: ignore[arg-type]
        stage=str(row["stage"] or ""),
        progress=float(row["progress"] or 0.0),
        message=str(row["message"] or ""),
        error=str(row["error"] or ""),
        attempts=int(row["attempts"] or 0),
        payload=parsed if isinstance(parsed, dict) else {},
        created_at=str(row["created_at"] or ""),
        updated_at=str(row["updated_at"] or ""),
        updated_ts=float(row["updated_ts"] or 0.0),
    )
