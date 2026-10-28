"""``backend.storage`` —— 结构化元数据仓库。

分层定位
--------
``core/storage`` 只负责**结构化元数据**（谁、什么时候、什么状态、哪一段文本），
它不认识向量也不认识文件内容：

* 向量 → ``core/rag/vector_store.py``
* 二进制文件 → 磁盘（``paths.frames`` / ``paths.videos`` / ``paths.exports``）

三者的粘合剂是 **溯源 ID**：``video_id`` / ``clip_id`` / ``keyframe_id``。
数据库是这些 ID 的唯一权威来源，向量库整库重建后仍能靠它们重新挂回。

导出的符号
----------
* :class:`SqliteRepository` —— SQLite 仓库本体（线程安全的连接池语义）
* :class:`VideoRecord` / :class:`ClipRecord` / :class:`KeyframeRecord`
  / :class:`SegmentRecord` / :class:`TaskRecord` —— 只读记录对象
* :class:`StorageStats` —— 容量水位（对应需求文档表 3）
* :func:`new_video_id` / :func:`ms_to_timecode` —— ID 与时间格式化工具
* 异常：:class:`StorageError` / :class:`RepositoryError`
  / :class:`NotFoundError` / :class:`CapacityExceededError`
"""

from __future__ import annotations

from backend.storage.repository import (
    SCHEMA_VERSION,
    CapacityExceededError,
    ClipRecord,
    KeyframeRecord,
    NotFoundError,
    RepositoryError,
    SegmentRecord,
    SqliteRepository,
    StorageError,
    StorageStats,
    TaskRecord,
    VideoRecord,
    VideoStatus,
    ms_to_timecode,
    new_video_id,
)

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
    "ms_to_timecode",
    "new_video_id",
]
