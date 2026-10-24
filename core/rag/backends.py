"""知影 ZhiYing —— 向量库的两种底层实现。

:class:`core.rag.vector_store.VectorStore` 是门面，负责集合命名、维度校验、
世代切换这些策略；真正的存取由本模块提供：

* **Chroma**（默认，``retrieval.backend: chroma``）—— HNSW 索引，查询快，
  需要 ``chromadb`` 包。
* **SQLite**（``retrieval.backend: sqlite``）—— 零额外依赖：向量存 BLOB，
  查询时全表扫一遍再算余弦。5 万条 1024 维大约 50 ms，够用；换来的是
  ``pip install chromadb`` 在某些 Windows 环境失败时（需求文档 2.4.3.4
  要求两种后端都支持），检索层仍然能跑。

两者对上层暴露同一套"集合对象"接口，所以 ``VectorStore`` 只改
``_ensure_client`` 一处即可切换：

============================  ====================================
``client.get_or_create_collection(name, metadata=..., embedding_function=...)``
``client.delete_collection(name)``
``collection.upsert(ids=, embeddings=, documents=, metadatas=)``
``collection.query(query_embeddings=, n_results=, include=, where=, where_document=)``
``collection.delete(ids=)`` / ``collection.delete(where=)``
``collection.count()``
============================  ====================================
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Any, Final, Mapping, Protocol, Sequence, runtime_checkable

logger = logging.getLogger("zhiying.rag.backends")


# =============================================================================
# 后端契约
# =============================================================================


@runtime_checkable
class VectorCollection(Protocol):
    """一个向量集合的最小契约。

    Chroma 的 ``Collection`` 与 :class:`SqliteCollection` 都满足，且都**不**
    继承本类 —— ``Protocol`` 是结构化类型，用来给 ``VectorStore`` 做注解与
    静态检查，不在运行时改变任何东西。
    """

    name: str

    def count(self) -> int:
        """集合内条目数。"""

    def upsert(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str] | None = None,
        metadatas: Sequence[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """按主键写入或覆盖。"""

    def query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 10,
        include: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
        where_document: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """近邻检索，返回 Chroma 同形的 ``{"ids": [[...]], ...}`` 字典。"""

    def delete(
        self,
        *,
        ids: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """按主键或元数据条件删除。"""


@runtime_checkable
class VectorBackend(Protocol):
    """向量库客户端契约。``VectorStore._ensure_client()`` 返回其中之一。"""

    def get_or_create_collection(
        self,
        name: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        embedding_function: Any | None = None,
    ) -> VectorCollection:
        """取集合，不存在则建。"""

    def delete_collection(self, name: str) -> None:
        """删除整个集合（不存在时静默）。"""

    def list_collections(self) -> Sequence[Any]:
        """所有集合名或集合对象。两种形态的元素在调用方统一归一化。"""

    def close(self) -> None:
        """释放连接与资源。"""

try:  # numpy 是项目依赖；真缺了也要能退回纯 Python
    import numpy as _np
except ImportError:  # pragma: no cover - 环境缺 numpy 时
    _np = None


__all__ = ["SqliteClient", "SqliteCollection", "SqliteVectorError"]


class SqliteVectorError(RuntimeError):
    """SQLite 向量后端出错。"""


_DDL: Final[tuple[str, ...]] = (
    "CREATE TABLE IF NOT EXISTS collections ("
    " name TEXT PRIMARY KEY,"
    " metric TEXT NOT NULL DEFAULT 'cosine',"
    " dim INTEGER NOT NULL DEFAULT 0)",
    "CREATE TABLE IF NOT EXISTS vectors ("
    " collection TEXT NOT NULL,"
    " id TEXT NOT NULL,"
    " doc TEXT NOT NULL DEFAULT '',"
    " meta TEXT NOT NULL DEFAULT '{}',"
    " vec BLOB NOT NULL,"
    " PRIMARY KEY (collection, id))",
    "CREATE INDEX IF NOT EXISTS idx_vectors_collection ON vectors(collection)",
)

_UPSERT_SQL: Final[str] = (
    "INSERT INTO vectors(collection, id, doc, meta, vec) VALUES(?, ?, ?, ?, ?) "
    "ON CONFLICT(collection, id) DO UPDATE SET "
    " doc=excluded.doc, meta=excluded.meta, vec=excluded.vec"
)


# -----------------------------------------------------------------------------
# 编解码与距离
# -----------------------------------------------------------------------------


def _pack(vector: Sequence[float]) -> bytes:
    if _np is not None:
        return _np.asarray(list(vector), dtype=_np.float32).tobytes()
    return struct.pack(f"<{len(vector)}f", *(float(x) for x in vector))


def _unpack(blob: bytes) -> tuple[float, ...]:
    count = len(blob) // 4
    return struct.unpack(f"<{count}f", blob)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return sum(x * x for x in a) ** 0.5


def _distance(a: Sequence[float], b: Sequence[float], metric: str) -> float:
    """统一成"距离"（越小越近），与 Chroma 的语义对齐。"""
    if metric == "cosine":
        na, nb = _norm(a), _norm(b)
        if na == 0.0 or nb == 0.0:
            return 1.0
        return 1.0 - _dot(a, b) / (na * nb)
    if metric == "ip":
        return -_dot(a, b)
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _distances_np(queries: "_np.ndarray", matrix: "_np.ndarray", metric: str) -> "_np.ndarray":
    """（n_query, n_vec）距离矩阵。"""
    if metric == "cosine":
        q = queries / (queries * queries).sum(axis=1, keepdims=True) ** 0.5
        m = matrix / (matrix * matrix).sum(axis=1, keepdims=True) ** 0.5
        return 1.0 - q @ m.T
    if metric == "ip":
        return -(queries @ matrix.T)
    # l2：用展开式避免 (n_query, n_vec, dim) 的大中间数组
    q2 = (queries * queries).sum(axis=1)[:, None]
    m2 = (matrix * matrix).sum(axis=1)[None, :]
    return _np.sqrt(_np.maximum(q2 + m2 - 2.0 * (queries @ matrix.T), 0.0))


def _match_condition(meta: Mapping[str, Any], cond: Mapping[str, Any]) -> bool:
    """Chroma ``where`` 语义的子集：等值 / ``$eq`` / ``$ne`` / ``$in`` / ``$and`` / ``$or``。

    不认识的算子**抛错**而不是放行 —— 过滤器静默失效比报错危险得多：
    ``where={"$and": [...]}`` 被忽略意味着删数据时多删、查询时多回。
    """
    for key, value in cond.items():
        if key == "$and":
            if not all(_match_condition(meta, c) for c in (value or [])):
                return False
        elif key == "$or":
            if not any(_match_condition(meta, c) for c in (value or [])):
                return False
        elif key.startswith("$"):
            raise SqliteVectorError(f"SQLite 向量后端不支持 where 算子 {key!r}")
        elif isinstance(value, Mapping) and any(str(k).startswith("$") for k in value):
            for op, operand in value.items():
                if op == "$eq":
                    matched = meta.get(key) == operand
                elif op == "$ne":
                    matched = meta.get(key) != operand
                elif op == "$in":
                    matched = meta.get(key) in list(operand or [])
                else:
                    raise SqliteVectorError(
                        f"SQLite 向量后端不支持 where 算子 {op!r}"
                    )
                if not matched:
                    return False
        elif meta.get(key) != value:
            return False
    return True


def _matches(meta: Mapping[str, Any], where: Mapping[str, Any] | None) -> bool:
    if not where:
        return True
    return _match_condition(meta, where)


def _doc_matches(text: str, where_document: Mapping[str, Any] | None) -> bool:
    if not where_document:
        return True
    needle = where_document.get("$contains")
    if needle is None:
        return True
    return str(needle) in (text or "")


# -----------------------------------------------------------------------------
# 集合
# -----------------------------------------------------------------------------


class SqliteCollection:
    """单个集合。接口与 Chroma 的 ``Collection`` 保持最小一致。"""

    def __init__(self, client: SqliteClient, name: str) -> None:
        self._client = client
        self.name = name

    # -- 写 ---------------------------------------------------------------- #

    def upsert(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str] | None = None,
        metadatas: Sequence[Mapping[str, Any]] | None = None,
        **_ignored: Any,
    ) -> None:
        if not ids:
            return
        docs = list(documents) if documents is not None else [""] * len(ids)
        metas = list(metadatas) if metadatas is not None else [{} for _ in ids]
        rows: list[tuple[str, str, str, str, bytes]] = []
        dim = 0
        for index, vector_id in enumerate(ids):
            vector = [float(x) for x in embeddings[index]]
            dim = dim or len(vector)
            meta = dict(metas[index] or {}) if index < len(metas) else {}
            doc = str(docs[index] or "") if index < len(docs) else ""
            rows.append(
                (
                    self.name,
                    str(vector_id),
                    doc,
                    json.dumps(meta, ensure_ascii=False, default=str),
                    _pack(vector),
                )
            )
        try:
            with self._client.lock:
                self._client.conn.executemany(_UPSERT_SQL, rows)
                self._client.conn.execute(
                    "UPDATE collections SET dim=? WHERE name=? AND dim=0", (dim, self.name)
                )
                self._client.conn.commit()
        except sqlite3.Error as exc:
            raise SqliteVectorError(f"写入集合 {self.name!r} 失败: {exc}") from exc

    def delete(
        self,
        *,
        ids: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
        **_ignored: Any,
    ) -> None:
        try:
            with self._client.lock:
                if ids:
                    self._client.conn.executemany(
                        "DELETE FROM vectors WHERE collection=? AND id=?",
                        [(self.name, str(i)) for i in ids],
                    )
                    self._client.conn.commit()
                    return
                if where:
                    doomed = self._select(where=where, where_document=None)
                    if doomed:
                        self._client.conn.executemany(
                            "DELETE FROM vectors WHERE collection=? AND id=?",
                            [(self.name, row["id"]) for row in doomed],
                        )
                        self._client.conn.commit()
                    return
                self._client.conn.execute(
                    "DELETE FROM vectors WHERE collection=?", (self.name,)
                )
                self._client.conn.commit()
        except sqlite3.Error as exc:
            raise SqliteVectorError(f"删除集合 {self.name!r} 的记录失败: {exc}") from exc

    # -- 读 ---------------------------------------------------------------- #

    def count(self) -> int:
        try:
            with self._client.lock:
                row = self._client.conn.execute(
                    "SELECT COUNT(*) AS n FROM vectors WHERE collection=?", (self.name,)
                ).fetchone()
            return int(row["n"]) if row else 0
        except sqlite3.Error as exc:
            raise SqliteVectorError(f"统计集合 {self.name!r} 失败: {exc}") from exc

    def _select(
        self,
        *,
        where: Mapping[str, Any] | None,
        where_document: Mapping[str, Any] | None,
    ) -> list[sqlite3.Row]:
        with self._client.lock:
            rows = self._client.conn.execute(
                "SELECT id, doc, meta, vec FROM vectors WHERE collection=?", (self.name,)
            ).fetchall()
        if not where and not where_document:
            return list(rows)
        keep: list[sqlite3.Row] = []
        for row in rows:
            try:
                meta = json.loads(row["meta"] or "{}")
            except json.JSONDecodeError:
                meta = {}
            if _matches(meta, where) and _doc_matches(row["doc"], where_document):
                keep.append(row)
        return keep

    def query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 10,
        include: Sequence[str] | None = None,  # noqa: ARG002 - 与 Chroma 签名兼容
        where: Mapping[str, Any] | None = None,
        where_document: Mapping[str, Any] | None = None,
        **_ignored: Any,
    ) -> dict[str, Any]:
        """返回结构与 Chroma 一致：每个键都是"按查询分组的二维列表"。"""
        queries = [[float(x) for x in q] for q in query_embeddings]
        empty: dict[str, Any] = {
            "ids": [[] for _ in queries],
            "distances": [[] for _ in queries],
            "metadatas": [[] for _ in queries],
            "documents": [[] for _ in queries],
        }
        if not queries:
            return empty

        rows = self._select(where=where, where_document=where_document)
        dim = len(queries[0])
        rows = [row for row in rows if len(row["vec"]) == dim * 4]
        if not rows:
            return empty

        metric = self._client.metric_of(self.name)
        limit = max(1, int(n_results))
        payload = {key: [] for key in empty}
        for index, vector in enumerate(queries):
            ranked = self._rank(vector, rows, metric, limit)
            payload["ids"].append([item[0] for item in ranked])
            payload["distances"].append([item[1] for item in ranked])
            payload["metadatas"].append([item[2] for item in ranked])
            payload["documents"].append([item[3] for item in ranked])
        return payload

    def _rank(
        self,
        query: Sequence[float],
        rows: Sequence[sqlite3.Row],
        metric: str,
        limit: int,
    ) -> list[tuple[str, float, dict[str, Any], str]]:
        """返回 ``(id, distance, meta, doc)``，按距离升序。"""
        metas: list[dict[str, Any]] = []
        for row in rows:
            try:
                metas.append(json.loads(row["meta"] or "{}"))
            except json.JSONDecodeError:
                metas.append({})

        if _np is not None:
            count, dim = len(rows), len(query)
            matrix = _np.frombuffer(
                b"".join(row["vec"] for row in rows), dtype=_np.float32
            ).reshape(count, dim)
            scores = _distances_np(
                _np.asarray([query], dtype=_np.float32), matrix, metric
            )[0]
            # stable 排序：平分时名次确定（RRF 对 rank 敏感，结果不该抖动）
            order = _np.argsort(scores, kind="stable")[:limit]
            return [
                (rows[i]["id"], float(scores[i]), metas[i], rows[i]["doc"])
                for i in order
            ]

        scored = [
            (_distance(query, _unpack(row["vec"]), metric), row)
            for row in rows
        ]
        scored.sort(key=lambda item: item[0])
        return [
            (row["id"], float(distance), metas[pos], row["doc"])
            for pos, (distance, row) in enumerate(scored[:limit])
        ]


# -----------------------------------------------------------------------------
# 客户端
# -----------------------------------------------------------------------------


class SqliteClient:
    """最小可用的"Chroma 客户端"替身：一个 SQLite 文件、两张表。

    与 Chroma 的关键差异：

    * 没有 HNSW 索引，查询是线性扫描（5 万条 1024 维约 50 ms）。
    * 集合名不受 Chroma 的 3-512 字符限制，但仍沿用同样的命名规则。
    * 数据与项目其他元数据可以用同一个备份策略，不需要单独导出。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._closed = False
        try:
            self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            # 多进程同时打开同一个 vectors.db 时兜住锁等待，
            # 否则立刻 "database is locked"（仓储主库设了 5s，这里对齐）。
            self.conn.execute("PRAGMA busy_timeout=5000")
            for ddl in _DDL:
                self.conn.execute(ddl)
            self.conn.commit()
        except sqlite3.Error as exc:
            raise SqliteVectorError(f"打开向量库失败({self.path}): {exc}") from exc
        logger.info("SQLite 向量库已就绪: %s", self.path)

    # -- Chroma 兼容接口 --------------------------------------------------- #

    def get_or_create_collection(
        self,
        name: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        embedding_function: Any | None = None,  # noqa: ARG002 - 与 Chroma 签名兼容
    ) -> SqliteCollection:
        metric = str((metadata or {}).get("hnsw:space", "cosine") or "cosine").lower()
        try:
            with self.lock:
                row = self.conn.execute(
                    "SELECT name FROM collections WHERE name=?", (name,)
                ).fetchone()
                if row is None:
                    self.conn.execute(
                        "INSERT INTO collections(name, metric) VALUES(?, ?)", (name, metric)
                    )
                    self.conn.commit()
        except sqlite3.Error as exc:
            raise SqliteVectorError(f"创建集合 {name!r} 失败: {exc}") from exc
        return SqliteCollection(self, name)

    def delete_collection(self, name: str) -> None:
        try:
            with self.lock:
                self.conn.execute("DELETE FROM vectors WHERE collection=?", (name,))
                self.conn.execute("DELETE FROM collections WHERE name=?", (name,))
                self.conn.commit()
        except sqlite3.Error as exc:
            raise SqliteVectorError(f"删除集合 {name!r} 失败: {exc}") from exc

    # -- 辅助 -------------------------------------------------------------- #

    def metric_of(self, name: str) -> str:
        with self.lock:
            row = self.conn.execute(
                "SELECT metric FROM collections WHERE name=?", (name,)
            ).fetchone()
        return str(row["metric"]).lower() if row else "cosine"

    def dimension_of(self, name: str) -> int:
        with self.lock:
            row = self.conn.execute(
                "SELECT dim FROM collections WHERE name=?", (name,)
            ).fetchone()
        return int(row["dim"]) if row else 0

    def collections(self) -> list[str]:
        with self.lock:
            rows = self.conn.execute("SELECT name FROM collections ORDER BY name").fetchall()
        return [str(row["name"]) for row in rows]

    def list_collections(self) -> list[dict[str, str]]:
        """与 Chroma 的 ``list_collections()`` 对齐（返回带 ``name`` 的记录）。"""
        return [{"name": name} for name in self.collections()]

    def close(self) -> None:
        with self.lock:
            if self._closed:
                return
            self._closed = True
            self.conn.close()
