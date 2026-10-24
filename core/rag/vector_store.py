"""知影 ZhiYing —— ChromaDB 多模态向量库封装。

职责边界
--------
本模块只管**向量的存放与近邻检索**，不产生向量：

* 文本向量（BGE-M3）→ ``core/models/embed.py``
* 图像向量（SigLIP2）→ ``core/models/embed.py``
* 元数据与溯源锚点 → ``core/storage/repository.py``

两个集合（对应需求文档表 3 的两类向量）：

===============  ==================  ==================  ==========================
集合             向量维度            主键                用途
===============  ==================  ==================  ==========================
``clip_text``    1024（BGE-M3）      ``clip_id``         文本语义通道
``frame_image``  768（SigLIP2-B）    ``keyframe_id``     图文跨模态通道
===============  ==================  ==================  ==========================

两条硬约束
----------
1. **ChromaDB 的 metadata 只接受 str / int / float / bool**。列表必须序列化
   （``keyframe_ids`` 存成逗号串），否则写入时抛 ``ValueError``，且报错信息不含字段名，
   极难排查。
2. **维度必须固定**。集合一旦建成，写入不同维度的向量会直接抛错。这里首次写入时
   记住维度，之后不一致抛 :class:`DimensionMismatchError` 并给出可读提示。

降级策略（需求文档表 4）
------------------------
向量库不可用（磁盘满、chroma 未安装、集合损坏）时抛
:class:`VectorStoreUnavailableError`。调用方捕获后应退到
``HybridSearcher`` 的 BM25 通道 —— 纯字面检索，精度略降但不至于白屏。

典型用法
--------
>>> store = ChromaVectorStore.from_config(get_app_config())
>>> store.upsert_clips(
...     ids=[c.clip_id for c in clips],
...     embeddings=text_vectors,
...     documents=[c.text for c in clips],
...     metadatas=[{"clip_id": c.clip_id, "video_id": c.video_id,
...                  "start_ms": c.start_ms, "end_ms": c.end_ms} for c in clips],
... )
>>> hits = store.query_clips(query_vector, top_k=8)
>>> hits[0].score, hits[0].metadata["clip_id"]
"""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from core.config import AppConfig, get_app_config

if TYPE_CHECKING:  # 只为注解；运行时由 _ensure_client 延迟导入，避免硬依赖
    from core.rag.backends import VectorBackend

logger = logging.getLogger("zhiying.rag.vector_store")

__all__ = [
    "CLIP_COLLECTION",
    "FRAME_COLLECTION",
    "CollectionKind",
    "DimensionMismatchError",
    "VectorHit",
    "VectorStore",
    "VectorStoreError",
    "VectorStoreUnavailableError",
]

CollectionKind = Literal["clip", "frame"]

#: 集合名的语义别名 → 默认真实集合名
CLIP_COLLECTION: Final[str] = "clip_text"
FRAME_COLLECTION: Final[str] = "frame_image"

_NAME_PATTERN: Final[str] = r"^[A-Za-z0-9][A-Za-z0-9._-]{1,510}[A-Za-z0-9]$"


# =============================================================================
# 异常
# =============================================================================


class VectorStoreError(RuntimeError):
    """向量库错误基类。"""


class VectorStoreUnavailableError(VectorStoreError):
    """向量库不可用 —— 调用方应降级到字面检索（BM25）。"""


class DimensionMismatchError(VectorStoreError):
    """向量维度与集合既有维度不一致。"""

    def __init__(self, collection: str, expected: int, got: int) -> None:
        super().__init__(
            f"集合 {collection!r} 期望 {expected} 维向量，本次收到 {got} 维。"
            f"切换 embedding 模型后需要重建该集合（store.reset(...) 后重新入库）。"
        )
        self.collection = collection
        self.expected = expected
        self.got = got


# =============================================================================
# 结果对象
# =============================================================================


@dataclass(frozen=True, slots=True)
class VectorHit:
    """一条近邻结果。

    ``distance`` 是 Chroma 原始距离；``score`` 统一折算成"越大越像"：

    * cosine 空间：``score = 1 - distance``（范围 -1~1，通常 0~1）
    * l2 空间：``score = 1 / (1 + distance)``
    """

    id: str
    distance: float
    score: float
    document: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    @property
    def clip_id(self) -> str:
        return str(self.metadata.get("clip_id") or self.id)

    @property
    def video_id(self) -> str:
        return str(self.metadata.get("video_id") or "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "distance": round(self.distance, 6),
            "score": round(self.score, 6),
            "document": self.document,
            "metadata": dict(self.metadata),
        }


# =============================================================================
# 工具
# =============================================================================


def _flatten_metadata(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    """把任意 metadata 压成 Chroma 能接受的四类标量。

    * ``None`` → 整个键**不写入**（Chroma 不接受 None）
    * ``list`` / ``tuple`` / ``set`` / ``dict`` → JSON 或逗号串
    * ``Path`` → ``str``
    * ``bool`` 必须先于 ``int`` 判断（``bool`` 是 ``int`` 的子类）
    """
    out: dict[str, Any] = {}
    for key, value in (meta or {}).items():
        if not isinstance(key, str) or not key:
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, (int, str)):
            out[key] = value
        elif isinstance(value, float):
            out[key] = float(value)
        elif isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (list, tuple, set)):
            items = list(value)
            if all(isinstance(x, str) for x in items):
                out[key] = ",".join(str(x) for x in items)
            else:
                out[key] = json.dumps(items, ensure_ascii=False, default=str)
        elif isinstance(value, Mapping):
            out[key] = json.dumps(dict(value), ensure_ascii=False, default=str)
        else:
            out[key] = str(value)
    return out


def _check_ids(ids: Sequence[str]) -> list[str]:
    out = [str(x) for x in ids]
    if any(not x for x in out):
        raise VectorStoreError("向量 id 不能为空字符串")
    return out


def _as_matrix(embeddings: Any, *, expected_rows: int) -> list[list[float]]:
    """接受 list[list[float]] 或 numpy 二维数组，统一成 list[list[float]]。"""
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    if not isinstance(embeddings, (list, tuple)):
        raise VectorStoreError(f"embeddings 必须是二维数组，收到 {type(embeddings).__name__}")
    matrix: list[list[float]] = []
    for row in embeddings:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)):
            raise VectorStoreError("embeddings 的元素必须是一维数组")
        matrix.append([float(x) for x in row])
    if len(matrix) != expected_rows:
        raise VectorStoreError(
            f"embeddings 行数({len(matrix)}) 与 ids 数量({expected_rows}) 不一致"
        )
    return matrix


# =============================================================================
# 仓库
# =============================================================================


class VectorStore:
    """ChromaDB 持久化客户端封装。

    Parameters
    ----------
    persist_dir:
        向量库落盘目录（``paths.chroma``）。父目录自动创建。
    clip_collection / frame_collection:
        两个集合的名字。改名等于换库，旧的向量不会迁移。
    metric:
        ``cosine``（默认）/ ``l2`` / ``ip``。必须与 embedding 模型的
        训练目标一致：BGE-M3 与 SigLIP2 都用 cosine。
    verify_dim:
        是否在写入时校验维度一致性。
    client:
        注入已有客户端（测试用 ``EphemeralClient`` / ``SqliteClient``）。
    backend:
        ``chroma``（默认）或 ``sqlite``。后者零额外依赖，向量存在
        ``persist_dir/vectors.db``，查询走线性扫描。
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        *,
        clip_collection: str = CLIP_COLLECTION,
        frame_collection: str = FRAME_COLLECTION,
        metric: str = "cosine",
        verify_dim: bool = True,
        client: Any | None = None,
        backend: str = "chroma",
    ) -> None:
        self._clip_name = self._validate_name(clip_collection, "clip_collection")
        self._frame_name = self._validate_name(frame_collection, "frame_collection")
        if self._clip_name == self._frame_name:
            raise VectorStoreError("clip_collection 与 frame_collection 不能同名")
        self._metric = str(metric or "cosine").lower()
        if self._metric not in {"cosine", "l2", "ip"}:
            raise VectorStoreError(f"不支持的 metric: {metric!r}（可选 cosine/l2/ip）")
        self._backend = str(backend or "chroma").lower()
        if self._backend not in {"chroma", "sqlite"}:
            raise VectorStoreError(f"不支持的向量库后端: {backend!r}（可选 chroma/sqlite）")
        self._verify_dim = bool(verify_dim)
        self._persist_dir = Path(persist_dir).expanduser() if persist_dir else None
        self._lock = threading.RLock()
        self._dims: dict[str, int] = {}
        self._collections: dict[str, Any] = {}
        self._client: Any | None = client
        self._closed = False
        self._generation = ""

    @staticmethod
    def _validate_name(name: str, field_name: str) -> str:
        text = str(name or "").strip()
        if not re.match(_NAME_PATTERN, text):
            raise VectorStoreError(
                f"{field_name}={name!r} 不是合法的 Chroma 集合名："
                f"长度 3-512，仅 [A-Za-z0-9._-]，首尾必须是字母或数字"
            )
        return text

    # -- 构造 --------------------------------------------------------------

    @classmethod
    def from_config(cls, config: AppConfig | None = None) -> VectorStore:
        """按 ``paths.chroma``、``storage.chroma`` 与 ``retrieval.backend`` 构造。"""
        cfg = config or get_app_config()
        persist = Path(cfg.paths.chroma)
        if not persist.is_absolute():
            persist = Path(cfg.root_dir) / persist
        return cls(
            persist,
            clip_collection=cfg.storage.chroma.clip_collection,
            frame_collection=cfg.storage.chroma.frame_collection,
            metric=cfg.storage.chroma.metric,
            verify_dim=cfg.storage.chroma.verify_dim,
            backend=cfg.retrieval.backend,
        )

    @classmethod
    def ephemeral(cls, **kwargs: Any) -> VectorStore:
        """内存实例（测试 / demo 用，不落盘）。"""
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreUnavailableError(
                "未安装 chromadb，无法创建向量库：pip install chromadb"
            ) from exc
        kwargs.setdefault("clip_collection", CLIP_COLLECTION)
        kwargs.setdefault("frame_collection", FRAME_COLLECTION)
        return cls(None, client=chromadb.EphemeralClient(), **kwargs)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def persist_dir(self) -> Path | None:
        return self._persist_dir

    @property
    def metric(self) -> str:
        return self._metric

    def collection_name(self, kind: CollectionKind) -> str:
        base = self._clip_name if kind == "clip" else self._frame_name
        return f"{base}_{self._generation}" if self._generation else base

    # -- 索引代（generation）-----------------------------------------------

    @property
    def generation(self) -> str:
        return self._generation

    def with_generation(self, generation: str) -> VectorStore:
        """返回共享后端、但集合名带代号的新实例。

        分析时先往 ``clip_text_<gen>`` / ``frame_image_<gen>`` 写，全部成功
        之后才把库里的活动指针切过去。中途模型或网络故障只丢半成品，上一代
        索引仍然完整可用 —— 固定集合名直连 ``upsert`` 的做法一旦重建失败，
        旧索引就跟着毁了。
        """
        text = str(generation or "").strip()
        if not text:
            return self
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", text):
            raise VectorStoreError(
                f"generation={generation!r} 只能包含 [A-Za-z0-9._-] 且以字母数字开头"
            )
        # 浅拷贝而不是手抄字段：__init__ 新增成员时不会漏掉（漏一个就是
        # AttributeError，且只在写入路径上炸，测试不覆盖到就上生产了）。
        clone = copy.copy(self)
        clone._collections = {}
        clone._closed = False
        clone._generation = text
        return clone

    def _names_for(self, kind: CollectionKind) -> tuple[str, ...]:
        """某类集合在当前代号下可能的名字（带代 + 不带代，兼容旧数据）。"""
        base = self._clip_name if kind == "clip" else self._frame_name
        if not self._generation:
            return (base,)
        return (f"{base}_{self._generation}", f"{base}")

    def activate_generation(self, generation: str) -> None:
        """把本次活动代号定下来（之后 ``collection_name`` 带后缀）。"""
        self._generation = str(generation or "").strip()
        self._collections = {}

    def drop_generation(self, generation: str) -> int:
        """删掉某一代的两张集合，返回删掉的集合数。

        ``generation=""`` 表示删**无后缀的基础集合**（即最早的那一代）。
        索引写到一半失败时调用：半成品留着只会占空间，而且下次同名重建会
        把残留向量和新向量混在一起。
        """
        text = str(generation or "").strip()
        suffix = f"_{text}" if text else ""
        client = self._ensure_client()
        wanted = {f"{self._clip_name}{suffix}", f"{self._frame_name}{suffix}"}
        removed = 0
        try:
            existing = list(client.list_collections())
        except Exception as exc:  # noqa: BLE001 - 纯清理，失败不影响主流程
            logger.warning("列出集合失败，跳过清理: %s", exc)
            return 0
        for item in existing:
            name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else str(item))
            if name in wanted:
                try:
                    client.delete_collection(name)
                    removed += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("删除集合 %s 失败: %s", name, exc)
        self._collections = {}
        return removed

    # -- 客户端 / 集合 ------------------------------------------------------

    def _ensure_client(self) -> VectorBackend:
        if self._closed:
            raise VectorStoreUnavailableError("向量库已关闭")
        if self._client is not None:
            return self._client
        target = self._persist_dir or Path("data/chroma")
        target.mkdir(parents=True, exist_ok=True)

        if self._backend == "sqlite":
            from core.rag.backends import SqliteClient, SqliteVectorError

            try:
                self._client = SqliteClient(target / "vectors.db")
            except SqliteVectorError as exc:
                raise VectorStoreUnavailableError(f"打开 SQLite 向量库失败: {exc}") from exc
            logger.info("向量库后端: sqlite（%s）", target / "vectors.db")
            return self._client

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise VectorStoreUnavailableError(
                "未安装 chromadb，无法使用向量检索："
                "pip install chromadb，或把 retrieval.backend 设为 sqlite"
            ) from exc
        try:
            self._client = chromadb.PersistentClient(
                path=str(target),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
        except Exception as exc:  # noqa: BLE001 - chroma 内部异常类型不稳定
            raise VectorStoreUnavailableError(f"打开 Chroma 持久化目录失败({target}): {exc}") from exc
        logger.info("向量库后端: chroma（%s）", target)
        return self._client

    def collection(self, kind: CollectionKind, *, create: bool = True) -> Any | None:
        """取原生集合对象（可直接喂给 ``VectorChannel``）。

        ``create=False`` 时集合不存在返回 ``None`` —— 检索是**读路径**，
        不该因为一次查询就凭空建出空集合（还会让 stats 的集合数虚涨）。
        """
        with self._lock:
            cached = self._collections.get(kind)
            if cached is not None:
                return cached
            client = self._ensure_client()
            name = self.collection_name(kind)
            try:
                if not create:
                    try:
                        existing = list(client.list_collections())
                    except Exception as exc:  # noqa: BLE001
                        raise VectorStoreUnavailableError(f"列出集合失败: {exc}") from exc
                    names = {
                        getattr(c, "name", None)
                        or (c.get("name") if isinstance(c, dict) else str(c))
                        for c in existing
                    }
                    if name not in names:
                        return None
                    coll = client.get_or_create_collection(
                        name=name,
                        metadata={"hnsw:space": self._metric},
                        embedding_function=None,
                    )
                else:
                    coll = client.get_or_create_collection(
                        name=name,
                        metadata={"hnsw:space": self._metric},
                        embedding_function=None,
                    )
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreUnavailableError(
                    f"获取集合 {name!r} 失败: {exc}"
                ) from exc
            self._collections[kind] = coll
            return coll

    def _remember_dim(self, kind: CollectionKind, dim: int) -> None:
        name = self.collection_name(kind)
        known = self._dims.get(kind)
        if known is None:
            self._dims[kind] = dim
            logger.debug("集合 %s 记为 %d 维", name, dim)
            return
        if self._verify_dim and known != dim:
            raise DimensionMismatchError(name, known, dim)

    def dimension(self, kind: CollectionKind) -> int | None:
        """已记录的维度；尚未写入过向量时为 ``None``。"""
        return self._dims.get(kind)

    # -- 写入 --------------------------------------------------------------

    def upsert(
        self,
        kind: CollectionKind,
        *,
        ids: Sequence[str],
        embeddings: Any,
        documents: Sequence[str] | None = None,
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        """批量 upsert，返回条数。同 id 覆盖（幂等重跑安全）。"""
        clean_ids = _check_ids(ids)
        if not clean_ids:
            return 0
        matrix = _as_matrix(embeddings, expected_rows=len(clean_ids))
        dim = len(matrix[0]) if matrix else 0
        if dim <= 0:
            raise VectorStoreError("向量维度为 0")
        self._remember_dim(kind, dim)

        docs: list[str] = list(documents) if documents is not None else [""] * len(clean_ids)
        if len(docs) != len(clean_ids):
            raise VectorStoreError(
                f"documents 数量({len(docs)}) 与 ids({len(clean_ids)}) 不一致"
            )
        metas: list[dict[str, Any]] = (
            [_flatten_metadata(m) for m in metadatas]
            if metadatas is not None
            else [{} for _ in clean_ids]
        )
        if len(metas) != len(clean_ids):
            raise VectorStoreError(
                f"metadatas 数量({len(metas)}) 与 ids({len(clean_ids)}) 不一致"
            )

        coll = self.collection(kind)
        try:
            coll.upsert(ids=clean_ids, embeddings=matrix, documents=docs, metadatas=metas)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                f"写入集合 {self.collection_name(kind)!r} 失败（{len(clean_ids)} 条）: {exc}"
            ) from exc
        logger.debug("写入 %s: %d 条 (%d 维)", self.collection_name(kind), len(clean_ids), dim)
        return len(clean_ids)

    def upsert_clips(
        self,
        *,
        ids: Sequence[str],
        embeddings: Any,
        documents: Sequence[str] | None = None,
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        """写入文本切片向量（``clip_text`` 集合）。"""
        return self.upsert(
            "clip", ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def upsert_frames(
        self,
        *,
        ids: Sequence[str],
        embeddings: Any,
        documents: Sequence[str] | None = None,
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        """写入关键帧图像向量（``frame_image`` 集合）。"""
        return self.upsert(
            "frame", ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    # -- 检索 --------------------------------------------------------------

    def _score(self, distance: float) -> float:
        d = float(distance)
        if self._metric == "cosine":
            return 1.0 - d
        if self._metric == "ip":
            return -d
        return 1.0 / (1.0 + max(0.0, d))

    def query(
        self,
        kind: CollectionKind,
        embedding: Any,
        *,
        top_k: int = 10,
        where: Mapping[str, Any] | None = None,
        where_document: Mapping[str, Any] | None = None,
    ) -> list[VectorHit]:
        """单条向量近邻检索，按相似度降序返回。"""
        vector = self._as_query_vector(embedding)
        self._remember_dim(kind, len(vector))
        n = max(1, int(top_k))
        coll = self.collection(kind)
        kwargs: dict[str, Any] = {
            "query_embeddings": [vector],
            "n_results": n,
            "include": ["metadatas", "documents", "distances"],
        }
        if where:
            kwargs["where"] = dict(where)
        if where_document:
            kwargs["where_document"] = dict(where_document)
        try:
            raw = coll.query(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                f"检索集合 {self.collection_name(kind)!r} 失败: {exc}"
            ) from exc
        return self._parse_hits(raw)

    def query_clips(
        self, embedding: Any, *, top_k: int = 10, where: Mapping[str, Any] | None = None
    ) -> list[VectorHit]:
        return self.query("clip", embedding, top_k=top_k, where=where)

    def query_frames(
        self, embedding: Any, *, top_k: int = 10, where: Mapping[str, Any] | None = None
    ) -> list[VectorHit]:
        return self.query("frame", embedding, top_k=top_k, where=where)

    def query_many(
        self,
        kind: CollectionKind,
        embeddings: Any,
        *,
        top_k: int = 10,
    ) -> list[list[VectorHit]]:
        """批量近邻检索（一次往返），返回与输入等长的结果列表。"""
        rows = _as_matrix(embeddings, expected_rows=len(embeddings))
        if not rows:
            return []
        self._remember_dim(kind, len(rows[0]))
        coll = self.collection(kind)
        try:
            raw = coll.query(
                query_embeddings=rows,
                n_results=max(1, int(top_k)),
                include=["metadatas", "documents", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"批量检索失败: {exc}") from exc
        return [self._parse_hits(raw, index=i) for i in range(len(rows))]

    def _parse_hits(self, raw: Mapping[str, Any], *, index: int = 0) -> list[VectorHit]:
        ids = (raw.get("ids") or [[]])
        dists = (raw.get("distances") or [[]])
        docs = (raw.get("documents") or [[]])
        metas = (raw.get("metadatas") or [[]])
        if index >= len(ids):
            return []
        row_ids = ids[index] or []
        row_dist = dists[index] if index < len(dists) else []
        row_docs = docs[index] if index < len(docs) else []
        row_metas = metas[index] if index < len(metas) else []
        hits: list[VectorHit] = []
        for pos, vid in enumerate(row_ids):
            dist = float(row_dist[pos]) if pos < len(row_dist) else 0.0
            hits.append(
                VectorHit(
                    id=str(vid),
                    distance=dist,
                    score=self._score(dist),
                    document=str(row_docs[pos]) if pos < len(row_docs) and row_docs[pos] else "",
                    metadata=dict(row_metas[pos]) if pos < len(row_metas) and row_metas[pos] else {},
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    @staticmethod
    def _as_query_vector(embedding: Any) -> list[float]:
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        if isinstance(embedding, (list, tuple)) and embedding and isinstance(
            embedding[0], (list, tuple)
        ):
            # 允许误传 [[...]]，取第一行
            embedding = embedding[0]
        if not isinstance(embedding, (list, tuple)):
            raise VectorStoreError("query embedding 必须是一维数组")
        vector = [float(x) for x in embedding]
        if not vector:
            raise VectorStoreError("query embedding 不能为空")
        return vector

    # -- 删除 --------------------------------------------------------------

    def delete(self, kind: CollectionKind, ids: Sequence[str]) -> int:
        clean = [str(x) for x in ids if str(x)]
        if not clean:
            return 0
        coll = self.collection(kind)
        try:
            coll.delete(ids=clean)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"删除向量失败: {exc}") from exc
        return len(clean)

    def delete_video(self, video_id: str) -> dict[str, int]:
        """按 ``metadata.video_id`` 删掉该视频的全部向量（两个集合）。"""
        if not video_id:
            return {"clip": 0, "frame": 0}
        removed: dict[str, int] = {}
        for kind in ("clip", "frame"):
            coll = self.collection(kind)
            before = self.count(kind)
            try:
                coll.delete(where={"video_id": video_id})
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"按 video_id 删除向量失败: {exc}") from exc
            removed[kind] = max(0, before - self.count(kind))
        return removed

    def reset(self, kind: CollectionKind) -> None:
        """删除集合后重建（换 embedding 模型时用）。"""
        with self._lock:
            client = self._ensure_client()
            name = self.collection_name(kind)
            self._collections.pop(kind, None)
            self._dims.pop(kind, None)
            try:
                client.delete_collection(name)
            except Exception as exc:  # noqa: BLE001 - 不存在时也可能抛
                logger.debug("删除集合 %s 时忽略: %s", name, exc)
            self.collection(kind)
        logger.warning("集合 %s 已重建（原有向量全部丢弃）", name)

    # -- 观测 --------------------------------------------------------------

    def count(self, kind: CollectionKind) -> int:
        try:
            return int(self.collection(kind).count())
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            # 集合刚被上一代清理删掉时，缓存里的 handle 会指向已不存在的集合。
            # 对调用方来说"这个代有 0 条"是准确答案，不是错误。
            if "does not exist" in str(exc).lower() or "not found" in str(exc).lower():
                self._collections.pop(kind, None)
                return 0
            raise VectorStoreError(f"统计集合条数失败: {exc}") from exc

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "persist_dir": str(self._persist_dir) if self._persist_dir else ":memory:",
            "metric": self._metric,
            "collections": {},
        }
        live = self._list_names()
        for kind in ("clip", "frame"):
            base = self.collection_name(kind)
            entry: dict[str, Any] = {"count": self._safe_count(kind), "dim": self._dims.get(kind)}
            # 带代号集合（``clip_text_<gen>``）才是真正装着向量的地方。只数
            # 基础集合会让界面显示 0 条 —— 用户以为索引没建，其实全在代里。
            generations: dict[str, int] = {}
            for name in live:
                if name == base or not name.startswith(f"{base}_"):
                    continue
                probed = self._probe_collection(name)
                if probed is None:
                    continue
                count, dim = probed
                generations[name] = count
                entry["count"] += count
                if entry["dim"] is None and dim is not None:
                    entry["dim"] = dim
            if generations:
                entry["generations"] = generations
            out["collections"][base] = entry
        return out

    def _list_names(self) -> list[str]:
        """当前后端里所有集合名（两种后端都实现了 ``list_collections``）。"""
        client = self._ensure_client()
        lister = getattr(client, "list_collections", None)
        if lister is None:
            return []
        try:
            raw = lister()
        except Exception:  # noqa: BLE001
            logger.debug("列集合失败", exc_info=True)
            return []
        names: list[str] = []
        for item in raw or []:
            if isinstance(item, str):
                names.append(item)
                continue
            name = item.get("name") if isinstance(item, Mapping) else getattr(item, "name", None)
            if name:
                names.append(str(name))
        return names

    def _safe_count(self, kind: CollectionKind) -> int:
        try:
            return self.count(kind)
        except VectorStoreError:
            return 0

    def _probe_collection(self, name: str) -> tuple[int, int | None] | None:
        """某集合的 ``(条数, 维度)``；读不到返回 ``None``。"""
        client = self._ensure_client()
        try:
            # Chroma 有 ``get_collection``；SQLite 后端只有
            # ``get_or_create_collection``，名字来自 ``list_collections``
            # 所以「创建」分支不会真的新建空集合。
            getter = getattr(client, "get_collection", None)
            if getter is not None:
                collection = getter(name)
            else:
                collection = client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": self._metric},
                    embedding_function=None,
                )
            dim: int | None = getattr(collection, "dim", None)
            if dim is None:
                prober = getattr(client, "dimension_of", None)
                if prober is not None:
                    try:
                        dim = prober(name)
                    except Exception:  # noqa: BLE001
                        dim = None
            if dim is None:
                meta = getattr(collection, "metadata", None)
                raw = meta.get("dim") if isinstance(meta, Mapping) else None
                if isinstance(raw, int):
                    dim = raw
                elif isinstance(raw, str) and raw.isdigit():
                    dim = int(raw)
            count = int(collection.count())
            return count, (dim if dim else None)
        except Exception:  # noqa: BLE001
            return None

    def health(self) -> bool:
        """能否正常读写（用于健康检查与降级判定）。"""
        try:
            self.collection("clip")
            self.collection("frame")
            return True
        except VectorStoreError:
            return False

    def close(self) -> None:
        """释放客户端引用。Chroma 无显式 close；SQLite 后端会关连接。"""
        with self._lock:
            self._collections.clear()
            client = self._client
            self._client = None
            self._closed = True
            closer = getattr(client, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception as exc:  # noqa: BLE001 - 关闭失败不影响退出
                    logger.debug("关闭向量库客户端时忽略: %s", exc)

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"VectorStore(dir={str(self._persist_dir) if self._persist_dir else ':memory:'!r}, "
            f"clip={self._clip_name!r}, frame={self._frame_name!r}, metric={self._metric!r})"
        )


def iter_batches(items: Sequence[Any], size: int = 64) -> Iterable[Sequence[Any]]:
    """把长列表切片，避免一次性把几万条向量喂给 Chroma。"""
    step = max(1, int(size))
    for start in range(0, len(items), step):
        yield items[start : start + step]
