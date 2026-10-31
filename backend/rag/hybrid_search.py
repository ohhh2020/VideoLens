"""混合检索：文本向量 / 跨模态图像 / BM25 字面匹配三通道 + RRF 融合。

该模块解决两个工程问题：

1. **多模态分数分布不一致**
   BGE-M3 的余弦相似度、SigLIP2 的图文相似度、BM25 的 TF-IDF 加权分，
   三者量纲与分布完全不同，直接相加或加权求和会让分数绝对值大的通道
   （通常是 BM25，无上界）吞掉其他通道。这里统一改用 **排名** 而非原始分：
   RRF 只依赖 ``rank``，天然对分数分布免疫。

2. **跨模态混合检索**
   图像通道的检索单元是「关键帧」，而结果消费方要的是「切片」。
   融合阶段以 ``clip_id`` 为去重键，把同一片段在三个通道里的命中合并成
   一条带完整引用证据（时间区间 + 关键帧路径）的上下文条目。

RRF 公式::

    RRF_Score(d) = Σ_c  weight_c / (k + rank_c(d))

``k`` 默认 60（原论文取值，作用是压平头部差异，避免单个通道霸榜），
``rank`` 从 1 开始。默认权重：文本 0.5 / 图像 0.3 / BM25 0.2。

典型用法::

    from backend.rag.hybrid_search import (
        HybridSearcher, HybridConfig, BM25Channel, VectorChannel,
    )

    channels = [
        VectorChannel("text", chroma_text_collection, bge_m3_embedder),
        VectorChannel("image", chroma_image_collection, siglip2_embedder,
                      id_field="keyframe_id"),
        BM25Channel.from_clips(clips),
    ]
    searcher = HybridSearcher(channels, HybridConfig(), clip_index=clip_index)
    response = searcher.search("切片长度是怎么定的？", top_k=5)
    print(searcher.build_context(response.items))

所有依赖（Chroma 集合、嵌入模型、jieba）均为**注入式**或懒加载：
缺失时对应通道被标记为不可用并跳过，其余通道照常工作，绝不因单点缺失
导致整条检索链崩溃。
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "BM25Channel",
    "BM25Index",
    "Candidate",
    "ChannelResult",
    "ChannelStatus",
    "Embedder",
    "FusedItem",
    "HybridConfig",
    "HybridSearchError",
    "HybridSearcher",
    "RetrievalChannel",
    "SearchResponse",
    "VectorChannel",
    "format_hms",
]

logger = logging.getLogger("zhiying.rag.hybrid_search")


# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #
class HybridSearchError(RuntimeError):
    """混合检索的基类异常。"""


class ConfigError(HybridSearchError, ValueError):
    """配置非法。"""


class ChannelDisabledError(HybridSearchError):
    """通道不可用（依赖缺失 / 未初始化）。"""


# --------------------------------------------------------------------------- #
# 通用小工具
# --------------------------------------------------------------------------- #
_HMS_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3})$")
_TOKEN_SPLIT_RE = re.compile(r"[\s\u3000]+")
_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def format_hms(timestamp_ms: int) -> str:
    """毫秒 → ``HH:MM:SS.mmm``（与媒体层保持同一对外格式）。"""
    total = max(0, int(timestamp_ms))
    hours, rest = divmod(total, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _parse_ms(value: Any, unit: str = "auto") -> int:
    """按已知单位把时间戳解析成毫秒；无法解析返回 -1。

    ``unit`` 取值：
    - ``"ms"``：毫秒（``*_ms`` 后缀字段）
    - ``"s"``：秒（``*_s`` / whisper 的 ``start``/``end``）
    - ``"hms"``：``HH:MM:SS.mmm`` 字符串（``*_time`` 后缀字段）
    - ``"auto"``：字符串带冒号按 hms，否则 <1e5 视为秒
    """
    if value is None:
        return -1
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return -1
        if unit == "hms" or (unit == "auto" and ":" in text):
            match = _HMS_RE.match(text)
            if match:
                h, m, s, ms = match.groups()
                return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1_000 + int(ms)
            if unit == "hms":
                return -1
        try:
            value = float(text)
        except ValueError:
            return -1
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -1
    if number < 0:
        return -1
    if unit == "ms":
        return int(round(number))
    if unit == "s":
        return int(round(number * 1_000))
    # auto：秒级数值通常 < 1e5（约 27 小时）
    return int(round(number * 1_000)) if number < 100_000 else int(round(number))


def _unit_for_key(key: str) -> str:
    if key.endswith("_ms"):
        return "ms"
    if key.endswith("_s"):
        return "s"
    if key.endswith("_time") or key.endswith("_ts"):
        return "hms"
    return "auto"


def _first_ms(payload: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        if key in payload:
            ms = _parse_ms(payload[key], _unit_for_key(key))
            if ms >= 0:
                return ms
    return -1


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    """把 list / 逗号串 / 单值统一成去重后的字符串元组。"""
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if text.startswith("["):
            try:
                import json  # noqa: PLC0415 - 仅在需要时导入

                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return tuple(
                        dict.fromkeys(str(item) for item in parsed if str(item).strip())
                    )
            except (ValueError, TypeError):
                pass
        parts = [p.strip() for p in re.split(r"[,;|]", text)]
        return tuple(dict.fromkeys(p for p in parts if p))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(dict.fromkeys(str(item) for item in value if str(item).strip()))
    return (str(value),)


# --------------------------------------------------------------------------- #
# 候选与通道结果
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Candidate:
    """单个通道返回的一条候选。

    Attributes:
        key: 通道内的唯一标识（文本/BM25 通道是 ``clip_id``，图像通道是 ``keyframe_id``）。
        rank: 通道内排名，**从 1 开始**（RRF 直接消费该值）。
        score: 通道原始分（余弦相似度 / BM25 分），仅用于诊断展示，不参与融合。
        payload: 元数据；必须尽量携带 ``clip_id`` 与 ``video_id``，
            以便融合阶段把图像候选折叠回所属切片。
    """

    key: str
    rank: int
    score: float = 0.0
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"rank 必须 ≥ 1，收到 {self.rank}")


@dataclass(frozen=True, slots=True)
class ChannelResult:
    """单个通道的执行结果（含失败信息，便于逐级降级诊断）。"""

    channel: str
    candidates: tuple[Candidate, ...] = ()
    ok: bool = True
    error: str = ""
    elapsed_ms: int = 0
    timed_out: bool = False

    @property
    def count(self) -> int:
        return len(self.candidates)


@dataclass(frozen=True, slots=True)
class ChannelStatus:
    """通道状态的对外快照。"""

    channel: str
    ok: bool
    count: int
    elapsed_ms: int
    error: str = ""
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "channel": self.channel,
            "ok": self.ok,
            "count": self.count,
            "elapsed_ms": self.elapsed_ms,
        }
        if self.error:
            data["error"] = self.error
        if self.timed_out:
            data["timed_out"] = True
        return data


# --------------------------------------------------------------------------- #
# 通道协议
# --------------------------------------------------------------------------- #
@runtime_checkable
class RetrievalChannel(Protocol):
    """检索通道协议：任何实现 ``name`` + ``search`` 的对象都能接入融合器。"""

    name: str

    def search(
        self,
        query: str,
        top_k: int,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[Candidate]:
        """返回 top_k 个候选，``rank`` 必须从 1 连续递增。"""
        ...

    def available(self) -> bool:
        """通道当前是否可用（依赖是否就绪）。"""
        ...


@runtime_checkable
class Embedder(Protocol):
    """嵌入器协议：BGE-M3 / SigLIP2 的封装都收敛到这两个方法。"""

    def embed_query(self, text: str) -> Sequence[float]: ...

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class CallableEmbedder:
    """把普通函数包装成 :class:`Embedder`，便于测试与轻量接入。"""

    def __init__(
        self,
        query_fn: Callable[[str], Sequence[float]],
        texts_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
    ) -> None:
        self._query_fn = query_fn
        self._texts_fn = texts_fn or (lambda items: [query_fn(item) for item in items])

    def embed_query(self, text: str) -> Sequence[float]:
        return self._query_fn(text)

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self._texts_fn(texts)


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
DEFAULT_WEIGHTS: dict[str, float] = {"text": 0.5, "image": 0.3, "bm25": 0.2}


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """混合检索配置。

    Attributes:
        k: RRF 平滑常数，原论文取 60；越小头部越集中，越大越平均。
        weights: 通道权重，未列出的通道权重按 ``default_weight`` 处理。
        default_weight: 未配置权重通道的兜底权重。
        top_k: 最终返回条数。
        fetch_k: 每通道召回的候选数；应显著大于 ``top_k``，
            RRF 需要更长的候选列表才有融合价值（缺省 ``max(top_k*4, 20)``）。
        timeout_ms: 单通道超时；超时通道被丢弃但不影响其他通道。
        min_score: 归一化得分下限，低于该值的条目被过滤。
        max_context_items: 组装上下文时最多包含的条数。
        context_max_chars: 上下文总字符预算。
        parallel: 是否并发执行各通道。
    """

    k: int = 60
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    default_weight: float = 0.1
    top_k: int = 8
    fetch_k: int = 0  # 0 = 自动
    timeout_ms: int = 8_000
    min_score: float = 0.0
    max_context_items: int = 8
    context_max_chars: int = 6_000
    parallel: bool = True

    def validate(self) -> None:
        if self.k < 1:
            raise ConfigError("k 必须 ≥ 1")
        if self.top_k < 1:
            raise ConfigError("top_k 必须 ≥ 1")
        if self.fetch_k < 0:
            raise ConfigError("fetch_k 不能为负")
        if self.timeout_ms <= 0:
            raise ConfigError("timeout_ms 必须为正")
        if not (0.0 <= self.min_score <= 1.0):
            raise ConfigError("min_score 必须落在 [0, 1]")
        if self.max_context_items < 1:
            raise ConfigError("max_context_items 必须 ≥ 1")
        if any(weight < 0 for weight in self.weights.values()):
            raise ConfigError("通道权重不能为负")

    @property
    def effective_fetch_k(self) -> int:
        return self.fetch_k if self.fetch_k > 0 else max(self.top_k * 4, 20)

    def weight_of(self, channel: str) -> float:
        return float(self.weights.get(channel, self.default_weight))


# --------------------------------------------------------------------------- #
# 向量通道（Chroma）
# --------------------------------------------------------------------------- #
class VectorChannel:
    """基于 Chroma 集合 + 嵌入器的向量检索通道。

    同一个类同时服务文本通道与图像通道：差别只在
    ``id_field``（``clip_id`` / ``keyframe_id``）与所用嵌入器。

    集合查询用 ``query_embeddings`` 而非 ``query_texts``，因为嵌入必须由
    项目自己的模型层产生（ModelManager 管理的 BGE-M3 / SigLIP2），
    不能依赖 Chroma 内置的默认嵌入函数。
    """

    def __init__(
        self,
        name: str,
        collection: Any,
        embedder: Embedder | None,
        *,
        id_field: str = "clip_id",
        include: Sequence[str] = ("metadatas", "distances"),
        collection_ready: bool = True,
        logger_: logging.Logger | None = None,
    ) -> None:
        self.name = name
        self._collection = collection
        self._embedder = embedder
        self._id_field = id_field
        self._include = list(include)
        self._ready = collection_ready
        self._log = logger_ or logger

    def available(self) -> bool:
        return (
            self._ready and self._collection is not None and self._embedder is not None
        )

    def search(
        self,
        query: str,
        top_k: int,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[Candidate]:
        if not self.available():
            raise ChannelDisabledError(f"通道 {self.name} 不可用（集合或嵌入器缺失）")
        if top_k <= 0:
            return ()
        query = (query or "").strip()
        if not query:
            return ()

        vector = self._embedder.embed_query(query)  # type: ignore[union-attr]
        kwargs: dict[str, Any] = {
            "query_embeddings": [list(vector)],
            "n_results": int(top_k),
            "include": self._include,
        }
        if filters:
            kwargs["where"] = dict(filters)

        raw = self._collection.query(**kwargs)
        ids = (raw.get("ids") or [[]])[0] or []
        metas = (raw.get("metadatas") or [[]])[0] or []
        dists = (raw.get("distances") or [[]])[0] or []

        candidates: list[Candidate] = []
        for index, doc_id in enumerate(ids):
            meta: Mapping[str, Any] = {}
            if index < len(metas) and isinstance(metas[index], Mapping):
                meta = metas[index]
            distance = (
                float(dists[index])
                if index < len(dists) and dists[index] is not None
                else None
            )
            # Chroma 返回的是距离（cosine 空间下 1 - similarity），转成可读相似度
            score = 0.0 if distance is None else max(0.0, 1.0 - distance)
            key = str(meta.get(self._id_field) or doc_id)
            candidates.append(
                Candidate(
                    key=key,
                    rank=index + 1,
                    score=score,
                    payload=dict(meta) | {"_doc_id": str(doc_id)},
                )
            )
        self._log.debug(
            "通道 %s 命中 %d 条（top_k=%d）", self.name, len(candidates), top_k
        )
        return candidates

    @classmethod
    def from_chroma(
        cls,
        name: str,
        client: Any,
        collection_name: str,
        embedder: Embedder | None,
        *,
        id_field: str = "clip_id",
        create: bool = False,
        metadata: Mapping[str, Any] | None = None,
        logger_: logging.Logger | None = None,
    ) -> "VectorChannel":
        """从 Chroma 客户端构造通道；集合缺失且 ``create=False`` 时通道不可用。"""
        collection = None
        ready = False
        try:
            names = {
                item if isinstance(item, str) else getattr(item, "name", "")
                for item in client.list_collections()
            }
            if collection_name in names:
                collection = client.get_collection(collection_name)
                ready = True
            elif create:
                collection = client.create_collection(
                    collection_name,
                    metadata=dict(metadata or {"hnsw:space": "cosine"}),
                )
                ready = True
        except Exception as exc:  # noqa: BLE001 - 依赖/版本差异一律降级
            (logger_ or logger).warning("通道 %s 初始化 Chroma 集合失败: %s", name, exc)
        return cls(
            name,
            collection,
            embedder,
            id_field=id_field,
            collection_ready=ready,
            logger_=logger_,
        )


# --------------------------------------------------------------------------- #
# BM25 通道
# --------------------------------------------------------------------------- #
class BM25Index:
    """纯 Python BM25（Okapi）实现，避免额外依赖。

    中文分词优先用 jieba；jieba 缺失时退回「字符 unigram + bigram」，
    对中文专有名词仍有一定召回能力。英文统一小写并按空白切分。

    公式::

        idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))
        score  = Σ_t idf(t) * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl))
    """

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: Iterable[str] | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        if k1 <= 0:
            raise ConfigError("BM25 k1 必须为正")
        if not (0.0 <= b <= 1.0):
            raise ConfigError("BM25 b 必须落在 [0, 1]")
        self._k1 = float(k1)
        self._b = float(b)
        self._log = logger_ or logger
        self._stopwords = set(stopwords or _DEFAULT_STOPWORDS)
        self._lock = threading.RLock()
        self._doc_ids: list[str] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_len: list[int] = []
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._jieba: Any | None = None
        self._jieba_checked = False

    # -- 分词 --
    @property
    def tokenizer_backend(self) -> str:
        return "jieba" if self._get_jieba() is not None else "char-ngram"

    def _get_jieba(self) -> Any | None:
        if not self._jieba_checked:
            self._jieba_checked = True
            try:
                import jieba  # noqa: PLC0415 - 可选依赖

                jieba.setLogLevel(logging.WARNING)
                self._jieba = jieba
            except Exception as exc:  # noqa: BLE001 - 缺依赖走字符兜底
                self._log.info("jieba 不可用（%s），BM25 退回字符 n-gram 分词", exc)
                self._jieba = None
        return self._jieba

    def tokenize(self, text: str) -> list[str]:
        """分词 → 去停用词 → 去纯标点。"""
        raw = (text or "").strip()
        if not raw:
            return []
        jieba = self._get_jieba()
        if jieba is not None:
            tokens = [token.strip().lower() for token in jieba.lcut(raw)]
        else:
            lowered = raw.lower()
            chars = [ch for ch in lowered if not _PUNCT_RE.match(ch)]
            tokens = list(chars)
            tokens += [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
        return [
            token
            for token in tokens
            if token
            and token not in self._stopwords
            and not _PUNCT_RE.match(token)
            and not _TOKEN_SPLIT_RE.match(token)
        ]

    # -- 建索引 --
    def add(self, doc_id: str, text: str) -> None:
        with self._lock:
            existing = self._doc_ids.index(doc_id) if doc_id in self._doc_ids else -1
            tokens = self.tokenize(text)
            if existing >= 0:
                for token in set(self._doc_tokens[existing]):
                    self._df[token] = max(0, self._df.get(token, 1) - 1)
                self._doc_tokens[existing] = tokens
                self._doc_len[existing] = len(tokens)
            else:
                self._doc_ids.append(doc_id)
                self._doc_tokens.append(tokens)
                self._doc_len.append(len(tokens))
            for token in set(tokens):
                self._df[token] = self._df.get(token, 0) + 1
            self._recompute_avgdl()

    def add_many(self, docs: Iterable[tuple[str, str]]) -> int:
        count = 0
        for doc_id, text in docs:
            self.add(doc_id, text)
            count += 1
        return count

    def _recompute_avgdl(self) -> None:
        self._avgdl = (
            (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0
        )

    @property
    def size(self) -> int:
        return len(self._doc_ids)

    # -- 查询 --
    def idf(self, token: str) -> float:
        n = len(self._doc_ids)
        if n == 0:
            return 0.0
        df = self._df.get(token, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(
        self, query: str, top_k: int = 20
    ) -> list[tuple[str, float, tuple[str, ...]]]:
        """返回 ``[(doc_id, score, matched_terms), ...]``，按分数降序。"""
        if top_k <= 0:
            return []
        tokens = self.tokenize(query)
        if not tokens or not self._doc_ids:
            return []
        avgdl = self._avgdl or 1.0
        scores: dict[int, float] = {}
        matched: dict[int, set[str]] = {}
        with self._lock:
            idf_map = {token: self.idf(token) for token in set(tokens)}
            for token in tokens:
                idf = idf_map.get(token, 0.0)
                if idf <= 0.0:
                    continue
                for index, doc_tokens in enumerate(self._doc_tokens):
                    tf = doc_tokens.count(token)
                    if not tf:
                        continue
                    dl = self._doc_len[index] or 0
                    denom = tf + self._k1 * (1.0 - self._b + self._b * dl / avgdl)
                    scores[index] = (
                        scores.get(index, 0.0) + idf * tf * (self._k1 + 1.0) / denom
                    )
                    matched.setdefault(index, set()).add(token)
            ranked = sorted(
                scores.items(),
                key=lambda item: (-item[1], self._doc_ids[item[0]]),
            )[:top_k]
            return [
                (self._doc_ids[index], score, tuple(sorted(matched.get(index, ()))))
                for index, score in ranked
            ]

    def clear(self) -> None:
        with self._lock:
            self._doc_ids.clear()
            self._doc_tokens.clear()
            self._doc_len.clear()
            self._df.clear()
            self._avgdl = 0.0


_DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    {
        "的",
        "了",
        "和",
        "是",
        "在",
        "我",
        "有",
        "就",
        "不",
        "人",
        "都",
        "一",
        "一个",
        "上",
        "也",
        "很",
        "到",
        "说",
        "要",
        "去",
        "你",
        "会",
        "着",
        "没有",
        "看",
        "好",
        "自己",
        "这",
        "那",
        "什么",
        "怎么",
        "如何",
        "为什么",
        "然后",
        "还是",
        "但是",
        "a",
        "an",
        "the",
        "of",
        "to",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "in",
        "on",
        "at",
        "for",
        "with",
        "as",
        "by",
        "it",
        "this",
        "that",
        "be",
        "from",
    }
)


class BM25Channel:
    """BM25 字面匹配通道（专有名词 / 编号 / 人名保底召回）。"""

    def __init__(
        self,
        name: str = "bm25",
        index: BM25Index | None = None,
        *,
        index_ready: bool = True,
        logger_: logging.Logger | None = None,
    ) -> None:
        self.name = name
        self._index = index if index is not None else BM25Index(logger_=logger_)
        self._ready = index_ready
        self._log = logger_ or logger
        self._payloads: dict[str, Mapping[str, Any]] = {}

    @property
    def index(self) -> BM25Index:
        return self._index

    def available(self) -> bool:
        return self._ready and self._index.size > 0

    def add_clip(
        self, clip_id: str, text: str, payload: Mapping[str, Any] | None = None
    ) -> None:
        """写入一条切片；payload 会随候选一起返回，供融合阶段补齐元数据。"""
        self._payloads[clip_id] = dict(payload or {})
        self._index.add(clip_id, text)

    def add_clips(self, clips: Iterable[Any]) -> int:
        """批量写入切片。

        接受三类输入：:class:`backend.segmentation.chunker.Clip`（或任意带同名字段的对象）、
        ``Mapping``，以及 ``(clip_id, text)`` 二元组。
        """
        count = 0
        for clip in clips:
            # 二元组形式：最轻量的批量灌库方式
            if (
                isinstance(clip, (tuple, list))
                and len(clip) >= 2
                and isinstance(clip[0], str)
                and isinstance(clip[1], str)
            ):
                self.add_clip(clip[0], clip[1])
                count += 1
                continue
            clip_id = str(_get(clip, "clip_id", "") or "")
            text = str(_get(clip, "core_text", "") or _get(clip, "text", ""))
            if not clip_id:
                continue
            payload = {
                "clip_id": clip_id,
                "video_id": str(_get(clip, "video_id", "") or ""),
                "start_ms": _get(clip, "core_start_ms", 0) or 0,
                "end_ms": _get(clip, "core_end_ms", 0) or 0,
                "text": str(_get(clip, "text", "") or text),
                "keyframe_ids": list(_get(clip, "keyframe_ids", ()) or ()),
            }
            self.add_clip(clip_id, text, payload)
            count += 1
        self._log.debug(
            "BM25 通道索引 %d 条切片（分词器=%s）", count, self._index.tokenizer_backend
        )
        return count

    @classmethod
    def from_clips(
        cls,
        clips: Iterable[Any],
        *,
        name: str = "bm25",
        logger_: logging.Logger | None = None,
    ) -> "BM25Channel":
        channel = cls(name, logger_=logger_)
        channel.add_clips(clips)
        return channel

    def search(
        self,
        query: str,
        top_k: int,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[Candidate]:
        if not self.available():
            raise ChannelDisabledError(f"通道 {self.name} 不可用（索引为空）")
        hits = self._index.search(query, top_k=top_k)
        payloads: Mapping[str, Mapping[str, Any]] = getattr(self, "_payloads", {})
        candidates = [
            Candidate(
                key=doc_id,
                rank=rank,
                score=score,
                payload=dict(payloads.get(doc_id, {}))
                | {"_matched_terms": list(terms)},
            )
            for rank, (doc_id, score, terms) in enumerate(hits, start=1)
        ]
        if filters:
            candidates = [
                cand
                for cand in candidates
                if all(cand.payload.get(key) == value for key, value in filters.items())
            ]
        return candidates


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """同时兼容对象属性与 Mapping。"""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


# --------------------------------------------------------------------------- #
# 融合结果
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FusedItem:
    """RRF 融合后的一条上下文条目（可直接喂给 LLM 或返回给前端）。"""

    clip_id: str
    video_id: str
    start_ms: int
    end_ms: int
    text: str
    score: float
    normalized_score: float
    rank: int
    channel_ranks: Mapping[str, int] = field(default_factory=dict)
    channel_scores: Mapping[str, float] = field(default_factory=dict)
    keyframe_ids: tuple[str, ...] = ()
    keyframe_paths: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def start_time(self) -> str:
        return format_hms(self.start_ms)

    @property
    def end_time(self) -> str:
        return format_hms(self.end_ms)

    @property
    def time_range(self) -> str:
        return f"{self.start_time}–{self.end_time}"

    def evidence(self) -> list[str]:
        """引用证据链：切片区间 + 关键帧文件。"""
        items = [
            f"clip={self.clip_id}",
            f"video={self.video_id}",
            f"range={self.time_range}",
        ]
        items.extend(f"keyframe={path}" for path in self.keyframe_paths)
        if not self.keyframe_paths:
            items.extend(f"keyframe_id={fid}" for fid in self.keyframe_ids)
        return items

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "video_id": self.video_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "text": self.text,
            "score": round(self.score, 6),
            "normalized_score": round(self.normalized_score, 6),
            "rank": self.rank,
            "channel_ranks": dict(self.channel_ranks),
            "channel_scores": {k: round(v, 6) for k, v in self.channel_scores.items()},
            "keyframe_ids": list(self.keyframe_ids),
            "keyframe_paths": list(self.keyframe_paths),
            "matched_terms": list(self.matched_terms),
        }


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """一次检索的完整结果。"""

    query: str
    items: tuple[FusedItem, ...]
    channels: tuple[ChannelStatus, ...]
    elapsed_ms: int
    total_candidates: int = 0
    degraded: tuple[str, ...] = ()
    filters: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return any(status.ok for status in self.channels) and bool(self.items)

    def to_dict(self, *, include_payload: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "query": self.query,
            "elapsed_ms": self.elapsed_ms,
            "total_candidates": self.total_candidates,
            "items": [item.to_dict() for item in self.items],
            "channels": [status.to_dict() for status in self.channels],
        }
        if self.degraded:
            data["degraded"] = list(self.degraded)
        if self.filters:
            data["filters"] = dict(self.filters)
        if include_payload:
            data["payloads"] = [dict(item.payload) for item in self.items]
        return data


# --------------------------------------------------------------------------- #
# 融合器主体
# --------------------------------------------------------------------------- #
class HybridSearcher:
    """三通道混合检索 + RRF 融合。

    Args:
        channels: 通道列表；同名通道只保留第一个。
        config: :class:`HybridConfig`。
        clip_index: 可选 ``clip_id → Clip`` 映射（或可调用对象），
            用于在 Chroma 元数据不完整时补齐文本 / 关键帧 / 时间区间。
        clip_resolver: 可选 ``clip_id → Clip`` 回调，优先于 ``clip_index``。
        keyframe_resolver: 可选 ``keyframe_id → 本地文件路径`` 回调，
            把证据链从 ID 提升为可直接打开的路径。
    """

    def __init__(
        self,
        channels: Sequence[RetrievalChannel],
        config: HybridConfig | None = None,
        *,
        clip_index: Mapping[str, Any] | None = None,
        clip_resolver: Callable[[str], Any | None] | None = None,
        keyframe_resolver: Callable[[str], str | None] | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        self._cfg = config or HybridConfig()
        self._cfg.validate()
        self._log = logger_ or logger

        seen: dict[str, RetrievalChannel] = {}
        for channel in channels:
            name = str(getattr(channel, "name", "") or "").strip()
            if not name:
                raise ConfigError(f"通道 {channel!r} 缺少 name")
            if name in seen:
                self._log.warning("重复的通道名 %r，已忽略后一个", name)
                continue
            seen[name] = channel
        if not seen:
            raise ConfigError("至少需要一个检索通道")
        self._channels: tuple[RetrievalChannel, ...] = tuple(seen.values())

        self._clip_index = dict(clip_index or {})
        self._clip_resolver = clip_resolver
        self._keyframe_resolver = keyframe_resolver
        self._pool = (
            ThreadPoolExecutor(
                max_workers=min(4, len(self._channels)), thread_name_prefix="hs"
            )
            if self._cfg.parallel
            else None
        )

    # -- 生命周期 --
    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def __enter__(self) -> "HybridSearcher":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @property
    def config(self) -> HybridConfig:
        return self._cfg

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self._channels)

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        filters: Mapping[str, Any] | None = None,
        min_score: float | None = None,
    ) -> SearchResponse:
        """执行混合检索。

        Args:
            query: 自然语言查询，保留原始问题及检索补充。
            top_k: 返回条数，缺省用配置值。
            filters: 透传给向量通道的元数据过滤（如 ``{"video_id": "v1"}``）。
            min_score: 覆盖配置里的归一化分数下限。
        """
        started = time.monotonic()
        text = (query or "").strip()
        if not text:
            raise HybridSearchError("查询文本不能为空")

        limit = self._cfg.top_k if top_k is None else int(top_k)
        if limit < 1:
            raise HybridSearchError(f"top_k 必须 ≥ 1，收到 {limit}")
        fetch_k = max(self._cfg.effective_fetch_k, limit)
        threshold = self._cfg.min_score if min_score is None else float(min_score)
        degraded: list[str] = []

        # 1) 过滤不可用通道（依赖缺失不视为故障，只是不参与）
        active: list[RetrievalChannel] = []
        statuses: list[ChannelStatus] = []
        for channel in self._channels:
            try:
                ready = bool(channel.available())
            except Exception as exc:  # noqa: BLE001 - available() 本身也不可信
                ready = False
                statuses.append(
                    ChannelStatus(channel.name, False, 0, 0, f"available() 异常: {exc}")
                )
                degraded.append(f"通道 {channel.name} 状态检测异常: {exc}")
                continue
            if ready:
                active.append(channel)
            else:
                statuses.append(
                    ChannelStatus(
                        channel.name, False, 0, 0, "通道未就绪（依赖或索引缺失）"
                    )
                )
                degraded.append(f"通道 {channel.name} 未就绪，已跳过")

        if not active:
            self._log.error("所有检索通道均不可用，query=%r", text)
            return SearchResponse(
                query=text,
                items=(),
                channels=tuple(statuses),
                elapsed_ms=int((time.monotonic() - started) * 1_000),
                degraded=tuple(degraded + ["没有可用通道，返回空结果"]),
                filters=dict(filters or {}),
            )

        # 2) 执行（并发或串行）
        results: list[ChannelResult] = self._run_channels(
            active, text, fetch_k, filters
        )

        total_candidates = sum(result.count for result in results)
        for result in results:
            if not result.ok:
                degraded.append(f"通道 {result.channel} 失败: {result.error}")
            elif result.timed_out:
                degraded.append(
                    f"通道 {result.channel} 超时（>{self._cfg.timeout_ms}ms），已丢弃"
                )

        # 3) RRF 融合 + 去重
        items = self.fuse(results)

        # 4) 阈值过滤 + 截断
        if threshold > 0:
            items = [item for item in items if item.normalized_score >= threshold]
        items = items[:limit]
        items = [replace(item, rank=index + 1) for index, item in enumerate(items)]

        elapsed_ms = int((time.monotonic() - started) * 1_000)

        # 通道状态按 self._channels 原顺序输出，未就绪通道也要留痕
        status_by_name: dict[str, ChannelStatus] = {s.channel: s for s in statuses}
        for result in results:
            status_by_name[result.channel] = ChannelStatus(
                channel=result.channel,
                ok=result.ok,
                count=result.count,
                elapsed_ms=result.elapsed_ms,
                error=result.error,
                timed_out=result.timed_out,
            )
        ordered_channels = tuple(
            status_by_name[channel.name]
            for channel in self._channels
            if channel.name in status_by_name
        )

        response = SearchResponse(
            query=text,
            items=tuple(items),
            channels=ordered_channels,
            elapsed_ms=elapsed_ms,
            total_candidates=total_candidates,
            degraded=tuple(degraded),
            filters=dict(filters or {}),
        )
        self._log.info(
            "混合检索完成 query=%r 通道=%d 候选=%d 命中=%d 耗时=%dms",
            text[:40],
            len(results),
            total_candidates,
            len(items),
            elapsed_ms,
        )
        return response

    def _run_channels(
        self,
        channels: Sequence[RetrievalChannel],
        query: str,
        fetch_k: int,
        filters: Mapping[str, Any] | None,
    ) -> list[ChannelResult]:
        if not self._cfg.parallel or self._pool is None or len(channels) == 1:
            return [
                self._call_channel(channel, query, fetch_k, filters)
                for channel in channels
            ]

        futures: list[tuple[RetrievalChannel, Future[ChannelResult]]] = [
            (
                channel,
                self._pool.submit(self._call_channel, channel, query, fetch_k, filters),
            )
            for channel in channels
        ]
        deadline = self._cfg.timeout_ms / 1_000.0
        results: list[ChannelResult] = []
        for channel, future in futures:
            try:
                results.append(future.result(timeout=deadline))
            except TimeoutError:
                future.cancel()
                results.append(
                    ChannelResult(
                        channel=channel.name,
                        ok=False,
                        error=f"超时（>{self._cfg.timeout_ms}ms）",
                        timed_out=True,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - 兜底，避免单通道炸掉整体
                results.append(
                    ChannelResult(
                        channel=channel.name,
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return results

    def _call_channel(
        self,
        channel: RetrievalChannel,
        query: str,
        fetch_k: int,
        filters: Mapping[str, Any] | None,
    ) -> ChannelResult:
        started = time.monotonic()
        try:
            raw = channel.search(query, fetch_k, filters=filters) or ()
        except ChannelDisabledError as exc:
            return ChannelResult(
                channel.name, (), False, str(exc), _elapsed_ms(started)
            )
        except Exception as exc:  # noqa: BLE001 - 逐级降级：单通道失败不影响其他通道
            self._log.warning(
                "通道 %s 检索失败: %s: %s", channel.name, type(exc).__name__, exc
            )
            return ChannelResult(
                channel.name,
                (),
                False,
                f"{type(exc).__name__}: {exc}",
                _elapsed_ms(started),
            )

        candidates: list[Candidate] = []
        for index, item in enumerate(raw, start=1):
            if not isinstance(item, Candidate):
                continue
            # 规范 rank：通道实现可能返回稀疏或不连续的 rank，统一按返回顺序重排
            candidates.append(item if item.rank == index else replace(item, rank=index))
        return ChannelResult(
            channel.name, tuple(candidates), True, "", _elapsed_ms(started)
        )

    # ------------------------------------------------------------------ #
    # RRF 融合
    # ------------------------------------------------------------------ #
    def fuse(self, results: Sequence[ChannelResult]) -> list[FusedItem]:
        """RRF 融合 + 按 ``clip_id`` 去重。

        同一片段在多个通道命中时得分累加；关键帧证据取并集，
        排序稳定性用 ``clip_id`` 兜底（避免同分顺序随机）。
        """
        cfg = self._cfg
        k = cfg.k

        # 每个通道内，同一 clip 只保留最好（rank 最小）的候选
        best_per_channel: dict[str, dict[str, Candidate]] = {}
        keyframes: dict[str, set[str]] = {}
        singles: dict[str, set[str]] = {}
        paths: dict[str, set[str]] = {}
        terms: dict[str, set[str]] = {}
        payloads: dict[str, dict[str, Any]] = {}

        for result in results:
            if not result.ok:
                continue
            slot = best_per_channel.setdefault(result.channel, {})
            for cand in result.candidates:
                key = self._dedupe_key(cand)
                if not key:
                    continue
                previous = slot.get(key)
                if previous is None or cand.rank < previous.rank:
                    slot[key] = cand
                self._collect_evidence(key, cand, keyframes, singles, paths, terms)
                if key not in payloads:
                    payloads[key] = dict(cand.payload)

        scores: dict[str, float] = {}
        channel_ranks: dict[str, dict[str, int]] = {}
        channel_scores: dict[str, dict[str, float]] = {}
        for channel, slot in best_per_channel.items():
            weight = cfg.weight_of(channel)
            if weight <= 0:
                continue
            for key, cand in slot.items():
                scores[key] = scores.get(key, 0.0) + weight / (k + cand.rank)
                channel_ranks.setdefault(key, {})[channel] = cand.rank
                channel_scores.setdefault(key, {})[channel] = cand.score

        if not scores:
            return []

        max_possible = sum(cfg.weight_of(channel.name) for channel in self._channels)
        ceiling = max_possible / (k + 1) if max_possible > 0 else 1.0

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        items: list[FusedItem] = []
        for index, (key, score) in enumerate(ordered, start=1):
            payload = payloads.get(key, {})
            item = self._build_item(
                key=key,
                score=score,
                normalized=min(1.0, score / ceiling) if ceiling > 0 else 0.0,
                rank=index,
                payload=payload,
                channel_ranks=channel_ranks.get(key, {}),
                channel_scores=channel_scores.get(key, {}),
                keyframe_ids=tuple(sorted(keyframes.get(key, ()))),
                single_frames=tuple(sorted(singles.get(key, ()))),
                keyframe_paths=tuple(sorted(paths.get(key, ()))),
                matched_terms=tuple(sorted(terms.get(key, ()))),
            )
            items.append(item)
        return items

    def _dedupe_key(self, cand: Candidate) -> str:
        """去重键：优先 ``clip_id``，让图像通道的关键帧候选折叠回所属切片。"""
        clip_id = str(cand.payload.get("clip_id") or "").strip()
        return clip_id or str(cand.key or "").strip()

    def _collect_evidence(
        self,
        key: str,
        cand: Candidate,
        keyframes: dict[str, set[str]],
        singles: dict[str, set[str]],
        paths: dict[str, set[str]],
        terms: dict[str, set[str]],
    ) -> None:
        payload = cand.payload
        # 文档随附的关键帧列表（可被 clip 索引覆盖）
        for fid in _as_str_tuple(payload.get("keyframe_ids")):
            keyframes.setdefault(key, set()).add(fid)
        # 图像通道单帧命中：逐条累积，是最精确的证据
        for fid in _as_str_tuple(payload.get("keyframe_id")):
            singles.setdefault(key, set()).add(fid)
        for fid in _as_str_tuple(payload.get("keyframe_path")):
            paths.setdefault(key, set()).add(fid)
        for term in _as_str_tuple(payload.get("_matched_terms")):
            terms.setdefault(key, set()).add(term)

    def _build_item(
        self,
        *,
        key: str,
        score: float,
        normalized: float,
        rank: int,
        payload: Mapping[str, Any],
        channel_ranks: Mapping[str, int],
        channel_scores: Mapping[str, float],
        keyframe_ids: tuple[str, ...],
        single_frames: tuple[str, ...],
        keyframe_paths: tuple[str, ...],
        matched_terms: tuple[str, ...],
    ) -> FusedItem:
        clip = self._lookup_clip(key)
        meta: dict[str, Any] = dict(payload)
        clip_kf: tuple[str, ...] = ()
        if clip is not None:
            # clip 索引是权威源：非空字段覆盖向量库里的旧元数据
            authoritative = {
                "video_id": _get(clip, "video_id", ""),
                "start_ms": _get(clip, "core_start_ms", None),
                "end_ms": _get(clip, "core_end_ms", None),
                "text": _get(clip, "text", "") or _get(clip, "core_text", ""),
                "keyframe_ids": _get(clip, "keyframe_ids", ()),
            }
            clip_kf = _as_str_tuple(authoritative["keyframe_ids"])
            for field_name, value in authoritative.items():
                if value not in (None, "", (), []):
                    meta[field_name] = value

        video_id = str(meta.get("video_id") or "")
        start_ms = _first_ms(
            meta, "start_ms", "core_start_ms", "start", "start_s", "start_time"
        )
        end_ms = _first_ms(meta, "end_ms", "core_end_ms", "end", "end_s", "end_time")
        if end_ms < 0:
            end_ms = max(start_ms, 0)
        if start_ms < 0:
            start_ms = 0
        start_ms, end_ms = min(start_ms, end_ms), max(start_ms, end_ms)

        # 证据优先级：clip 索引（权威）> 单帧命中 > 文档随附的关键帧列表
        merged_ids = tuple(dict.fromkeys(clip_kf or keyframe_ids)) + tuple(
            fid
            for fid in dict.fromkeys(single_frames)
            if fid not in set(clip_kf or keyframe_ids)
        )
        merged_ids = tuple(dict.fromkeys(merged_ids))
        merged_paths = list(keyframe_paths)
        if self._keyframe_resolver is not None:
            for fid in merged_ids:
                try:
                    resolved = self._keyframe_resolver(fid)
                except Exception:  # noqa: BLE001 - 回调异常不影响结果
                    resolved = None
                if resolved and str(resolved) not in merged_paths:
                    merged_paths.append(str(resolved))

        if not video_id:
            video_id = self._infer_video_id(key, meta)

        return FusedItem(
            clip_id=key,
            video_id=video_id,
            start_ms=start_ms,
            end_ms=end_ms,
            text=str(meta.get("text") or meta.get("core_text") or ""),
            score=score,
            normalized_score=normalized,
            rank=rank,
            channel_ranks=dict(channel_ranks),
            channel_scores=dict(channel_scores),
            keyframe_ids=merged_ids,
            keyframe_paths=tuple(dict.fromkeys(merged_paths)),
            matched_terms=matched_terms,
            payload=dict(payload),
        )

    def _lookup_clip(self, clip_id: str) -> Any | None:
        if self._clip_resolver is not None:
            try:
                found = self._clip_resolver(clip_id)
            except Exception:  # noqa: BLE001
                found = None
            if found is not None:
                return found
        return self._clip_index.get(clip_id)

    def _infer_video_id(self, key: str, meta: Mapping[str, Any]) -> str:
        """从 ``video_id-c0001`` 之类的 ID 反推 video_id（最后手段）。"""
        if "-c" in key:
            return key.rsplit("-c", 1)[0]
        return str(meta.get("_doc_id") or "")

    # ------------------------------------------------------------------ #
    # 上下文组装
    # ------------------------------------------------------------------ #
    def build_context(
        self,
        items: Sequence[FusedItem],
        *,
        max_items: int | None = None,
        max_chars: int | None = None,
        include_scores: bool = True,
        instruction: bool = True,
    ) -> str:
        """把融合结果拼成带编号、带溯源信息的上下文块。

        每条都携带 ``video_id / clip_id / 时间区间 / 关键帧路径``，
        满足"任何对外结论都必须可回溯"的硬约束。
        """
        limit = int(max_items or self._cfg.max_context_items)
        budget = int(max_chars or self._cfg.context_max_chars)
        lines: list[str] = []
        used = 0
        for item in items[:limit]:
            header = f"[{item.rank}] {item.video_id} {item.time_range}"
            if include_scores:
                header += f" (score={item.normalized_score:.3f}, clip={item.clip_id})"
            else:
                header += f" (clip={item.clip_id})"
            body = item.text.strip() or "（该区间没有识别到语音）"
            block = header + "\n" + body
            if item.keyframe_paths:
                block += "\n关键帧: " + ", ".join(item.keyframe_paths)
            elif item.keyframe_ids:
                block += "\n关键帧ID: " + ", ".join(item.keyframe_ids)
            if used + len(block) > budget:
                lines.append("…（上下文预算已满，其余结果被截断）")
                break
            lines.append(block)
            used += len(block) + 2
        if instruction and lines:
            lines.append(
                "回答要求：只使用以上片段作答；每条结论后必须标注 [编号]，"
                "编号必须对应视频、切片与时间区间。"
            )
        return "\n\n".join(lines)

    def explain(self, response: SearchResponse) -> str:
        """人类可读的降级/排名诊断（调试与评测用）。"""
        parts = [f"query={response.query!r} elapsed={response.elapsed_ms}ms"]
        for status in response.channels:
            flag = "OK " if status.ok else "FAIL"
            parts.append(
                f"  [{flag}] {status.channel:<6} n={status.count:<3} {status.elapsed_ms}ms {status.error}"
            )
        for item in response.items:
            ranks = ", ".join(
                f"{name}#{rank}" for name, rank in sorted(item.channel_ranks.items())
            )
            parts.append(
                f"  #{item.rank} {item.clip_id} score={item.normalized_score:.3f} [{ranks}] "
                f"{item.time_range}"
            )
        if response.degraded:
            parts.append("  degraded: " + "; ".join(response.degraded))
        return "\n".join(parts)


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1_000)
