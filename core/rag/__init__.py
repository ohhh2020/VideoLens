"""``core.rag`` —— 检索层：切片 → 建索引 → 混合检索。

三层职责
--------
1. **切片**（``chunker``）
   把 faster-whisper 的毫秒级片段合并成 10~30s 的 ``clip_id`` 单元，
   在静音点/句末切断，两侧各留 1.5s 重叠。
2. **检索**（``hybrid_search``）
   三通道并行 —— BGE-M3 文本语义、SigLIP2 图文跨模态、Jieba+BM25 字面匹配，
   用 RRF 融合成统一排序，并组装带 ``video_id`` / ``clip_id`` / 毫秒区间 /
   关键帧路径的引用证据。
3. **存储**（``vector_store``）
   ChromaDB 持久化两个集合：``clip_text``（1024 维）与 ``frame_image``（768 维）。
   向量库随时可重建，溯源锚点始终在 ``core.storage`` 里。

为什么用 RRF 而不是分数加权
--------------------------
BGE-M3 的 cosine 相似度常年在 0.6~0.9 之间，BM25 分数是无上界的正数，
SigLIP2 的 logits 可能为负 —— 三者**量纲不可比**。RRF 只看排名
(``score = Σ weight / (k + rank)``)，天然免疫分布差异，也不需要归一化。
"""

from __future__ import annotations

from core.rag.chunker import (
    ChunkConfig,
    Chunker,
    ChunkError,
    Clip,
    RawSegment,
    Word,
    bind_keyframes,
    segments_from_whisper,
)
from core.rag.hybrid_search import (
    BM25Channel,
    BM25Index,
    CallableEmbedder,
    Candidate,
    ChannelDisabledError,
    ChannelResult,
    ChannelStatus,
    ConfigError,
    Embedder,
    FusedItem,
    HybridConfig,
    HybridSearchError,
    HybridSearcher,
    RetrievalChannel,
    SearchResponse,
    VectorChannel,
    format_hms,
)
from core.rag.vector_store import (
    CLIP_COLLECTION,
    FRAME_COLLECTION,
    CollectionKind,
    DimensionMismatchError,
    VectorHit,
    VectorStore,
    VectorStoreError,
    VectorStoreUnavailableError,
    iter_batches,
)

__all__ = [
    # chunker
    "ChunkConfig",
    "Chunker",
    "ChunkError",
    "Clip",
    "RawSegment",
    "Word",
    "bind_keyframes",
    "segments_from_whisper",
    # hybrid_search
    "BM25Channel",
    "BM25Index",
    "CallableEmbedder",
    "Candidate",
    "ChannelDisabledError",
    "ChannelResult",
    "ChannelStatus",
    "ConfigError",
    "Embedder",
    "FusedItem",
    "HybridConfig",
    "HybridSearchError",
    "HybridSearcher",
    "RetrievalChannel",
    "SearchResponse",
    "VectorChannel",
    "format_hms",
    # vector_store
    "CLIP_COLLECTION",
    "FRAME_COLLECTION",
    "CollectionKind",
    "DimensionMismatchError",
    "VectorHit",
    "VectorStore",
    "VectorStoreError",
    "VectorStoreUnavailableError",
    "iter_batches",
]
