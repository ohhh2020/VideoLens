"""知影 · 混合检索调用示例（可离线直接运行）。

演示三件事：

1. 用 :class:`core.rag.chunker.Chunker` 把转录片段切成 10~30s 的检索单元；
2. 把切片写进真实 Chroma（文本集合 + 关键帧图像集合）并建 BM25 索引；
3. 三通道并行召回，用 RRF 融合成带溯源证据的上下文。

关于嵌入的说明
--------------
本示例不加载 BGE-M3 / SigLIP2 权重（几十 GB，且需要 ModelManager 预先
下载），改用「jieba 分词 + 哈希词袋」的确定性向量近似替代：

- 文本通道：语义近似为"共享词多则相似度高"，足以演示通道行为；
- 图像通道：真实实现应调用 SigLIP2 的图像/文本编码器，
  但接口形状完全一致（``Embedder.embed_query`` / ``embed_texts``），
  换成真模型时只需替换 ``embed`` 的实现。

运行::

    python examples/hybrid_search_demo.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jieba  # noqa: E402

from core.rag import (  # noqa: E402
    BM25Channel,
    CallableEmbedder,
    Chunker,
    HybridConfig,
    HybridSearcher,
    VectorChannel,
)

DIM = 256


def embed(text: str) -> list[float]:
    """确定性词袋嵌入：共享词越多，余弦相似度越高。"""
    vector = [0.0] * DIM
    for token in jieba.lcut(text):
        token = token.strip().lower()
        if not token or token.isspace():
            continue
        index = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % DIM
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


# --------------------------------------------------------------------------- #
# 1) 造数据：一段 60s 的教学视频转录
# --------------------------------------------------------------------------- #
TRANSCRIPT = [
    {"start": 0.0, "end": 6.2, "text": "大家好，今天介绍知影的设计。它是一个本地优先的视频分析助手。"},
    {"start": 6.4, "end": 12.8, "text": "第一步做场景检测，用 PySceneDetect 找出镜头切换点。"},
    {"start": 13.6, "end": 20.1, "text": "关键帧用拉普拉斯方差过滤模糊画面，保证清晰度不低于七百二十p。"},
    {"start": 20.3, "end": 26.9, "text": "第二步是语音转写，优先读取视频自带的字幕文件。"},
    {"start": 27.1, "end": 34.5, "text": "没有字幕时才用 faster-whisper 做识别，输出毫秒级时间戳。"},
    {"start": 35.2, "end": 42.0, "text": "第三步把转录合并成十秒到三十秒的切片，边界保留一秒半重叠。"},
    {"start": 43.1, "end": 50.4, "text": "检索阶段同时跑 BGE-M3 文本向量、SigLIP2 图像向量和 BM25 关键词。"},
    {"start": 51.0, "end": 58.8, "text": "最后用 RRF 融合三路排名，把最相关的片段连同关键帧一起交给大模型。"},
]

chunker = Chunker()
clips = chunker.chunk("demo-video", TRANSCRIPT, total_duration_ms=60_000)
print(f"[1] 生成 {len(clips)} 个切片")
for clip in clips:
    print(f"    {clip.clip_id}  {clip.start_time}–{clip.end_time}  {clip.text[:28]}…")

# 模拟 image 层已经抽好的关键帧（真实场景由 MediaPipeline 产出）
KEYFRAMES = [
    {"keyframe_id": "kf-0000", "clip_id": "demo-video-c0000", "timestamp_ms": 5_000,
     "keyframe_path": "D:/VideoLens/data/demo-video/kf_000000_000005000.jpg",
     "caption": "片头标题页，写着知影 ZhiYing"},
    {"keyframe_id": "kf-0001", "clip_id": "demo-video-c0001", "timestamp_ms": 25_000,
     "keyframe_path": "D:/VideoLens/data/demo-video/kf_000001_000025000.jpg",
     "caption": "终端里 faster-whisper 正在输出带时间戳的字幕"},
    {"keyframe_id": "kf-0002", "clip_id": "demo-video-c0002", "timestamp_ms": 45_000,
     "keyframe_path": "D:/VideoLens/data/demo-video/kf_000002_000045000.jpg",
     "caption": "架构图：三条检索通道汇入 RRF 融合节点"},
]

# --------------------------------------------------------------------------- #
# 2) 建三通道
# --------------------------------------------------------------------------- #
import chromadb  # noqa: E402

client = chromadb.EphemeralClient()  # 生产用 PersistentClient(path=...)

# --- 通道 A：文本语义 ---
text_collection = client.create_collection("clip_text", metadata={"hnsw:space": "cosine"})
text_collection.add(
    ids=[clip.clip_id for clip in clips],
    embeddings=[embed(clip.text) for clip in clips],
    documents=[clip.text for clip in clips],
    # 注意：Chroma 元数据只接受 str/int/float/bool，列表必须序列化
    metadatas=[
        {
            "clip_id": clip.clip_id,
            "video_id": clip.video_id,
            "start_ms": clip.core_start_ms,
            "end_ms": clip.core_end_ms,
            "text": clip.text,
            "keyframe_ids": ",".join(clip.keyframe_ids),
        }
        for clip in clips
    ],
)

# --- 通道 B：跨模态图文（检索单元是单张关键帧）---
image_collection = client.create_collection("frame_image", metadata={"hnsw:space": "cosine"})
image_collection.add(
    ids=[frame["keyframe_id"] for frame in KEYFRAMES],
    embeddings=[embed(frame["caption"]) for frame in KEYFRAMES],
    documents=[frame["caption"] for frame in KEYFRAMES],
    metadatas=[
        {
            "keyframe_id": frame["keyframe_id"],
            "clip_id": frame["clip_id"],
            "video_id": "demo-video",
            "timestamp_ms": frame["timestamp_ms"],
            "keyframe_path": frame["keyframe_path"],
        }
        for frame in KEYFRAMES
    ],
)

embedder = CallableEmbedder(embed)
channels = [
    VectorChannel("text", text_collection, embedder, id_field="clip_id"),
    VectorChannel("image", image_collection, embedder, id_field="keyframe_id"),
    BM25Channel.from_clips(clips),
]

# clip 索引是权威源：向量库里的元数据可能陈旧，融合时以它为准
clip_index = {clip.clip_id: clip for clip in clips}
keyframe_paths = {frame["keyframe_id"]: frame["keyframe_path"] for frame in KEYFRAMES}

searcher = HybridSearcher(
    channels,
    HybridConfig(top_k=3, min_score=0.10),
    clip_index=clip_index,
    keyframe_resolver=lambda fid: keyframe_paths.get(fid),
)

# --------------------------------------------------------------------------- #
# 3) 查询
# --------------------------------------------------------------------------- #
QUERIES = [
    "切片长度是怎么定的",          # 语义型：靠文本向量
    "faster-whisper 时间戳",       # 专名型：靠 BM25 保底
    "架构图里的融合节点长什么样",   # 跨模态型：靠图像通道
]

for query in QUERIES:
    print("\n" + "=" * 78)
    print(f"查询: {query}")
    print("=" * 78)
    response = searcher.search(query)
    print(searcher.explain(response))

    print("\n--- 交给 LLM 的上下文 ---")
    print(searcher.build_context(response.items))

    print("\n--- 结构化输出（前端 / MCP 用）---")
    print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2)[:1200])

searcher.close()
print("\n[完成] 三条通道 + RRF 融合演示结束。")
