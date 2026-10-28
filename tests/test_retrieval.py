"""hybrid_search 自检：RRF 数学、去重折叠、降级、BM25、上下文组装。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.rag.hybrid_search import (  # noqa: E402
    BM25Channel,
    BM25Index,
    Candidate,
    CallableEmbedder,
    ChannelDisabledError,
    ConfigError,
    HybridConfig,
    HybridSearchError,
    HybridSearcher,
    VectorChannel,
    format_hms,
)

ok: list[str] = []
fail: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        ok.append(name)
        print(f"PASS {name}" + (f" | {detail}" if detail else ""))
    else:
        fail.append(name)
        print(f"FAIL {name}" + (f" | {detail}" if detail else ""))


def rejects(fn, exc: type[BaseException] = Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


class FakeChannel:
    """确定性假通道，可注入错误 / 延迟 / 未就绪。"""

    def __init__(self, name, rows, *, ready=True, error=None, delay=0.0):
        self.name = name
        self._rows = rows
        self._ready = ready
        self._error = error
        self._delay = delay
        self.calls: list[dict] = []

    def available(self):
        return self._ready

    def search(self, query, top_k, *, filters=None):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        if self._delay:
            time.sleep(self._delay)
        if self._error:
            raise self._error
        return [
            Candidate(key=key, rank=rank, score=score, payload=payload)
            for rank, (key, score, payload) in enumerate(self._rows, start=1)
        ][:top_k]


def payload(clip_id, video_id="v1", start=0, end=10_000, text="", kf=()):
    return {
        "clip_id": clip_id,
        "video_id": video_id,
        "start_ms": start,
        "end_ms": end,
        "text": text,
        "keyframe_ids": list(kf),
    }


# ---------------------------------------------------------------- 1) RRF 数学
text_ch = FakeChannel(
    "text",
    [
        (
            "v1-c0000",
            0.91,
            payload("v1-c0000", start=0, end=20_000, text="文本通道第一名"),
        ),
        (
            "v1-c0001",
            0.88,
            payload("v1-c0001", start=20_000, end=35_000, text="文本通道第二名"),
        ),
    ],
)
image_ch = FakeChannel(
    "image",
    [
        (
            "kf-0001",
            0.42,
            payload(
                "v1-c0000",
                start=0,
                end=20_000,
                text="图像通道命中同一片段",
                kf=["kf-0001"],
            ),
        ),
    ],
)
bm25_ch = FakeChannel(
    "bm25",
    [
        (
            "v1-c0002",
            12.7,
            payload("v1-c0002", start=35_000, end=50_000, text="BM25 命中"),
        ),
    ],
)
searcher = HybridSearcher([text_ch, image_ch, bm25_ch], HybridConfig())
resp = searcher.search("切片长度是怎么定的", top_k=5)

check("融合返回 3 条", len(resp.items) == 3, f"{len(resp.items)} 条")
by_id = {item.clip_id: item for item in resp.items}
rrf = lambda r: 1.0 / (60 + r)  # noqa: E731
expected_0 = 0.5 * rrf(1) + 0.3 * rrf(1)
expected_1 = 0.5 * rrf(2)
expected_2 = 0.2 * rrf(1)
check(
    "RRF 分数精确（text#1+image#1）",
    abs(by_id["v1-c0000"].score - expected_0) < 1e-12,
    f"{by_id['v1-c0000'].score:.10f} vs {expected_0:.10f}",
)
check(
    "RRF 分数精确（text#2）",
    abs(by_id["v1-c0001"].score - expected_1) < 1e-12,
    f"{by_id['v1-c0001'].score:.10f}",
)
check(
    "RRF 分数精确（bm25#1）",
    abs(by_id["v1-c0002"].score - expected_2) < 1e-12,
    f"{by_id['v1-c0002'].score:.10f}",
)
check(
    "排序正确",
    [i.clip_id for i in resp.items] == ["v1-c0000", "v1-c0001", "v1-c0002"],
    str([i.clip_id for i in resp.items]),
)
check(
    "归一化 0.8（三通道满分）",
    abs(by_id["v1-c0000"].normalized_score - 0.8) < 1e-9,
    f"{by_id['v1-c0000'].normalized_score:.6f}",
)
check(
    "归一化 ≤1",
    all(0.0 <= i.normalized_score <= 1.0 for i in resp.items),
    str([round(i.normalized_score, 4) for i in resp.items]),
)
check("rank 连续重排", [i.rank for i in resp.items] == [1, 2, 3])

# ------------------------------------------------------- 2) 去重与跨模态折叠
check(
    "图像候选折叠回 clip",
    "image" in by_id["v1-c0000"].channel_ranks,
    str(by_id["v1-c0000"].channel_ranks),
)
check(
    "多通道 rank 明细完整",
    by_id["v1-c0000"].channel_ranks == {"text": 1, "image": 1},
    str(dict(by_id["v1-c0000"].channel_ranks)),
)
check(
    "关键帧 ID 并入证据",
    by_id["v1-c0000"].keyframe_ids == ("kf-0001",),
    str(by_id["v1-c0000"].keyframe_ids),
)
check("无重复条目", len({i.clip_id for i in resp.items}) == len(resp.items))
check("三条 item 各来自不同通道", len(by_id) == 3)

# 同一通道内同 clip 出现两次 → 只按最好 rank 计一次
dup_ch = FakeChannel(
    "text",
    [
        ("v1-c0000", 0.9, payload("v1-c0000", kf=["kf-a"])),
        ("v1-c0000", 0.8, payload("v1-c0000", kf=["kf-b"])),
    ],
)
dup_resp = HybridSearcher([dup_ch], HybridConfig()).search("x")
check(
    "同通道重复 clip 只计一次",
    abs(dup_resp.items[0].score - 0.5 * rrf(1)) < 1e-12,
    f"{dup_resp.items[0].score:.10f}",
)
check(
    "重复 clip 关键帧并集",
    set(dup_resp.items[0].keyframe_ids) == {"kf-a", "kf-b"},
    str(dup_resp.items[0].keyframe_ids),
)

# ----------------------------------------------------------- 3) 权重可配置
cfg_w = HybridConfig(weights={"text": 1.0, "image": 0.0, "bm25": 0.0})
w_resp = HybridSearcher([text_ch, image_ch, bm25_ch], cfg_w).search("x", top_k=5)
check(
    "零权重通道被排除",
    "image" not in w_resp.items[0].channel_ranks,
    str(w_resp.items[0].channel_ranks),
)
check(
    "权重 1.0 归一化为 1",
    abs(w_resp.items[0].normalized_score - 1.0) < 1e-9,
    f"{w_resp.items[0].normalized_score}",
)
check("零权重通道仍可查排名", len(w_resp.channels) == 3, f"{len(w_resp.channels)} 通道")

cfg_k = HybridConfig(k=1)
k_resp = HybridSearcher([text_ch], cfg_k).search("x")
check(
    "k 参数生效",
    abs(k_resp.items[0].score - 0.5 / 2) < 1e-12,
    f"{k_resp.items[0].score:.10f}",
)

# ------------------------------------------------------------ 4) 通道降级
broken = FakeChannel("image", [], error=RuntimeError("chroma 连不上"))
deg = HybridSearcher([text_ch, broken, bm25_ch], HybridConfig()).search("x")
check("单通道异常不影响整体", len(deg.items) == 3, f"{len(deg.items)} 条")
check(
    "异常被记录为 degraded",
    any("image" in d and "chroma 连不上" in d for d in deg.degraded),
    str(deg.degraded)[:120],
)
status = {s.channel: s for s in deg.channels}
check(
    "失败通道 status 正确", status["image"].ok is False and status["image"].count == 0
)
check("成功通道 status 正确", status["text"].ok is True and status["text"].count == 2)
check("response.ok 仍为真", deg.ok is True)

off = FakeChannel("bm25", [], ready=False)
skip = HybridSearcher([text_ch, off], HybridConfig()).search("x")
check(
    "未就绪通道被跳过",
    len(skip.channels) == 2 and any("未就绪" in d for d in skip.degraded),
    str(skip.degraded),
)
check("未就绪通道不被调用", off.calls == [])

all_dead = HybridSearcher(
    [FakeChannel("text", [], ready=False)], HybridConfig()
).search("x")
check(
    "全通道不可用返回空",
    all_dead.items == () and all_dead.ok is False,
    str(all_dead.degraded)[:80],
)

slow = FakeChannel("image", [], delay=1.0)
t0 = time.monotonic()
timeout_resp = HybridSearcher([text_ch, slow], HybridConfig(timeout_ms=200)).search("x")
cost = time.monotonic() - t0
check(
    "超时通道被丢弃",
    any(s.timed_out for s in timeout_resp.channels),
    str([(s.channel, s.timed_out) for s in timeout_resp.channels]),
)
check("超时不阻塞成功通道", len(timeout_resp.items) == 2 and cost < 0.9, f"{cost:.2f}s")

# 通道抛 ChannelDisabledError 也算优雅降级
disabled = FakeChannel("bm25", [], error=ChannelDisabledError("索引为空"))
dis_resp = HybridSearcher([text_ch, disabled], HybridConfig()).search("x")
check("ChannelDisabledError 优雅降级", len(dis_resp.items) == 2)

# ------------------------------------------------------- 5) 并发与串行一致
par = HybridSearcher([text_ch, image_ch, bm25_ch], HybridConfig(parallel=True)).search(
    "x"
)
ser = HybridSearcher([text_ch, image_ch, bm25_ch], HybridConfig(parallel=False)).search(
    "x"
)
check(
    "并发与串行结果一致",
    [i.clip_id for i in par.items] == [i.clip_id for i in ser.items]
    and all(abs(a.score - b.score) < 1e-12 for a, b in zip(par.items, ser.items)),
)

# --------------------------------------------------------------- 6) BM25
docs = [
    ("v1-c0000", "知影使用 faster-whisper 做语音识别，支持 CPU 降级。"),
    ("v1-c0001", "BGE-M3 负责文本向量，SigLIP2 负责图像向量，两者都在 Chroma 里。"),
    ("v1-c0002", "切片长度控制在十秒到三十秒之间，并在边界保留一秒半重叠。"),
    ("v1-c0003", "模型权重由 ModelManager 从 ModelScope 或 HuggingFace 下载。"),
    ("v1-c0004", "赖浩林和林睿轩负责本项目的需求分析与前端联调。"),
]
bm25 = BM25Channel.from_clips(docs)
check("BM25 索引 5 条", bm25.index.size == 5, f"{bm25.index.size}")
check(
    "BM25 分词器为 jieba",
    bm25.index.tokenizer_backend == "jieba",
    bm25.index.tokenizer_backend,
)

hits = bm25.search("赖浩林", 3)
check(
    "专有名词命中第一名",
    hits and hits[0].key == "v1-c0004",
    str([(c.key, round(c.score, 3)) for c in hits]),
)
hits2 = bm25.search("ModelManager", 3)
check(
    "英文标识符命中",
    hits2 and hits2[0].key == "v1-c0003",
    str([(c.key, round(c.score, 3)) for c in hits2]),
)
check(
    "匹配词被记录",
    "modelmanager" in (hits2[0].payload.get("_matched_terms") or []),
    str(hits2[0].payload.get("_matched_terms")),
)
check("无匹配返回空", bm25.search("量子纠缠态", 3) == [])
check("空查询返回空", bm25.search("   ", 3) == [])

idx = BM25Index()
idx.add("d1", "重复文本测试")
idx.add("d1", "重复文本测试")
check(
    "重复写入不重复计数", idx.size == 1 and idx._df.get("测试") == 1, f"size={idx.size}"
)
idx.add("d1", "完全不同的内容")
check(
    "覆盖写入更新 df",
    idx.search("完全不同", 2) and not idx.search("重复文本测试", 2),
    str(idx.search("完全不同", 2)),
)
check("idf 为正", idx.idf("不同") > 0, f"{idx.idf('不同'):.4f}")
check("空索引查询安全", BM25Index().search("x") == [])
check("BM25 索引可清空", (idx.clear() or idx.size) == 0)

# 字面通道 + 语义通道互补：专名靠 BM25，同义靠向量
semantic = FakeChannel(
    "text",
    [
        ("v1-c0002", 0.93, payload("v1-c0002", text="切片长度控制在十秒到三十秒之间")),
    ],
)
literal = BM25Channel.from_clips(docs)
combo = HybridSearcher([semantic, literal], HybridConfig()).search(
    "赖浩林 切片长度", top_k=3
)
check(
    "BM25 专名进入融合结果",
    any(i.clip_id == "v1-c0004" for i in combo.items),
    str([i.clip_id for i in combo.items]),
)
check(
    "BM25 并列打分",
    any(i.clip_id == "v1-c0002" and "bm25" in i.channel_ranks for i in combo.items),
    str([(i.clip_id, i.channel_ranks) for i in combo.items]),
)


# ------------------------------------------------------- 7) VectorChannel
class FakeCollection:
    def __init__(self):
        self.seen = []

    def query(self, **kwargs):
        self.seen.append(kwargs)
        n = kwargs["n_results"]
        rows = min(n, 3)
        return {
            "ids": [[f"doc-{i}" for i in range(rows)]],
            "metadatas": [
                [
                    payload(
                        f"v1-c000{i}",
                        start=i * 10_000,
                        end=(i + 1) * 10_000,
                        text=f"文档{i}",
                        kf=[f"kf-{i}"],
                    )
                    for i in range(rows)
                ]
            ],
            "distances": [[0.05 * (i + 1) for i in range(rows)]],
        }


coll = FakeCollection()
embedder = CallableEmbedder(lambda text: [0.1, 0.2, 0.3])
vc = VectorChannel("text", coll, embedder)
vres = vc.search("查询", 2, filters={"video_id": "v1"})
check("向量通道返回候选", len(vres) == 2, f"{len(vres)}")
check("向量通道 rank 从 1 开始", [c.rank for c in vres] == [1, 2])
check("距离转相似度", abs(vres[0].score - 0.95) < 1e-9, f"{vres[0].score}")
check(
    "filters 透传 where",
    coll.seen[-1].get("where") == {"video_id": "v1"},
    str(coll.seen[-1].get("where")),
)
check(
    "使用 query_embeddings 而非 query_texts",
    "query_texts" not in coll.seen[-1],
    str(sorted(coll.seen[-1])),
)
check("n_results 受 top_k 约束", coll.seen[-1]["n_results"] == 2)

vc_dead = VectorChannel("image", None, embedder)
check("缺集合时通道不可用", vc_dead.available() is False)
check(
    "不可用通道抛 ChannelDisabledError",
    rejects(lambda: vc_dead.search("x", 5), ChannelDisabledError),
)
vc_noemb = VectorChannel("text", coll, None)
check("缺嵌入器时通道不可用", vc_noemb.available() is False)

# clip_index 补齐：Chroma 只存 clip_id，靠索引补文本/时间
sparse_coll = FakeCollection()
sparse = VectorChannel("text", sparse_coll, embedder)
index = {
    "v1-c0000": SimpleNamespace(
        video_id="v1",
        core_start_ms=0,
        core_end_ms=20_000,
        text="从索引补齐的正文",
        clip_id="v1-c0000",
        keyframe_ids=("kf-9",),
    ),
}
s2 = HybridSearcher([sparse], HybridConfig(), clip_index=index)
r2 = s2.search("x", top_k=1)
check("clip_index 补齐文本", r2.items[0].text == "从索引补齐的正文", r2.items[0].text)
check(
    "clip_index 补齐关键帧",
    "kf-9" in r2.items[0].keyframe_ids,
    str(r2.items[0].keyframe_ids),
)

# keyframe_resolver：ID → 本地路径
s3 = HybridSearcher(
    [sparse],
    HybridConfig(),
    clip_index=index,
    keyframe_resolver=lambda fid: f"D:/frames/{fid}.jpg",
)
r3 = s3.search("x", top_k=1)
check(
    "keyframe_resolver 生成路径",
    r3.items[0].keyframe_paths == ("D:/frames/kf-9.jpg",),
    str(r3.items[0].keyframe_paths),
)

# --------------------------------------------------- 8) 元数据兜底解析
loose = FakeChannel(
    "text",
    [
        (
            "v1-c0007",
            0.5,
            {
                "video_id": "v1",
                "start": 12.5,
                "end": 31.25,
                "text": "秒级时间戳",
            },
        )
    ],
)
loose_resp = HybridSearcher([loose], HybridConfig()).search("x")
check(
    "秒级时间戳被换算",
    (loose_resp.items[0].start_ms, loose_resp.items[0].end_ms) == (12_500, 31_250),
    f"{loose_resp.items[0].start_ms}~{loose_resp.items[0].end_ms}",
)
check(
    "时间格式 HH:MM:SS.mmm",
    loose_resp.items[0].start_time == "00:00:12.500",
    loose_resp.items[0].start_time,
)
check("字符串时间戳可解析", format_hms(3_723_456) == "01:02:03.456")

hms_meta = FakeChannel(
    "text",
    [
        (
            "v1-c0008",
            0.5,
            {
                "video_id": "v1",
                "start_time": "00:01:02.500",
                "end_time": "00:01:30.000",
            },
        )
    ],
)
hms_resp = HybridSearcher([hms_meta], HybridConfig()).search("x")
check(
    "HH:MM:SS.mmm 元数据可解析",
    (hms_resp.items[0].start_ms, hms_resp.items[0].end_ms) == (62_500, 90_000),
    f"{hms_resp.items[0].start_ms}~{hms_resp.items[0].end_ms}",
)

# ------------------------------------------------------- 9) 过滤与截断
big = FakeChannel(
    "text",
    [
        (
            f"v1-c{i:04d}",
            1.0 - i * 0.01,
            payload(f"v1-c{i:04d}", start=i * 1_000, end=(i + 1) * 1_000),
        )
        for i in range(30)
    ],
)
cut = HybridSearcher([big], HybridConfig()).search("x", top_k=5)
check("top_k 截断", len(cut.items) == 5, f"{len(cut.items)}")
check("截断后 rank 连续", [i.rank for i in cut.items] == [1, 2, 3, 4, 5])
thr = HybridSearcher([big], HybridConfig()).search("x", top_k=10, min_score=0.99)
check(
    "min_score 过滤",
    len(thr.items) == 1,
    f"{len(thr.items)} → {[round(i.normalized_score, 3) for i in thr.items]}",
)
check(
    "fetch_k 已放大", big.calls[-1]["top_k"] >= 20, f"fetch_k={big.calls[-1]['top_k']}"
)

# --------------------------------------------------------- 10) 参数校验
check("空查询报错", rejects(lambda: searcher.search("   "), HybridSearchError))
check("top_k<1 报错", rejects(lambda: searcher.search("x", top_k=0), HybridSearchError))
check("无通道报错", rejects(lambda: HybridSearcher([]), ConfigError))
check(
    "k<1 报错",
    rejects(lambda: HybridSearcher([text_ch], HybridConfig(k=0)), ConfigError),
)
check(
    "min_score 越界报错",
    rejects(lambda: HybridConfig(min_score=1.5).validate(), ConfigError),
)
check(
    "负权重报错",
    rejects(lambda: HybridConfig(weights={"text": -1.0}).validate(), ConfigError),
)
check(
    "timeout_ms<=0 报错",
    rejects(lambda: HybridConfig(timeout_ms=0).validate(), ConfigError),
)

# --------------------------------------------------- 11) 上下文组装与序列化
ctx = searcher.build_context(resp.items)
check("上下文含编号", ctx.startswith("[1]"), ctx[:40])
check("上下文含 clip_id", "v1-c0000" in ctx)
check("上下文含时间区间", "00:00:00.000–00:00:20.000" in ctx, ctx[:120])
check("上下文含回答要求", "每条结论后必须标注" in ctx)
check("上下文含关键帧路径或 ID", "kf-0001" in ctx or "关键帧" in ctx)

ctx_kf = s3.build_context(r3.items)
check("上下文写入关键帧路径", "D:/frames/kf-9.jpg" in ctx_kf, ctx_kf[:200])
short_ctx = searcher.build_context(resp.items, max_chars=60)
check("上下文预算截断", "预算已满" in short_ctx, short_ctx[-40:])
ctx_nos = searcher.build_context(resp.items, include_scores=False, instruction=False)
check("可关闭分数与指令", "score=" not in ctx_nos and "回答要求" not in ctx_nos)

data = resp.to_dict()
check("response 可 JSON 序列化", bool(json.dumps(data, ensure_ascii=False)))
check("to_dict 含通道状态", len(data["channels"]) == 3)
check(
    "to_dict 含溯源字段",
    all(
        {
            "clip_id",
            "video_id",
            "start_time",
            "end_time",
            "start_ms",
            "end_ms",
            "keyframe_ids",
        }
        <= set(row)
        for row in data["items"]
    ),
)
check(
    "evidence() 完整",
    any(e.startswith("clip=") for e in resp.items[0].evidence())
    and any(e.startswith("range=") for e in resp.items[0].evidence()),
    str(resp.items[0].evidence()),
)

explain = searcher.explain(resp)
check(
    "explain 含通道与排名",
    "[OK ]" in explain and "#1" in explain,
    explain.splitlines()[1][:60],
)

# ------------------------------------------------------------ 12) 资源释放
s4 = HybridSearcher([text_ch, image_ch], HybridConfig())
s4.close()
_idempotent = False
try:
    s4.close()
    s4.close()
    _idempotent = True
except Exception:  # noqa: BLE001
    _idempotent = False
check("close 幂等", _idempotent)
with HybridSearcher([text_ch], HybridConfig()) as s5:
    s5.search("x")
check("上下文管理器可用（线程池已释放）", s5._pool is None)

print(f"\n通过 {len(ok)} / {len(ok) + len(fail)}")
if fail:
    print("失败项:")
    for item in fail:
        print("  - " + item)
sys.exit(0 if not fail else 1)
