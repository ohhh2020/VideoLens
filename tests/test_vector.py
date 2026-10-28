"""``backend.rag.vector_store`` 自检。

覆盖：集合创建、metadata 扁平化（list/bool/None/Path）、upsert 幂等、
维度校验、近邻排序、where 过滤、批量查询、按 video_id 删除、reset、
与 hybrid_search.VectorChannel 的对接。

跑法：
    python .tmp_test/t_vector.py
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.rag.vector_store import (  # noqa: E402
    DimensionMismatchError,
    VectorStore,
    VectorStoreError,
    _flatten_metadata,
)

PASSED = 0
FAILED: list[str] = []

DIM = 64
_ROOTS: list[Path] = []


def mk(**kwargs: object) -> VectorStore:
    """每个实例一个独立落盘目录 —— EphemeralClient 在同进程内共享后端，会串味。"""
    root = Path(tempfile.mkdtemp(prefix="zy-vs-"))
    _ROOTS.append(root)
    return VectorStore(root / "chroma", **kwargs)  # type: ignore[arg-type]


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED
    if cond:
        PASSED += 1
        print(f"  ok   {name}")
    else:
        FAILED.append(f"{name}  {detail}")
        print(f"  FAIL {name}  {detail}")


def vec(seed: int, *, dim: int = DIM) -> list[float]:
    rng = random.Random(seed)
    raw = [rng.uniform(-1, 1) for _ in range(dim)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


def main() -> int:
    global PASSED

    # --- 1. metadata 扁平化 ---
    flat = _flatten_metadata(
        {
            "clip_id": "c-1",
            "start_ms": 1500,
            "score": 0.25,
            "ok": True,
            "missing": None,
            "keyframe_ids": ["kf-1", "kf-2"],
            "nums": [1, 2],
            "nested": {"a": 1},
            "path": Path("a/b.jpg"),
            "": "empty-key-dropped",
        }
    )
    check(
        "str/int/float/bool 原样",
        flat["clip_id"] == "c-1"
        and flat["start_ms"] == 1500
        and flat["score"] == 0.25
        and flat["ok"] is True,
        str(flat),
    )
    check("None 被丢弃", "missing" not in flat, str(list(flat)))
    check("空键被丢弃", "empty-key-dropped" not in flat.values(), str(flat))
    check(
        "str 列表 → 逗号串", flat["keyframe_ids"] == "kf-1,kf-2", flat["keyframe_ids"]
    )
    check("非 str 列表 → JSON", flat["nums"] == "[1, 2]", flat["nums"])
    check("dict → JSON", flat["nested"] == '{"a": 1}', flat["nested"])
    check(
        "Path → str",
        isinstance(flat["path"], str)
        and flat["path"] == "a\\b.jpg"
        or flat["path"] in ("a/b.jpg", "a\\b.jpg"),
        flat["path"],
    )

    # --- 2. 集合名校验 ---
    for bad in ("ab", "", "a b", "-ab", "ab-", "x" * 600):
        try:
            VectorStore.ephemeral(clip_collection=bad)
            check(f"非法集合名 {bad[:12]!r} 应报错", False, "")
        except VectorStoreError:
            PASSED += 1
            print(f"  ok   非法集合名 {bad[:12]!r} 报错")
    try:
        VectorStore.ephemeral(clip_collection="same", frame_collection="same")
        check("同名集合应报错", False, "")
    except VectorStoreError:
        check("同名集合报错", True, "")
    try:
        VectorStore.ephemeral(metric="hamming")
        check("非法 metric 应报错", False, "")
    except VectorStoreError:
        check("非法 metric 报错", True, "")

    # --- 3. 写入 / 检索 ---
    store = mk()
    check("health", store.health() is True, "")
    check("初始 count=0", store.count("clip") == 0, str(store.count("clip")))
    check(
        "初始 dim 未知", store.dimension("clip") is None, str(store.dimension("clip"))
    )
    check("metric 记录", store.metric == "cosine", store.metric)

    ids = ["c-0", "c-1", "c-2"]
    embs = [vec(i) for i in range(3)]
    docs = ["第一段讲采集", "第二段讲切片", "第三段讲融合"]
    metas = [
        {
            "clip_id": "c-0",
            "video_id": "v1",
            "start_ms": 0,
            "end_ms": 20_100,
            "keyframe_ids": ["kf-0"],
        },
        {
            "clip_id": "c-1",
            "video_id": "v1",
            "start_ms": 18_600,
            "end_ms": 38_500,
            "keyframe_ids": ["kf-1", "kf-2"],
        },
        {"clip_id": "c-2", "video_id": "v2", "start_ms": 0, "end_ms": 12_000},
    ]
    n = store.upsert_clips(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    check("写入 3 条", n == 3, str(n))
    check("count=3", store.count("clip") == 3, str(store.count("clip")))
    check("dim 已记录", store.dimension("clip") == DIM, str(store.dimension("clip")))

    store.upsert_clips(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    check("同 id upsert 幂等", store.count("clip") == 3, str(store.count("clip")))

    hits = store.query_clips(vec(1), top_k=3)
    check("检索返回 3 条", len(hits) == 3, str(len(hits)))
    check("最相似是自己", hits[0].id == "c-1", hits[0].id)
    check(
        "score 降序",
        all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1)),
        str([round(h.score, 4) for h in hits]),
    )
    check("self score≈1", abs(hits[0].score - 1.0) < 1e-4, str(hits[0].score))
    check("document 带回", hits[0].document == "第二段讲切片", hits[0].document)
    check("metadata 带回", hits[0].metadata["video_id"] == "v1", str(hits[0].metadata))
    check("clip_id 属性", hits[0].clip_id == "c-1", hits[0].clip_id)
    by_id = {h.id: h for h in hits}
    check(
        "list 被序列化",
        by_id["c-1"].metadata["keyframe_ids"] == "kf-1,kf-2",
        str(by_id["c-1"].metadata.get("keyframe_ids")),
    )
    check(
        "to_dict JSON 化", json.dumps(hits[0].to_dict(), ensure_ascii=False) != "", ""
    )

    # where 过滤
    only_v1 = store.query_clips(vec(1), top_k=10, where={"video_id": "v1"})
    check(
        "where 过滤生效",
        {h.id for h in only_v1} == {"c-0", "c-1"},
        str([h.id for h in only_v1]),
    )
    only_v2 = store.query_clips(vec(0), top_k=10, where={"video_id": "v2"})
    check(
        "where 命中单条",
        [h.id for h in only_v2] == ["c-2"],
        str([h.id for h in only_v2]),
    )

    # top_k 限制
    check("top_k=1", len(store.query_clips(vec(0), top_k=1)) == 1, "")

    # --- 4. 批量查询 ---
    many = store.query_many("clip", [vec(0), vec(2)], top_k=2)
    check("批量返回 2 组", len(many) == 2, str(len(many)))
    check("批量组内 2 条", all(len(g) == 2 for g in many), str([len(g) for g in many]))
    check(
        "批量首条正确",
        many[0][0].id == "c-0" and many[1][0].id == "c-2",
        f"{many[0][0].id} {many[1][0].id}",
    )
    check("空输入返回空", store.query_many("clip", [], top_k=2) == [], "")

    # --- 5. 维度校验 ---
    try:
        store.upsert_clips(
            ids=["bad"],
            embeddings=[vec(9, dim=DIM + 8)],
            documents=["x"],
            metadatas=[{"clip_id": "bad"}],
        )
        check("维度不一致应报错", False, "")
    except DimensionMismatchError as exc:
        check(
            "维度报错含期望/实际", exc.expected == DIM and exc.got == DIM + 8, str(exc)
        )
        check("维度报错含建议", "重建" in str(exc), str(exc))
    try:
        store.query_clips(vec(0, dim=DIM + 8))
        check("查询维度不一致应报错", False, "")
    except DimensionMismatchError:
        check("查询维度校验", True, "")

    store_noverify = mk(verify_dim=False)
    store_noverify.upsert_clips(
        ids=["a"], embeddings=[vec(1)], documents=["t"], metadatas=[{"clip_id": "a"}]
    )
    try:
        store_noverify.upsert_clips(
            ids=["b"],
            embeddings=[vec(1, dim=DIM + 4)],
            documents=["t"],
            metadatas=[{"clip_id": "b"}],
        )
        check("verify_dim=False 交给 Chroma 兜底", False, "竟然写进去了")
    except DimensionMismatchError:
        check("verify_dim=False 交给 Chroma 兜底", False, "仍抛 DimensionMismatchError")
    except VectorStoreError as exc:
        # 预检查被关闭，Chroma 自己会拒绝并报维度不符
        check(
            "verify_dim=False 交给 Chroma 兜底",
            "dimension" in str(exc).lower(),
            str(exc),
        )
    store_noverify.close()

    # --- 6. 参数校验 ---
    for bad_case, desc in (
        ((dict(ids=[""], embeddings=[[0.0] * DIM]),), "空 id"),
        ((dict(ids=["x", "y"], embeddings=[vec(1)]),), "行数不符"),
        (
            (dict(ids=["x"], embeddings=[vec(1)], documents=["a", "b"]),),
            "documents 数量不符",
        ),
        (
            (dict(ids=["x"], embeddings=[vec(1)], metadatas=[{}, {}]),),
            "metadatas 数量不符",
        ),
    ):
        try:
            store.upsert_clips(**bad_case[0])
            check(f"{desc} 应报错", False, "")
        except VectorStoreError:
            PASSED += 1
            print(f"  ok   {desc} 报错")
    try:
        store.upsert_clips(ids=[], embeddings=[], documents=[], metadatas=[])
        check("空写入返回 0", True, "")
    except VectorStoreError:
        check("空写入返回 0", False, "抛错")

    # --- 7. 图像集合 ---
    store.upsert_frames(
        ids=["kf-0", "kf-1"],
        embeddings=[vec(10), vec(11)],
        documents=["架构图", "界面截图"],
        metadatas=[
            {
                "keyframe_id": "kf-0",
                "video_id": "v1",
                "clip_id": "c-0",
                "timestamp_ms": 5000,
                "keyframe_path": "data/frames/kf-0.jpg",
            },
            {
                "keyframe_id": "kf-1",
                "video_id": "v1",
                "clip_id": "c-1",
                "timestamp_ms": 25_000,
                "keyframe_path": "data/frames/kf-1.jpg",
            },
        ],
    )
    check("帧集合 count=2", store.count("frame") == 2, str(store.count("frame")))
    fh = store.query_frames(vec(11), top_k=1)
    check("帧检索命中", fh[0].id == "kf-1", fh[0].id)
    check(
        "帧 metadata 带路径",
        fh[0].metadata["keyframe_path"].endswith("kf-1.jpg"),
        str(fh[0].metadata),
    )
    check(
        "两集合独立维度", store.dimension("frame") == DIM, str(store.dimension("frame"))
    )

    # --- 8. 删除 ---
    check("按 id 删除", store.delete("clip", ["c-2"]) == 1, "")
    check("删除后 count", store.count("clip") == 2, str(store.count("clip")))
    check("删不存在的 id 不报错", store.delete("clip", ["nope"]) == 1, "返回请求条数")
    check("空 id 列表返回 0", store.delete("clip", []) == 0, "")

    removed = store.delete_video("v1")
    check("按 video 删 clip", removed["clip"] == 2, str(removed))
    check("按 video 删 frame", removed["frame"] == 2, str(removed))
    check("清理后 clip=0", store.count("clip") == 0, str(store.count("clip")))
    check("清理后 frame=0", store.count("frame") == 0, str(store.count("frame")))
    check("空 video_id 返回 0", store.delete_video("") == {"clip": 0, "frame": 0}, "")

    # --- 9. reset ---
    store.upsert_clips(
        ids=["z"], embeddings=[vec(3)], documents=["t"], metadatas=[{"clip_id": "z"}]
    )
    check("reset 前有数据", store.count("clip") == 1, "")
    store.reset("clip")
    check("reset 后清空", store.count("clip") == 0, str(store.count("clip")))
    check(
        "reset 后 dim 清空",
        store.dimension("clip") is None,
        str(store.dimension("clip")),
    )
    check(
        "reset 后仍可写",
        store.upsert_clips(
            ids=["z2"],
            embeddings=[vec(4)],
            documents=["t"],
            metadatas=[{"clip_id": "z2"}],
        )
        == 1,
        "",
    )

    # --- 10. stats / 关闭 ---
    st = store.stats()
    check("stats 结构", "collections" in st and st["metric"] == "cosine", str(st))
    check("stats JSON 化", json.dumps(st, ensure_ascii=False) != "", "")
    store.close()
    check("关闭后 health=False", store.health() is False, "")
    try:
        store.count("clip")
        check("关闭后操作应报错", False, "")
    except VectorStoreError as exc:
        check("关闭后报 Unavailable", "已关闭" in str(exc), str(exc))
    store.close()  # 重复 close

    # --- 11. 持久化 ---
    tmp = Path(tempfile.mkdtemp(prefix="zy-vec-"))
    pstore = VectorStore(tmp / "chroma")
    pstore.upsert_clips(
        ids=["p-1"],
        embeddings=[vec(7)],
        documents=["持久化测试"],
        metadatas=[{"clip_id": "p-1", "video_id": "vx"}],
    )
    check("持久化写入", pstore.count("clip") == 1, "")
    pstore.close()

    pstore2 = VectorStore(tmp / "chroma")
    check("重开后数据还在", pstore2.count("clip") == 1, str(pstore2.count("clip")))
    ph = pstore2.query_clips(vec(7), top_k=1)
    check(
        "重开后可检索",
        ph[0].id == "p-1" and ph[0].document == "持久化测试",
        ph[0].document,
    )
    check("父目录被创建", (tmp / "chroma").is_dir(), str(tmp / "chroma"))
    pstore2.close()

    # --- 12. 与 hybrid_search.VectorChannel 对接 ---
    from backend.rag.hybrid_search import VectorChannel

    store2 = mk()
    store2.upsert_clips(
        ids=["x-0", "x-1"],
        embeddings=[vec(20), vec(21)],
        documents=["讲检索融合", "讲场景检测"],
        metadatas=[
            {"clip_id": "x-0", "video_id": "v9"},
            {"clip_id": "x-1", "video_id": "v9"},
        ],
    )

    class Emb:
        def embed_query(self, text: str) -> list[float]:
            return vec(20 if "融合" in text else 21)

        def embed_texts(self, texts):  # noqa: ANN001, ANN201
            return [self.embed_query(t) for t in texts]

    ch = VectorChannel("text", store2.collection("clip"), Emb(), id_field="clip_id")
    check("VectorChannel 就绪", ch.available() is True, "")
    cands = ch.search("融合", top_k=2, filters=None)
    check("通道返回候选", len(cands) == 2, str(len(cands)))
    check(
        "通道首位正确", cands[0].payload.get("clip_id") == "x-0", str(cands[0].payload)
    )
    check(
        "通道 rank 从 1 起",
        [c.rank for c in cands] == [1, 2],
        str([c.rank for c in cands]),
    )
    store2.close()

    # -- 10. SQLite 后端 ---------------------------------------------------
    print("\n[10] SQLite 向量后端（零依赖降级路径）")

    def mk_sqlite(**kwargs: object) -> VectorStore:
        root = Path(tempfile.mkdtemp(prefix="zy-vs-sql-"))
        _ROOTS.append(root)
        return VectorStore(root / "vdb", backend="sqlite", **kwargs)  # type: ignore[arg-type]

    sq = mk_sqlite()
    check("后端名为 sqlite", sq.backend == "sqlite", sq.backend)
    check("空库健康", sq.health() is True, "")
    check("空库查询返回空", sq.query_clips(vec(1), top_k=3) == [], "")

    # 与 Chroma 同数据同查询，排序必须一致 —— 两种后端不能给出不同答案。
    ids = [f"s-{i}" for i in range(5)]
    embs = [vec(i) for i in range(5)]
    metas = [
        {"video_id": "v1" if i < 3 else "v2", "clip_id": ids[i], "start_ms": i * 1000}
        for i in range(5)
    ]
    chrome = mk()
    for store in (sq, chrome):
        store.upsert_clips(
            ids=ids,
            embeddings=embs,
            documents=[f"文本{i}" for i in range(5)],
            metadatas=metas,
        )
    probe = vec(0)
    sql_hits = [h.id for h in sq.query_clips(probe, top_k=3)]
    chr_hits = [h.id for h in chrome.query_clips(probe, top_k=3)]
    check("两种后端排序一致", sql_hits == chr_hits, f"sql={sql_hits} chroma={chr_hits}")
    check("首位是自身", sql_hits[0] == "s-0", str(sql_hits))
    check("count 正确", sq.count("clip") == 5, str(sq.count("clip")))

    filtered = [h.id for h in sq.query_clips(probe, top_k=5, where={"video_id": "v2"})]
    check("where 过滤生效", filtered == ["s-3", "s-4"], str(filtered))
    check(
        "where 无匹配返回空",
        sq.query_clips(probe, top_k=3, where={"video_id": "nope"}) == [],
        "",
    )

    batch = sq.query_many("clip", [vec(0), vec(4)], top_k=2)
    check("批量查询两组", len(batch) == 2, str(len(batch)))
    check("批量首组首位", batch[0][0].id == "s-0", batch[0][0].id)
    check("批量次组首位", batch[1][0].id == "s-4", batch[1][0].id)

    removed = sq.delete_video("v2")
    check("按 video 删除计数", removed.get("clip") == 2, str(removed))
    check("删除后剩 3 条", sq.count("clip") == 3, str(sq.count("clip")))

    # 元数据往返：中文与嵌套结构不能丢。
    sq.upsert_clips(
        ids=["m-1"],
        embeddings=[vec(7)],
        documents=["中文文档"],
        metadatas=[
            {
                "video_id": "v9",
                "clip_id": "m-1",
                "nested": {"a": 1, "b": "中文"},
                "score": 0.5,
            }
        ],
    )
    meta = sq.query_clips(vec(7), top_k=1)[0].metadata
    check(
        "元数据中文往返",
        meta.get("clip_id") == "m-1" and meta.get("video_id") == "v9",
        str(meta),
    )
    check("元数据嵌套保留", "nested" in meta, str(meta))

    # 维度不一致必须报错（否则余弦算出来是垃圾）。
    try:
        sq.upsert_clips(
            ids=["bad"], embeddings=[[1.0, 0.0]], metadatas=[{"video_id": "v9"}]
        )
        check("维度不一致报错", False, "未抛异常")
    except VectorStoreError as exc:
        check("维度不一致报错", "64" in str(exc) and "2 维" in str(exc), str(exc)[:60])

    # 代切换：新代独立、可回退、可丢弃。
    child = sq.with_generation("abcd1234")
    check(
        "代后缀集合名",
        child.collection_name("clip").endswith("_abcd1234"),
        child.collection_name("clip"),
    )
    child.upsert_clips(ids=["g-1"], embeddings=[vec(8)], metadatas=[{"video_id": "v9"}])
    check(
        "新代条数独立",
        child.count("clip") == 1 and sq.count("clip") == 4,
        f"{child.count('clip')}/{sq.count('clip')}",
    )
    child.activate_generation("abcd1234")
    check("激活后走新代", child.count("clip") == 1, str(child.count("clip")))
    check(
        "丢弃旧代返回 2", child.drop_generation("") == 2, str(child.drop_generation(""))
    )

    # 持久化：重开文件仍是同一份数据（基础集合已被丢弃，故需带代读取）。
    sq.close()
    sq2 = VectorStore(sq.persist_dir or Path("."), backend="sqlite")
    check("重开后基础集合为空", sq2.count("clip") == 0, str(sq2.count("clip")))
    sq3 = sq2.with_generation("abcd1234")
    check("重开后新代条数不变", sq3.count("clip") == 1, str(sq3.count("clip")))
    sq2.close()

    try:
        VectorStore(Path(tempfile.mkdtemp()) / "x", backend="faiss")
        check("非法后端名报错", False, "未抛异常")
    except VectorStoreError as exc:
        check("非法后端名报错", "faiss" in str(exc), str(exc)[:60])

    print(f"\nPASS {PASSED}  FAIL {len(FAILED)}")
    for f in FAILED:
        print("  -", f)
    return 0 if not FAILED else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(1) from None
