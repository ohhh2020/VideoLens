"""验证重建索引失败时保留旧索引，成功后再切换活动代。"""

from pathlib import Path
from types import SimpleNamespace as Obj
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reindex_vectors import reindex_video


class Repo:
    def __init__(self, frame):
        self.generation = "old"
        self.frame = frame
        self.payload = {"summary": "保留已有总结", "vector_count": 2}

    def get_video(self, _):
        return Obj(vector_generation=self.generation)

    def get_clips(self, _):
        return [
            Obj(
                clip_id="c1",
                text="原转写",
                core_text="原转写",
                index_no=0,
                start_ms=0,
                end_ms=1000,
                core_start_ms=0,
                core_end_ms=1000,
            )
        ]

    def get_keyframes(self, _):
        return [
            Obj(
                keyframe_id="f1",
                clip_id="c1",
                timestamp_ms=500,
                path=str(self.frame),
                description="原画面描述",
            )
        ]

    def update_video_meta(self, _, *, vector_generation):
        self.generation = vector_generation

    def get_video_report(self, _):
        return dict(self.payload)

    def set_video_report(self, _, report):
        self.payload = report


class Vectors:
    def __init__(self, fail=False):
        self.fail = fail
        self.dropped = []
        self.children = []

    def with_generation(self, name):
        child = Stage(self.fail)
        self.children.append((name, child))
        return child

    def drop_generation(self, name):
        self.dropped.append(name)


class Stage:
    def __init__(self, fail):
        self.fail = fail
        self.counts = {"clip": 0, "frame": 0}

    def upsert_clips(self, **data):
        self.counts["clip"] = len(data["ids"])

    def upsert_frames(self, **data):
        if self.fail:
            raise RuntimeError("模拟写图像集合失败")
        self.counts["frame"] = len(data["ids"])

    def count(self, kind):
        return self.counts[kind]


class Encoder:
    degraded = False
    text_model_name = "bge-m3"
    image_model_name = "siglip2-base"

    def embed_texts(self, texts):
        return [[1.0] for _ in texts]

    def embed_images(self, images):
        return [[1.0] for _ in images]


with tempfile.TemporaryDirectory() as tmp:
    frame = Path(tmp) / "frame.jpg"
    frame.write_bytes(b"fixture")
    # 写第二个集合失败：新代被清理，旧代不受影响。
    repo = Repo(frame)
    store = Vectors(fail=True)
    try:
        reindex_video(repo, store, Encoder(), "v1")
    except RuntimeError:
        pass
    else:
        raise AssertionError("写入失败应报错")
    assert repo.generation == "old"
    assert "old" not in store.dropped and store.children[0][0] in store.dropped
    assert repo.payload["summary"] == "保留已有总结"

    # 模型在编码阶段失败：不会建立新索引。
    class Broken(Encoder):
        def embed_images(self, images):
            raise RuntimeError("模型推理失败")

    store = Vectors()
    try:
        reindex_video(repo, store, Broken(), "v1")
    except RuntimeError:
        pass
    else:
        raise AssertionError("推理失败应报错")
    assert repo.generation == "old" and not store.children
    # 正常重建：指向新代，保存模型标记和原总结，最后删除旧代。
    store = Vectors()
    result = reindex_video(repo, store, Encoder(), "v1")
    assert repo.generation == result["新索引代"] != "old"
    assert store.dropped == ["old"]
    assert repo.payload["summary"] == "保留已有总结"
    assert repo.payload["profile"]["text_encoder"] == "bge-m3"
    assert result["文本向量"] == 1 and result["图像向量"] == 1
    # 降级编码器不能覆盖真实索引。
    degraded = Encoder()
    degraded.degraded = True
    try:
        reindex_video(repo, store, degraded, "v1")
    except RuntimeError:
        pass
    else:
        raise AssertionError("应拒绝哈希降级重建")
print("PASS：写入失败回退、推理失败无写入、发布后切代、保留已有内容、拒绝降级编码器。")
