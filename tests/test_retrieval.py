"""使用真实 Chroma/SQLite 验证索引事务边界；仅向量数值由确定性桩提供。

这些测试不验证神经网络语义质量，重点证明暂停、部分失败和删除不会发布半个索引。
"""
from __future__ import annotations

import math

import pytest

from videolens.config import Settings
from videolens.errors import TaskInterrupted
from videolens.retrieval import IndexManager
from videolens.store import Database


class FakeEmbeddingGateway:
    """避免联网和大模型依赖，故障注入仍经过真实 Chroma 写入路径。"""

    def __init__(self):
        self.fail_route = None
        self.cancel_route = None

    def _run(self, route, values, check):
        check()
        if route == self.cancel_route:
            raise TaskInterrupted("pause")
        if route == self.fail_route:
            raise RuntimeError("注入模型故障")
        vectors = []
        for value in values:
            raw = [float(len(value) % 13 + 1), 2.0, 3.0]
            norm = math.sqrt(sum(x * x for x in raw))
            vectors.append([x / norm for x in raw])
        return vectors

    def embed_texts(self, values, check):
        return self._run("text", values, check)

    def embed_images(self, values, check):
        return self._run("image", values, check)

    def release(self):
        pass


@pytest.fixture
def index_fixture(tmp_path):
    pytest.importorskip("chromadb", reason="真实 Chroma 依赖未安装，不能把桩当作存储验证")
    settings = Settings(data_dir=str(tmp_path), device="cpu")
    store = Database(tmp_path / "db.sqlite3")
    gateway = FakeEmbeddingGateway()
    manager = IndexManager(settings, store, gateway)
    video = {"id": "integration_video", "title": "索引测试", "status": "ready", "created_at": 0}
    clips = [{"id": "c000001", "start_ms": 0, "end_ms": 10000, "text": "红色苹果", "segment_ids": ["s000001"]}]
    frames = [{"id": "f000001", "clip_id": "c000001", "timestamp_ms": 2000, "path": str(tmp_path / "unused.jpg"),
               "description": "红色苹果", "ocr": "", "width": 20, "height": 20, "size_bytes": 10}]
    store.save_video(video)
    store.replace_clips(video["id"], clips)
    store.replace_frames(video["id"], frames)
    return manager, gateway, store, video, clips, frames


def collection_names(manager):
    return {getattr(item, "name", item) for item in manager.client().list_collections()}


def test_pending_index_remains_invisible_until_sqlite_publish(index_fixture):
    manager, gateway, store, video, clips, frames = index_fixture
    assert manager.build(video["id"], clips, frames, lambda: None) == []
    old = store.get_meta("index:" + video["id"])
    revised = [{**clips[0], "text": "蓝色天空"}]
    assert manager.build(video["id"], revised, frames, lambda: None, activate=False) == []
    pending = manager.pending_meta
    assert pending["generation"] != old["generation"]
    assert store.get_meta("index:" + video["id"]) == old
    assert store.clips(video["id"])[0]["text"] == "红色苹果"
    store.commit_analysis({**video, "report": {"summary": "新报告"}}, revised, frames, pending)
    assert store.get_meta("index:" + video["id"]) == pending
    assert store.clips(video["id"])[0]["text"] == "蓝色天空"
    text_collection = manager.client().get_collection(pending["collections"]["text"], embedding_function=None)
    assert "蓝色天空" in text_collection.get()["documents"][0]


@pytest.mark.parametrize("route", ["text", "image"])
def test_pause_cleans_every_unpublished_collection(index_fixture, route):
    manager, gateway, store, video, clips, frames = index_fixture
    manager.build(video["id"], clips, frames, lambda: None)
    old = store.get_meta("index:" + video["id"])
    before = collection_names(manager)
    gateway.cancel_route = route
    with pytest.raises(TaskInterrupted):
        manager.build(video["id"], clips, frames, lambda: None, activate=False)
    assert collection_names(manager) == before
    assert store.get_meta("index:" + video["id"]) == old


def test_failed_rebuild_preserves_previous_complete_index(index_fixture):
    manager, gateway, store, video, clips, frames = index_fixture
    manager.build(video["id"], clips, frames, lambda: None)
    old = store.get_meta("index:" + video["id"])
    before = collection_names(manager)
    gateway.fail_route = "image"
    warnings = manager.build(video["id"], clips, frames, lambda: None)
    assert warnings
    assert store.get_meta("index:" + video["id"]) == old
    assert collection_names(manager) == before


def test_delete_removes_all_video_generations_and_is_retryable(index_fixture):
    manager, gateway, store, video, clips, frames = index_fixture
    manager.build(video["id"], clips, frames, lambda: None)
    manager.build(video["id"], clips, frames, lambda: None, activate=False)
    assert len(collection_names(manager)) == 4
    manager.delete(video["id"])
    assert collection_names(manager) == set()
    manager.delete(video["id"])


def test_new_analysis_without_vectors_removes_old_pointer(index_fixture):
    manager, gateway, store, video, clips, frames = index_fixture
    manager.build(video["id"], clips, frames, lambda: None)
    assert store.get_meta("index:" + video["id"])
    store.commit_analysis(video, [{**clips[0], "text": "新的文字证据"}], frames, None)
    assert store.get_meta("index:" + video["id"]) is None
    assert store.clips(video["id"])[0]["text"] == "新的文字证据"
