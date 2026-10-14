"""核心业务回归：验证恢复、数据一致性、引用与离线降级，而非重复实现细节。"""

import json
import time
from pathlib import Path

import pytest

from videolens.config import Settings, load_settings
from videolens.errors import VideoLensError
from videolens.generation import Composer, fit_refs, validate_claims
from videolens.retrieval import fuse_rankings
from videolens.service import VideoService
from videolens.store import Database


class FakeGateway:
    def __init__(self, settings, getter):
        self.settings = settings

    def release(self):
        pass

    def generate(self, messages, check):
        check()
        payload = json.loads(messages[-1]["content"])
        ref = payload["evidence"][0]
        return json.dumps({"claims": [{"text": ref["text"], "refs": [ref["id"]]}]}, ensure_ascii=False)

    def stream(self, messages, check):
        for part in self.generate(messages, check):
            yield part

    def describe(self, frames, check):
        raise RuntimeError("没有视觉服务")

    def health(self):
        return {"mode": "test"}


@pytest.fixture
def service(tmp_path, monkeypatch):
    from videolens import service as module

    monkeypatch.setattr(module, "ModelGateway", FakeGateway)
    monkeypatch.setattr(module.media, "probe", lambda *a: {"duration_ms": 40000, "width": 1280, "height": 720, "has_audio": True, "subtitle_streams": []})
    monkeypatch.setattr(module.media, "find_subtitles", lambda *a: [{"id": "s1", "start_ms": 0, "end_ms": 20000, "text": "二叉树前序遍历先访问根节点。"}, {"id": "s2", "start_ms": 20000, "end_ms": 40000, "text": "队列采用先进先出的顺序。"}])

    def frames(video, clips, folder, settings, check):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "f000001.jpg"
        path.write_bytes(b"test frame")
        return [{"id": "f000001", "clip_id": clips[0]["id"], "timestamp_ms": 5000, "path": str(path), "width": 1280, "height": 720, "size_bytes": 10}]

    monkeypatch.setattr(module.media, "extract_frames", frames)

    def index_build(index, *args, **kwargs):
        index.pending_meta = None
        return ["测试使用关键词检索"]

    monkeypatch.setattr(module.IndexManager, "build", index_build)
    settings = Settings(data_dir=str(tmp_path / "data"), models_dir=str(tmp_path / "models"),
                        runtime_dir=str(tmp_path / "runtime"), config_path=str(tmp_path / "config.json"),
                        api_token="test-token", ui_password="test-token")
    result = VideoService(settings)
    yield result
    result.close()


def imported(service, tmp_path):
    source = tmp_path / "中文 & 测试.mp4"
    source.write_bytes(b"media-input")
    return service.import_file(source), source


def run_job(service, job):
    service.store.update_job(job["id"], status="running")
    service._run_task(job["id"])
    return service.get_task(job["id"])


def test_rrf_uses_rank_and_renormalizes_missing_route():
    result = fuse_rankings({"text": ["a", "b", "a"], "keyword": ["b"]}, {"text": 0.5, "image": 0.3, "keyword": 0.2}, 60)
    assert result[0][0] == "b"
    assert dict(result)["a"] == pytest.approx((0.5 / 0.7) / 61)


def test_citations_reject_unknown_and_fabricated_quote():
    refs = [{"id": "t:c1", "text": "根节点先被访问"}]
    with pytest.raises(ValueError):
        validate_claims({"claims": [{"text": "结论", "refs": ["nonexistent"]}]}, refs)
    with pytest.raises(ValueError):
        validate_claims({"claims": [{"text": "结论", "refs": [{"id": "t:c1", "quote": "叶节点先访问"}]}]}, refs)
    assert validate_claims({"claims": [{"text": "结论", "refs": ["t:c1"]}]}, refs)[0]["refs"][0]["text"] == "根节点先被访问"


def test_evidence_budget_is_bounded():
    refs = [{"id": str(i), "text": "中文证据" * 100} for i in range(10)]
    selected = fit_refs(refs, 3000)
    assert 0 < len(selected) < len(refs)
    assert sum(len(json.dumps(r, ensure_ascii=False).encode()) + 16 for r in selected) <= 3000


def test_history_summary_survives_full_recent_window():
    settings = Settings(recent_messages=2)
    composer = Composer(settings, None)
    history = [{"role": "system", "content": "之前讨论二叉树"}, {"role": "user", "content": "一"}, {"role": "assistant", "content": "二"}]
    messages, _ = composer._messages("问题", [], history, "")
    assert "之前讨论二叉树" in messages[-1]["content"]


def test_end_to_end_pipeline_with_injected_models(service, tmp_path):
    video, source = imported(service, tmp_path)
    job = run_job(service, service.create_task(video["id"]))
    assert job["status"] == "partial"
    details = service.get_video(video["id"])
    assert details["report"]["claims"]
    assert len(details["clips"]) >= 2
    assert "_path" not in json.dumps(details)
    assert str(tmp_path) not in json.dumps(details)
    hits = service.search(video["id"], "二叉树")
    assert hits and hits[0]["refs"][0]["source"] == "transcript"
    session = service.create_session(video["id"])
    result = service.ask(session["id"], "二叉树如何遍历？")
    assert result["mode"] == "generated" and result["refs"]
    events = list(service.ask_stream(session["id"], "队列顺序是什么？"))
    assert any(e["event"] == "delta" for e in events)
    assert events[-1]["event"] == "final"
    assert len(service.session_messages(session["id"])) == 4
    exported = json.loads(service.export(video["id"], "json"))
    assert exported["id"] == video["id"]
    assert "# 中文 & 测试.mp4" in service.export(video["id"], "markdown")
    service.delete_video(video["id"])
    assert source.exists(), "删除受管理副本不能删除用户原件"
    assert not service.store.get_session(session["id"])


def test_paused_job_cannot_be_duplicated(service, tmp_path):
    video, _ = imported(service, tmp_path)
    first = service.create_task(video["id"])
    service.control_task(first["id"], "pause")
    second = service.create_task(video["id"])
    assert second["id"] == first["id"]
    assert service.control_task(first["id"], "resume")["status"] == "queued"


def test_failed_reanalysis_preserves_old_report_and_evidence(service, tmp_path, monkeypatch):
    from videolens import service as module

    video, _ = imported(service, tmp_path)
    run_job(service, service.create_task(video["id"]))
    before = service.get_video(video["id"])
    monkeypatch.setattr(module.media, "find_subtitles", lambda *a: [{"id": "new", "start_ms": 0, "end_ms": 40000, "text": "新的内容"}])

    def fail(*args):
        raise RuntimeError("模拟抽帧失败")

    monkeypatch.setattr(module.media, "extract_frames", fail)
    service.settings.clip_target_seconds = 25
    after_job = run_job(service, service.create_task(video["id"]))
    assert after_job["status"] == "failed"
    after = service.get_video(video["id"])
    assert after["clips"] == before["clips"]
    assert after["frames"] == before["frames"]
    assert after["report"] == before["report"]


def test_queue_cancel_and_active_delete_guard(service, tmp_path):
    video, source = imported(service, tmp_path)
    job = service.create_task(video["id"])
    with pytest.raises(VideoLensError) as exc:
        service.delete_video(video["id"])
    assert exc.value.code == "TASK_ACTIVE"
    assert service.control_task(job["id"], "cancel")["status"] == "cancelled"
    service.delete_video(video["id"])
    assert source.is_file()


def test_restarted_queue_recovers_persisted_task(service, tmp_path):
    video, _ = imported(service, tmp_path)
    job = service.create_task(video["id"])
    service.store.update_job(job["id"], status="running")
    restarted = VideoService(service.settings)
    try:
        restarted.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and restarted.get_task(job["id"])["status"] in {"queued", "running"}:
            time.sleep(0.05)
        assert restarted.get_task(job["id"])["status"] == "partial"
    finally:
        restarted.close()


def test_settings_validate_and_never_serialize_secrets(tmp_path):
    settings = Settings(api_token="private-token", ui_password="secret")
    assert "private-token" not in settings.model_dump_json()
    with pytest.raises(ValueError):
        Settings(clip_min_seconds=30, clip_target_seconds=20)
    config = tmp_path / "local.json"
    config.write_text("{}", encoding="utf-8")
    a, b = load_settings(config), load_settings(config)
    assert a.api_token == b.api_token and len(a.api_token) >= 32


def test_database_checkpoint_and_events_survive_reopen(tmp_path):
    path = tmp_path / "store.sqlite"
    store = Database(path)
    store.checkpoint("j", "transcript", {"text": "中文"})
    first = store.event("j", "progress", {"stage": "transcript"})
    second = store.event("j", "complete", {"status": "partial"})
    reopened = Database(path)
    assert reopened.get_checkpoint("j", "transcript")["text"] == "中文"
    assert [event["id"] for event in reopened.events("j", first)] == [second]


def test_missing_frames_checkpoint_is_regenerated(service, tmp_path):
    video, _ = imported(service, tmp_path)
    job = service.create_task(video["id"])
    run_job(service, job)
    saved = service.store.get_checkpoint(job["id"], "frames")["value"]
    Path(saved[0]["path"]).unlink()
    service.store.update_job(job["id"], status="failed")
    service.control_task(job["id"], "resume")
    result = run_job(service, job)
    assert result["status"] == "partial"
    assert service.media_path(video["id"], saved[0]["id"]).is_file()


def test_cleanup_preserves_evidence_and_original(service, tmp_path):
    video, source = imported(service, tmp_path)
    run_job(service, service.create_task(video["id"]))
    stored = service.store.get_video(video["id"])
    stored["_media_created_at"] = time.time() - 8 * 86400
    service.store.save_video(stored)
    assert service.cleanup()["media"] == 1
    assert source.is_file()
    assert service.get_video(video["id"])["report"]
    with pytest.raises(VideoLensError) as exc:
        service.media_path(video["id"])
    assert exc.value.code == "MEDIA_NOT_AVAILABLE"


def test_question_receives_chapter_context_and_original_evidence(service, tmp_path, monkeypatch):
    video, _ = imported(service, tmp_path)
    run_job(service, service.create_task(video["id"]))
    captured = []
    original = service.gateway.generate

    def capture(messages, check):
        captured.append(json.loads(messages[-1]["content"]))
        return original(messages, check)

    monkeypatch.setattr(service.gateway, "generate", capture)
    session = service.create_session(video["id"])
    result = service.ask(session["id"], "二叉树如何遍历？")
    assert result["mode"] == "generated"
    assert captured[0]["chapter_summaries"]
    valid_ids = {ref["id"] for ref in captured[0]["evidence"]}
    assert all(set(chapter["source_refs"]) <= valid_ids for chapter in captured[0]["chapter_summaries"])


def test_cleanup_removes_unreferenced_generation_but_keeps_active_frames(service, tmp_path):
    video, source = imported(service, tmp_path)
    first = run_job(service, service.create_task(video["id"]))
    first_frame = service.media_path(video["id"], "f000001")
    service.settings.clip_target_seconds = 25
    second = run_job(service, service.create_task(video["id"]))
    current_frame = service.media_path(video["id"], "f000001")
    assert first["id"] != second["id"] and first_frame != current_frame
    # 模拟旧代已超过恢复保留期；新代仍受 SQLite 证据引用保护。
    old = service.store.get_job(first["id"])
    old["updated_at"] = time.time() - 25 * 3600
    service.store.save_job(old)
    service.cleanup()
    assert not first_frame.exists()
    assert current_frame.exists() and source.exists()


def test_model_revision_change_invalidates_resume_checkpoints(service, tmp_path):
    folder = Path(service.settings.models_dir)
    folder.mkdir()
    manifest = folder / "model-manifest.json"
    manifest.write_text(json.dumps({"models": {"text": {"repo_id": "BAAI/bge-m3", "revision": "version-one"}}}), encoding="utf-8")
    old_fingerprint = service.settings.fingerprint()
    video, _ = imported(service, tmp_path)
    job = service.create_task(video["id"])
    service.store.checkpoint(job["id"], "transcript", {"value": [{"id": "old", "start_ms": 0, "end_ms": 40000, "text": "旧模型残留"}]})
    manifest.write_text(json.dumps({"models": {"text": {"repo_id": "BAAI/bge-m3", "revision": "version-two"}}}), encoding="utf-8")
    assert service.settings.fingerprint() != old_fingerprint
    assert run_job(service, job)["status"] == "partial"
    assert all("旧模型残留" not in clip["text"] for clip in service.get_video(video["id"])["clips"])
    assert service.get_task(job["id"])["fingerprint"] == service.settings.fingerprint()


def test_settings_fingerprint_survives_database_json_roundtrip(tmp_path):
    original = Settings(models_dir=str(tmp_path / "models"))
    restored = Settings(**json.loads(original.model_dump_json()))
    assert original.fingerprint() == restored.fingerprint()


def test_pending_restart_settings_survive_later_prompt_edit(service):
    old_port = service.settings.port
    result = service.update_settings({"port": old_port + 1})
    assert result["restart_required"] and result["settings"]["port"] == old_port + 1
    assert service.settings.port == old_port, "当前监听在重启之前不能悄悄改变"
    result = service.update_settings({"answer_prompt": "仅根据原文简明回答"})
    saved = json.loads(Path(service.settings.config_path).read_text(encoding="utf-8"))
    assert saved["port"] == old_port + 1 and result["restart_required"]
    assert service.public_settings()["port"] == old_port + 1


def test_llama_launch_settings_require_restart(service):
    result = service.update_settings({"llama_gpu_layers": 99})
    assert result["restart_required"]
    assert service.settings.llama_gpu_layers == 0
    assert result["settings"]["llama_gpu_layers"] == 99
