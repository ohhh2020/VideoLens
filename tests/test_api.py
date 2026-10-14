"""接入层协议测试：替换业务服务，不把模拟测试算作真实模型验收。

本组专门验证 HTTP 鉴权、流式边界、浏览器媒体权限和失败响应。
真实 FFmpeg、模型和任务恢复由其它测试组负责。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from videolens.api import create_app, sse_event
from videolens.cli import read_sse
from videolens.config import Settings
from videolens.errors import VideoLensError


class StubService:
    def __init__(self, folder: Path):
        self.folder = folder
        self.started = self.closed = False
        self.video = {"id": "video-1", "title": "中文测试视频", "status": "ready", "clips": [], "frames": [], "report": {}}
        self.task = {"id": "task-1", "status": "partial_success", "stage": "report"}
        self.imports = []
        self.stream_failure = False
        self.stream_threads = []
        self.events = [{"id": 1, "event": "progress", "data": {"stage": "probe"}},
                       {"id": 2, "event": "progress", "data": {"stage": "report"}}]
        self.media = folder / "sample.mp4"
        self.media.write_bytes(b"sample-video-content")

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def list_videos(self):
        return [self.video]

    def get_video(self, video_id):
        if video_id != "video-1":
            raise VideoLensError("NOT_FOUND", "视频不存在", 404)
        return self.video

    def import_file(self, path, subtitle=None):
        self.imports.append({"path": path, "bytes": path.read_bytes(), "subtitle": subtitle.read_text(encoding="utf-8") if subtitle else None})
        return self.video

    def import_url(self, url):
        return self.video

    def get_task(self, task_id):
        if task_id != "task-1":
            raise VideoLensError("TASK_NOT_FOUND", "任务不存在", 404)
        return self.task

    def list_tasks(self):
        return [self.task]

    def create_task(self, video_id, kind):
        return {**self.task, "kind": kind}

    def control_task(self, task_id, action):
        self.task["status"] = {"pause": "paused", "resume": "queued", "cancel": "cancelled"}[action]
        return self.task

    def task_events(self, task_id, after=0):
        return [row for row in self.events if row["id"] > after]

    def session_messages(self, session_id):
        if session_id != "session-1":
            raise VideoLensError("SESSION_NOT_FOUND", "会话不存在", 404)
        return []

    def create_session(self, video_id):
        return {"id": "session-1", "video_id": video_id}

    def list_sessions(self, video_id):
        return [{"id": "session-1", "video_id": video_id}]

    def ask(self, session_id, question):
        return {"answer": "有证据的回答", "claims": [], "refs": [], "mode": "extractive"}

    def ask_stream(self, session_id, question):
        # 模拟真实服务的可重入推理锁，跨 yield 仍由取得它的同一线程释放。
        with threading.RLock():
            self.stream_threads.append(threading.get_ident())
            yield {"event": "status", "data": {"stage": "search"}}
            self.stream_threads.append(threading.get_ident())
            yield {"event": "delta", "data": {"text": "中文\n回答"}}
            if self.stream_failure:
                raise RuntimeError("SECRET_KEY_DONT_LEAK D:/private/video.mp4")
            yield {"event": "final", "data": self.ask(session_id, question)}
            self.stream_threads.append(threading.get_ident())

    def media_path(self, video_id, frame_id=None):
        self.get_video(video_id)
        return self.media

    def export(self, video_id, fmt):
        return json.dumps({"title": "中文报告"}, ensure_ascii=False) if fmt == "json" else "# 中文报告\n\n正文。"

    def public_settings(self):
        return {"cloud_enabled": False}

    def update_settings(self, patch):
        return patch

    def store_key(self, ref, key):
        self.last_key = (ref, key)

    def doctor(self):
        return {"status": "ok"}

    def search(self, video_id, query, image_path=None):
        return [{"clip_id": "clip-1", "text": query, "image": bool(image_path), "refs": []}]

    def check_provider(self, kind):
        return {"ok": False, "message": "尚未配置"}

    def cleanup(self):
        return {"removed": 0}

    def delete_video(self, video_id):
        return {"deleted": True}


@pytest.fixture
def setup(tmp_path):
    settings = Settings(data_dir=str(tmp_path), api_token="unit-test-token", ui_password="unit-test-token", rate_limit_per_minute=1000)
    service = StubService(tmp_path)
    return settings, service


@pytest.fixture
def client(setup):
    settings, service = setup
    with TestClient(create_app(settings=settings, service=service, with_ui=False),
                    headers={"Authorization": "Bearer unit-test-token"}, raise_server_exceptions=False) as current:
        yield current


def test_health_public_and_api_requires_token(setup):
    settings, service = setup
    with TestClient(create_app(settings, service, with_ui=False)) as client:
        assert service.started
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/api/v1/videos").status_code == 401
        assert client.get("/api/v1/videos", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/api/v1/videos", headers={"Authorization": "Bearer unit-test-token"}).status_code == 200
    assert service.closed


def test_upload_streams_and_removes_temporary_files(client, setup):
    settings, service = setup
    response = client.post("/api/v1/videos/upload", files={"file": ("中文 & 空格.mp4", b"video-data"),
                            "subtitle": ("字幕.srt", "1\n00:00:00,000 --> 00:00:01,000\n你好\n".encode())})
    assert response.status_code == 201
    assert service.imports[0]["bytes"] == b"video-data"
    assert service.imports[0]["path"].name == "中文 & 空格.mp4"
    assert "你好" in service.imports[0]["subtitle"]
    assert not service.imports[0]["path"].exists()
    assert not list((Path(settings.data_dir) / "tmp" / "uploads").iterdir())


def test_upload_rejects_empty_and_over_limit(client, setup):
    settings, service = setup
    settings.max_video_bytes = 5
    assert client.post("/api/v1/videos/upload", files={"file": ("empty.mp4", b"")}).status_code == 422
    assert client.post("/api/v1/videos/upload", files={"file": ("large.mp4", b"123456")}).status_code == 413
    assert not service.imports
    assert not list((Path(settings.data_dir) / "tmp" / "uploads").iterdir())


def test_local_import_denies_non_loopback(setup):
    settings, service = setup
    with TestClient(create_app(settings, service, with_ui=False), client=("198.51.100.10", 12345),
                    headers={"Authorization": "Bearer unit-test-token"}) as client:
        response = client.post("/api/v1/videos/import", json={"path": str(service.media)})
    assert response.status_code == 403
    assert not service.imports


def test_service_errors_are_structured_and_validation_does_not_echo_key(client):
    response = client.get("/api/v1/videos/missing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"
    response = client.put("/api/v1/keys/test", json={"key": "SECRET_VALUE", "typo": "SECRET_VALUE"})
    assert response.status_code == 422
    assert "SECRET_VALUE" not in response.text


def test_unexpected_errors_hide_private_data(client, setup):
    _, service = setup
    def fail():
        raise RuntimeError("SECRET_KEY_DONT_LEAK D:/private/video.mp4")
    service.doctor = fail
    response = client.get("/api/v1/doctor")
    assert response.status_code == 500
    assert "SECRET_KEY" not in response.text
    assert "private" not in response.text


def test_task_events_resume_at_last_event_and_have_terminal(client):
    response = client.get("/api/v1/tasks/task-1/events?after=0", headers={"Last-Event-ID": "1"})
    assert response.status_code == 200
    assert "id: 1\n" not in response.text
    assert "id: 2\n" in response.text
    events = list(read_sse(response.text.splitlines()))
    assert [event["event"] for event in events] == ["progress", "done"]
    assert events[-1]["data"]["status"] == "partial_success"
    assert client.get("/api/v1/tasks/task-1/events", headers={"Last-Event-ID": "bad"}).status_code == 422


def test_paused_task_ends_event_stream(client, setup):
    setup[1].task["status"] = "paused"
    response = client.get("/api/v1/tasks/task-1/events")
    assert "event: done" in response.text
    assert '"status": "paused"' in response.text


def test_ask_stream_final_and_failed_stream_distinguishable(client, setup):
    response = client.post("/api/v1/sessions/session-1/ask", json={"question": "问题", "stream": True})
    events = list(read_sse(response.text.splitlines()))
    assert [event["event"] for event in events] == ["status", "delta", "final"]
    assert events[1]["data"]["text"] == "中文\n回答"
    assert len(set(setup[1].stream_threads)) == 1
    setup[1].stream_failure = True
    response = client.post("/api/v1/sessions/session-1/ask", json={"question": "问题", "stream": True})
    assert "event: error" in response.text
    assert "event: final" not in response.text
    assert "SECRET_KEY" not in response.text
    assert client.post("/api/v1/sessions/missing/ask", json={"question": "问题", "stream": True}).status_code == 404


def test_sse_json_multiline_and_heartbeats():
    encoded = sse_event("delta", {"text": "第一行\n第二行"}, 7)
    assert list(read_sse(encoded.splitlines())) == [{"event": "delta", "data": {"text": "第一行\n第二行"}}]
    lines = [": 保持连接", "", "event: final", 'data: {"answer":', 'data: "你好"}', ""]
    assert list(read_sse(lines)) == [{"event": "final", "data": {"answer": "你好"}}]
    assert list(read_sse(["event: final", 'data: {"answer":"未完成"}'])) == []


def test_export_utf8_format_and_content_disposition(client):
    response = client.get("/api/v1/videos/video-1/export?format=markdown")
    assert response.text.startswith("# 中文报告")
    assert "attachment" in response.headers["content-disposition"]
    assert client.get("/api/v1/videos/video-1/export?format=exe").status_code == 422


def test_api_media_supports_range(client):
    response = client.get("/api/v1/videos/video-1/media", headers={"Range": "bytes=0-5"})
    assert response.status_code == 206
    assert response.content == b"sample"


def test_write_rate_limit_does_not_block_progress_reads(setup):
    settings, service = setup
    settings.rate_limit_per_minute = 1
    with TestClient(create_app(settings, service, with_ui=False), headers={"Authorization": "Bearer unit-test-token"}) as client:
        assert client.post("/api/v1/cleanup").status_code == 200
        assert client.post("/api/v1/cleanup").status_code == 429
        assert client.get("/api/v1/tasks/task-1").status_code == 200


def test_gradio_needs_no_login_but_api_keeps_token(setup, monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    pytest.importorskip("gradio")
    settings, service = setup
    with TestClient(create_app(settings, service, with_ui=True), raise_server_exceptions=True) as client:
        assert client.get("/app/").status_code == 200
        assert client.get("/app/config").json().get("auth_required") is not True
        config = client.get("/app/config").json()
        labels = [item.get("props", {}).get("label", "") for item in config["components"]]
        assert "可选字幕（SRT / VTT / ASS）" not in labels
        assert "本地模型" in labels and "问答提示词" in labels
        assert client.get("/app/videolens-media/video-1").content == b"sample-video-content"
        assert "中文报告" in client.get("/app/videolens-export/video-1/markdown").text
        # 免登录网页不改变内部 API 的 Bearer 鉴权。
        assert client.get("/api/v1/videos").status_code == 401
