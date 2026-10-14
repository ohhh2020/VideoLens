"""新版界面与本地仓库回归；假 GGUF 只用于文件管理，不冒充模型推理。"""
import hashlib
import logging
import os
import re
from pathlib import Path

import httpx
import pytest

from videolens.config import Settings
from videolens.errors import VideoLensError
from videolens.local_models import LocalModels
from videolens.logging_setup import ProjectFormatter, task_context
from videolens.service import VideoService


@pytest.fixture
def manager(tmp_path):
    service = VideoService(Settings(data_dir=str(tmp_path / "data"), models_dir=str(tmp_path / "models"),
                                   runtime_dir=str(tmp_path / "runtime"), config_path=str(tmp_path / "config.json")))
    result = LocalModels(service)
    yield result
    if result.thread:
        result.thread.join(5)
    service.close()


def fixture_model(tmp_path):
    path = tmp_path / "原始模型.gguf"
    path.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\0" * 64)
    return path


def import_one(manager, source):
    job = manager.submit("import", path=str(source))
    manager.thread.join(5)
    result = manager.jobs[job["id"]]
    assert result["status"] == "succeeded"
    return result["model_id"]


def test_log_format_task_thread_multiline_and_secret():
    formatter = ProjectFormatter()
    record = logging.LogRecord("test", logging.WARNING, "", 1, "第一行\nBearer private-secret", (), None)
    token = task_context.set("task-123")
    try:
        output = formatter.format(record)
    finally:
        task_context.reset(token)
    assert "private-secret" not in output
    assert all(re.match(r"\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\]\[WARNING\]\[task-123\]: ", line) for line in output.splitlines())
    assert "[MainThread]:" in formatter.format(record)


def test_legacy_login_fields_discarded():
    settings = Settings(ui_username="old", ui_password="secret")
    assert "ui_username" not in settings.model_dump()
    assert "secret" not in settings.model_dump_json()


def test_import_deduplicates_and_delete_preserves_source(manager, tmp_path):
    source = fixture_model(tmp_path)
    first = import_one(manager, source)
    assert import_one(manager, source) == first
    assert len(manager.listing()["models"]) == 1
    manager.delete(first)
    assert source.is_file()
    assert manager.listing()["models"] == []


def test_bad_model_never_published(manager, tmp_path):
    source = tmp_path / "错误.gguf"
    source.write_bytes(b"<html>this is not a model</html>")
    job = manager.submit("import", path=str(source))
    manager.thread.join(5)
    assert manager.jobs[job["id"]]["status"] == "failed"
    assert not list(manager.root.glob("*.gguf"))
    assert not list(manager.root.glob("*.part"))


def test_download_hash_mismatch_not_published(manager, monkeypatch):
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"GGUF\x03\0\0\0" + b"\0" * 24))))
    job = manager.submit("download", url="https://example.test/model.gguf", sha256="0" * 64)
    manager.thread.join(5)
    assert manager.jobs[job["id"]]["status"] == "failed"
    assert "SHA-256" in manager.jobs[job["id"]]["message"]
    assert manager.listing()["models"] == []


def test_delete_rejects_traversal_and_running_model(manager, tmp_path, monkeypatch):
    source = fixture_model(tmp_path)
    identity = import_one(manager, source)
    with pytest.raises(VideoLensError):
        manager.delete("../../原始模型")
    from videolens.config import atomic_json
    runtime = Path(manager.service.settings.runtime_dir)
    atomic_json(runtime / "llama-process.json", {"model": str(manager._path(identity))})
    monkeypatch.setattr("videolens.local_models.managed_process", lambda record: True)
    with pytest.raises(VideoLensError, match="先停止"):
        manager.delete(identity)
    assert manager._path(identity).exists()


@pytest.mark.skipif(os.name != "nt", reason="DPAPI 是 Windows 专用")
def test_secret_roundtrip_stays_in_project(tmp_path):
    from videolens.secrets import SecretStore
    vault = SecretStore(tmp_path / "secrets")
    vault.set("test", "private-test-value")
    assert vault.get("test") == "private-test-value"
    files = list((tmp_path / "secrets").glob("*.dpapi"))
    assert len(files) == 1 and b"private-test-value" not in files[0].read_bytes()


def test_cache_paths_are_project_local():
    from videolens.paths import configure_paths
    import tempfile
    root = configure_paths()
    for key in ("TEMP", "TMP", "HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "GRADIO_TEMP_DIR", "PIP_CACHE_DIR", "CUDA_CACHE_PATH"):
        assert Path(os.environ[key]).resolve().is_relative_to(root)
    assert Path(tempfile.gettempdir()).is_relative_to(root)


def test_successful_download_is_published(manager, monkeypatch):
    content = b"GGUF\x03\0\0\0" + b"\0" * 64
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=content))))
    job = manager.submit("download", url="https://example.test/model.gguf", sha256=hashlib.sha256(content).hexdigest())
    manager.thread.join(5)
    result = manager.jobs[job["id"]]
    assert result["status"] == "succeeded"
    assert manager._path(result["model_id"]).read_bytes() == content


@pytest.mark.parametrize("base,model,parameter,value", [
    ("https://api.openai.com/v1", "gpt-6-luna", "reasoning_effort", "none"),
    ("https://api.openai.com/v1", "gpt-5.4-mini", "reasoning_effort", "low"),
    ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus", "enable_thinking", False),
    ("https://api.deepseek.com", "deepseek-flash", "thinking", {"type": "disabled"}),
])
def test_cloud_parameter_compatibility(monkeypatch, base, model, parameter, value):
    """只验证出站协议参数；不使用模拟响应声称云端账号已经验收。"""
    import json
    from videolens.models import ModelGateway
    captured = []
    def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handle)))
    gateway = ModelGateway(Settings(), lambda ref: None)
    assert ''.join(gateway._http_stream((base, model, "fake-key", 3), [], lambda: None)) == "ok"
    assert captured[0][parameter] == value
    if "openai.com" in base:
        assert "max_completion_tokens" in captured[0] and "temperature" not in captured[0]
