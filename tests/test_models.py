"""模型边界测试：本文件验证协议与控制逻辑，不声称验证真实神经网络质量。"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from videolens.models import ModelGateway, ModelUnavailable


def settings(tmp_path, **changes):
    def provider():
        return SimpleNamespace(enabled=False, base_url="", model="", key_ref="", timeout_seconds=1, context_tokens=8192)
    values = dict(models_dir=str(tmp_path), allow_model_download=False, device="cpu", cloud_enabled=False,
                  local_enabled=False, local_base_url="http://127.0.0.1:8081/v1", local_model="test", text_provider=provider(),
                  vision_provider=provider(), asr_model="Systran/faster-whisper-small", text_embedding_model="BAAI/bge-m3",
                  image_embedding_model="google/siglip2-base-patch16-224")
    values.update(changes)
    return SimpleNamespace(**values)


def test_cloud_disabled_never_reads_secret_or_calls_network(tmp_path, monkeypatch):
    cfg = settings(tmp_path)
    cfg.text_provider.enabled = True
    cfg.text_provider.base_url = "https://example.invalid/v1"
    cfg.text_provider.model = "test"
    gateway = ModelGateway(cfg, lambda _: pytest.fail("离线模式不应读取云端密钥"))
    monkeypatch.setattr(gateway, "_http_stream", lambda *args: pytest.fail("离线模式不应调用云端 HTTP"))
    with pytest.raises(ModelUnavailable):
        gateway.generate([{"role": "user", "content": "测试"}])


def test_remote_local_address_cannot_bypass_offline_switch(tmp_path):
    gateway = ModelGateway(settings(tmp_path, local_enabled=True, local_base_url="https://example.invalid/v1"), lambda _: None)
    with pytest.raises(ModelUnavailable):
        gateway._local_provider()


def test_fixed_manifest_snapshot_is_used(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    manifest = {"schema": 1, "models": {"text": {"repo_id": "BAAI/bge-m3", "revision": "fixed", "snapshot": str(snapshot)}}}
    (tmp_path / "model-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    gateway = ModelGateway(settings(tmp_path), lambda _: None)
    assert gateway._source("text", "BAAI/bge-m3") == str(snapshot)


def test_pcm_wav_decoding_avoids_upstream_pyav_metadata_errors(tmp_path):
    import struct
    import wave
    path = tmp_path / "规范音频.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(struct.pack("<hhh", -32768, 0, 32767))
    samples = ModelGateway._decode_audio(path, lambda: None)
    assert str(samples.dtype) == "float32"
    assert samples.tolist() == [-1.0, 0.0, 32767 / 32768]


def test_asr_gpu_selection_is_independent_from_torch_cpu_build(tmp_path, monkeypatch):
    import sys
    gateway = ModelGateway(settings(tmp_path, device="auto"), lambda _: None)
    monkeypatch.setattr(gateway, "_preferred_device", lambda: "cpu")
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 1))
    assert gateway._asr_devices() == ["cuda", "cpu"]


def test_modern_pyav_resamples_stereo_audio(tmp_path):
    pytest.importorskip("av", reason="该兼容测试需要 faster-whisper 的 PyAV 依赖")
    import array
    import wave
    path = tmp_path / "立体声.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(array.array("h", [1000, -500] * 8000).tobytes())
    samples = ModelGateway._decode_audio(path, lambda: None)
    assert samples.ndim == 1 and str(samples.dtype) == "float32"
    assert 15998 <= len(samples) <= 16002


def test_cancel_is_not_converted_to_cpu_fallback(tmp_path, monkeypatch):
    class Cancelled(RuntimeError):
        pass
    gateway = ModelGateway(settings(tmp_path), lambda _: None)
    monkeypatch.setattr(gateway, "_devices", lambda: ["cuda", "cpu"])
    monkeypatch.setattr(gateway, "_load", lambda *args: pytest.fail("取消后不得加载模型"))
    def cancel():
        raise Cancelled("停止")
    with pytest.raises(Cancelled):
        gateway.embed_texts(["取消"], cancel)


def test_gpu_model_failure_uses_cpu_and_normalizes(tmp_path, monkeypatch):
    calls = []
    gateway = ModelGateway(settings(tmp_path), lambda _: None)
    monkeypatch.setattr(gateway, "_devices", lambda: ["cuda", "cpu"])
    class Encoded:
        def tolist(self):
            return [[0.6, 0.8]]
    class Model:
        def encode(self, texts, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            return Encoded()
    def load(role, device):
        calls.append(device)
        if device == "cuda":
            raise RuntimeError("GPU 不兼容")
        gateway._model = Model()
    monkeypatch.setattr(gateway, "_load", load)
    assert gateway.embed_texts(["证据"]) == [[0.6, 0.8]]
    assert calls == ["cuda", "cpu"]


@pytest.fixture
def model_server():
    server_state = {"chunks": [], "delay": 0, "requests": []}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            size = int(self.headers["Content-Length"])
            server_state["requests"].append(json.loads(self.rfile.read(size)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for chunk in server_state["chunks"]:
                    time.sleep(server_state["delay"])
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", server_state
    server.shutdown()
    server.server_close()


def test_real_http_sse_chunk_boundaries(tmp_path, model_server):
    url, state = model_server
    wire = ('data: ' + json.dumps({"choices": [{"delta": {"content": "中文证据"}}]}, ensure_ascii=False) + '\n\ndata: [DONE]\n\n').encode("utf-8")
    state["chunks"] = [wire[i:i + 3] for i in range(0, len(wire), 3)]
    gateway = ModelGateway(settings(tmp_path, local_enabled=True, local_base_url=url), lambda _: None)
    assert gateway.generate([{"role": "user", "content": "你好"}]) == "中文证据"
    assert state["requests"][0]["messages"][0]["content"] == "你好"


def test_incomplete_sse_is_failure_not_success(tmp_path, model_server):
    url, state = model_server
    state["chunks"] = [b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n']
    gateway = ModelGateway(settings(tmp_path, local_enabled=True, local_base_url=url), lambda _: None)
    with pytest.raises(ModelUnavailable, match="中途断开"):
        gateway.generate([{"role": "user", "content": "测试"}])


def test_cancel_interrupts_wait_for_first_model_token(tmp_path, model_server):
    class Cancelled(RuntimeError):
        pass
    url, state = model_server
    state["chunks"] = [b'data: [DONE]\n\n']
    state["delay"] = 3
    gateway = ModelGateway(settings(tmp_path, local_enabled=True, local_base_url=url), lambda _: None)
    started = time.monotonic()
    def check():
        if time.monotonic() - started > 0.3:
            raise Cancelled()
    with pytest.raises(Cancelled):
        gateway.generate([{"role": "user", "content": "测试"}], check)
    assert time.monotonic() - started < 2


def test_cloud_failure_before_output_falls_back_locally(tmp_path, monkeypatch):
    cfg = settings(tmp_path, cloud_enabled=True, local_enabled=True)
    cfg.text_provider.enabled, cfg.text_provider.base_url, cfg.text_provider.model = True, "https://cloud.invalid/v1", "cloud"
    gateway = ModelGateway(cfg, lambda _: None)
    calls = []
    def stream(provider, messages, check):
        calls.append(provider[0])
        if "cloud.invalid" in provider[0]:
            raise ModelUnavailable("模拟服务不可用")
        yield "本地结果"
    monkeypatch.setattr(gateway, "_http_stream", stream)
    assert gateway.generate([]) == "本地结果"
    assert len(calls) == 2


def test_partial_cloud_failure_does_not_mix_local_answer(tmp_path, monkeypatch):
    cfg = settings(tmp_path, cloud_enabled=True, local_enabled=True)
    cfg.text_provider.enabled, cfg.text_provider.base_url, cfg.text_provider.model = True, "https://cloud.invalid/v1", "cloud"
    gateway = ModelGateway(cfg, lambda _: None)
    calls = []
    def stream(provider, messages, check):
        calls.append(provider[0])
        yield "未完"
        raise ModelUnavailable("断流")
    monkeypatch.setattr(gateway, "_http_stream", stream)
    with pytest.raises(ModelUnavailable):
        gateway.generate([])
    assert len(calls) == 1
