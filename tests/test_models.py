"""模型适配边界：异常传播与云端协议。"""

import sys, unittest, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.models.embed import EmbeddingService, EmbeddingError
from backend.models.asr import AsrConfig, AsrEngine, AsrError
from backend.models.llm import LlmGateway, LlmMessage, LlmError, OpenAICompatibleBackend
from backend.models.vl import VlService, VlAuthenticationError, split_description
from backend.config import load_app_config


class TextEncoder:
    name = "fake-text"
    dim = 4

    def embed_texts(self, items):
        return [[1.0, 0.0, 0.0, 0.0] for _ in items]


class ImageEncoder:
    name = "fake-image"
    dim = 3

    def embed_texts(self, items):
        return [[0.0, 1.0, 0.0] for _ in items]

    def embed_images(self, items):
        return [[0.0, 1.0, 0.0] for _ in items]


class ModelTests(unittest.TestCase):
    def test_text_and_image_spaces(self):
        svc = EmbeddingService(TextEncoder(), ImageEncoder())
        self.assertEqual(len(svc.embed_query("文本")), 4)
        self.assertEqual(len(svc.embed_cross_modal("图片")), 3)
        self.assertEqual(len(svc.embed_image(object())), 3)
        self.assertFalse(svc.degraded)

    def test_encode_error_propagates(self):
        bad = TextEncoder()
        bad.embed_texts = lambda items: (_ for _ in ()).throw(EmbeddingError("损坏"))
        with self.assertRaises(EmbeddingError):
            EmbeddingService(bad, ImageEncoder()).embed_query("文本")

    def test_dimension_mismatch_rejected(self):
        bad = TextEncoder()
        bad.dim = 8
        with self.assertRaises(EmbeddingError):
            EmbeddingService(bad, ImageEncoder()).embed_query("文本")

    def test_asr_does_not_switch_explicit_device(self):
        engine = AsrEngine(AsrConfig(device="cuda", compute_type="float16"))
        self.assertEqual(engine._device_plan(), [("cuda", "float16")])

    def test_asr_load_error_not_empty_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audio.wav"
            path.write_bytes(b"fixture")
            engine = AsrEngine(AsrConfig(device="cpu"))
            with patch.object(engine, "load", side_effect=AsrError("模型损坏")):
                with self.assertRaises(AsrError):
                    engine.transcribe(path, video_id="vtest", allow_subtitle=False)

    def test_lazy_asr_inference_error_propagates(self):
        def segments():
            raise RuntimeError("推理失败")
            yield

        engine = AsrEngine(AsrConfig(device="cpu"))
        engine._loaded_key = ("fixture", "cpu", "int8")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audio.wav"
            path.write_bytes(b"fixture")
            with patch.object(
                engine,
                "load",
                return_value=SimpleNamespace(
                    transcribe=lambda *a, **kw: (segments(), SimpleNamespace())
                ),
            ):
                with self.assertRaises(AsrError):
                    engine.transcribe(path, video_id="vtest", allow_subtitle=False)

    def test_cloud_only_gateway(self):
        with self.assertRaises(LlmError):
            LlmGateway([])
        cfg = load_app_config(
            overrides={"llm.cloud.api_key": "", "llm.cloud.api_key_env": ""}
        )
        with self.assertRaises(LlmError):
            LlmGateway.from_config(cfg)
        from backend.models import llm

        self.assertFalse(hasattr(llm, "LlamaCppChatBackend"))
        self.assertFalse(hasattr(llm, "KeywordBackend"))

    def test_non_thinking_cloud_payload(self):
        import json

        b = OpenAICompatibleBackend(
            "https://invalid.test/v1", "qwen3.5-flash", "sk-fixture"
        )
        data = json.loads(
            b._payload([LlmMessage("user", "测试")], 200, 0.1, (), stream=False)
        )
        self.assertFalse(data["enable_thinking"])

    def test_vision_cloud_only(self):
        cfg = load_app_config(
            overrides={"vl.cloud_api_key": "", "vl.cloud_api_key_env": ""}
        )
        with self.assertRaises(VlAuthenticationError):
            VlService.from_config(cfg)
        from backend.models import vl

        self.assertFalse(hasattr(vl, "QwenVlRunner"))

    def test_ocr_parse(self):
        desc, ocr = split_description("红色标题。\n可读文字：测试")
        self.assertIn("红色", desc)
        self.assertIn("测试", ocr)


if __name__ == "__main__":
    unittest.main()
