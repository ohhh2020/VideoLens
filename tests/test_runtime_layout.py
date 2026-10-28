"""强制模型策略、外置路径与严格启动的回归验证。"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import load_app_config, ConfigValueError, UnknownConfigKeyError
from backend.runtime import runtime_root
from backend.models.required import REQUIRED_MODELS, verify_required_models
from backend.models.manager import ModelManager, ConfigError
from backend.models.embed import EmbeddingError, EmbeddingService
from backend.pipeline.factory import build_orchestrator


class RuntimeTests(unittest.TestCase):
    def test_paths_are_outside_source(self):
        cfg = load_app_config(ignore_user_settings=True)
        self.assertFalse(runtime_root().is_relative_to(cfg.root_dir))
        for path in cfg.all_paths().values():
            self.assertTrue(path.is_relative_to(runtime_root()))

    def test_exact_local_allowlist(self):
        manager = ModelManager(allow_download=False)
        self.assertEqual(set(manager.list_models()), set(REQUIRED_MODELS))
        self.assertTrue(all(manager.get_spec(n).required for n in REQUIRED_MODELS))

    def test_custom_fourth_model_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "models.yaml"
            p.write_text(
                "models:\n  forbidden-model:\n    kind: llm\n", encoding="utf-8"
            )
            with self.assertRaises(ConfigError):
                ModelManager(config_path=p)

    def test_missing_model_blocks_startup(self):
        with patch(
            "backend.models.required.verify_required_models",
            side_effect=RuntimeError("必需模型缺失"),
        ):
            with self.assertRaisesRegex(RuntimeError, "必需模型缺失"):
                build_orchestrator(load_app_config())

    def test_each_required_model_missing_fails(self):
        for missing in REQUIRED_MODELS:
            manager = SimpleNamespace(
                status=lambda name: SimpleNamespace(
                    available=name != missing, state=SimpleNamespace(value="MISSING")
                )
            )
            with self.assertRaisesRegex(RuntimeError, missing):
                verify_required_models(manager)

    def test_cloud_cannot_be_disabled(self):
        for values in (
            {"llm.mode": "local"},
            {"vl.cloud_enabled": False},
            {"asr.model": "other"},
        ):
            with self.assertRaises(ConfigValueError):
                load_app_config(overrides=values)

    def test_no_local_llm_config(self):
        with self.assertRaises(UnknownConfigKeyError):
            load_app_config(overrides={"llm.local.model": "anything"})

    def test_no_hash_fallback(self):
        from backend.models import embed

        self.assertFalse(hasattr(embed, "HashingEmbedder"))
        with self.assertRaises(EmbeddingError):
            EmbeddingService(None, None)
        with self.assertRaises(EmbeddingError):
            EmbeddingService.from_config(manager=None)

    def test_startup_failure_propagates(self):
        from fastapi.testclient import TestClient
        from backend.web.main import create_app

        with (
            patch(
                "backend.web.main.Orchestrator.from_config",
                side_effect=RuntimeError("必需模型损坏"),
            ),
            patch(
                "backend.media.ffmpeg_utils.resolve_binary",
                return_value="manual-ffmpeg.exe",
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "必需模型损坏"):
                with TestClient(create_app()):
                    pass

    def test_failed_report_is_failed_job(self):
        from backend.web.jobs import JobState
        from backend.pipeline.domain import AnalysisReport

        job = JobState("t-test-failure")
        job.finish(AnalysisReport(status="failed", errors=["模型推理失败"]))
        self.assertEqual(job.status, "failed")
        self.assertIn("模型推理失败", job.error)

    def test_storage_factories_use_runtime_paths(self):
        from backend.storage.repository import SqliteRepository
        from backend.rag.vector_store import VectorStore

        cfg = load_app_config()
        with patch(
            "backend.storage.repository.SqliteRepository.__init__", return_value=None
        ) as init:
            SqliteRepository.from_config(cfg)
            self.assertEqual(init.call_args.args[0], cfg.path("sqlite"))
        with patch(
            "backend.rag.vector_store.VectorStore.__init__", return_value=None
        ) as init:
            VectorStore.from_config(cfg)
            self.assertEqual(init.call_args.args[0], cfg.path("chroma"))


if __name__ == "__main__":
    unittest.main()
