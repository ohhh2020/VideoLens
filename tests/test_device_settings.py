"""独立设备路由、设置持久化、实际设备展示和 GPU 并发边界。"""

from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import load_app_config, ConfigValueError
from backend.models.asr import AsrConfig, AsrEngine
from backend.models.embed import EmbeddingService
from backend.models._runtime import gpu_slot
from backend.models.devices import effective_settings, active_devices
from backend.user_settings import (
    save_settings,
    settings_file,
    validate_settings,
    SettingsError,
)


class DeviceSettingsTests(unittest.TestCase):
    def config(self, **overrides):
        return load_app_config(
            root=ROOT, ignore_user_settings=True, overrides=overrides
        )

    def test_all_eight_independent_device_combinations(self):
        manager = SimpleNamespace(ensure_model=lambda name: Path(name))
        for asr, text, image in product(("cpu", "cuda"), repeat=3):
            cfg = self.config(
                **{
                    "asr.device": asr,
                    "embedding.text_device": text,
                    "embedding.image_device": image,
                }
            )
            with (
                patch("backend.models.embed.BgeM3TextEmbedder") as bge,
                patch("backend.models.embed.SigLip2ImageEmbedder") as siglip,
            ):
                EmbeddingService.from_config(cfg, manager=manager)
                self.assertEqual(bge.call_args.kwargs["device"], text)
                self.assertEqual(siglip.call_args.kwargs["device"], image)
                self.assertEqual(AsrConfig.from_config(cfg).device, asr)
                self.assertEqual(
                    bge.call_args.kwargs["batch_size"], 4 if text == "cuda" else 16
                )
                self.assertEqual(
                    siglip.call_args.kwargs["batch_size"], 4 if image == "cuda" else 16
                )

    def test_legacy_shared_device_remains_compatible(self):
        cfg = self.config(**{"models.device": "cuda"})
        values = effective_settings(cfg, {"torch_cuda": True})
        self.assertEqual(values["embedding.text_device"], "cuda")
        self.assertEqual(values["embedding.image_device"], "cuda")
        self.assertEqual(values["asr.device"], "cpu")

    def test_save_preferences_preserves_keys_and_other_devices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_settings(
                {
                    "llm.cloud.api_key": "sk-complete-test-key-for-preservation",
                    "embedding.image_device": "cpu",
                },
                root=root,
            )
            save_settings(
                {
                    "asr.device": "cuda",
                    "asr.compute_type": "float16",
                    "embedding.text_device": "cuda",
                    "ui.report_mode": "detailed",
                },
                root=root,
            )
            raw = json.loads(settings_file(root).read_text(encoding="utf-8"))
            self.assertEqual(
                raw["llm.cloud.api_key"], "sk-complete-test-key-for-preservation"
            )
            cfg = load_app_config(root=root)
            self.assertEqual(cfg.embedding.image_device, "cpu")
            self.assertEqual(cfg.embedding.text_device, "cuda")
            self.assertEqual(cfg.ui.report_mode, "detailed")
            self.assertEqual(
                AsrEngine(AsrConfig.from_config(cfg))._device_plan(),
                [("cuda", "float16")],
            )

    def test_invalid_devices_and_modes_do_not_write(self):
        for key, value in (
            ("asr.device", "auto"),
            ("embedding.text_device", "mps"),
            ("embedding.image_device", "gpu"),
            ("ui.report_mode", "hidden"),
        ):
            with self.assertRaises(SettingsError):
                validate_settings({key: value})
        with self.assertRaises(ConfigValueError):
            self.config(**{"embedding.text_device": "bad"})
        with self.assertRaises(ConfigValueError):
            self.config(**{"ui.report_mode": "bad"})

    def test_active_devices_are_not_saved_preferences(self):
        orch = SimpleNamespace(
            _asr=SimpleNamespace(active_device=("cpu", "int8")),
            _embedder=SimpleNamespace(
                _text=SimpleNamespace(device="cuda"),
                _image=SimpleNamespace(device="cpu"),
            ),
        )
        self.assertEqual(
            active_devices(orch),
            {"faster-whisper-small": "cpu", "bge-m3": "cuda", "siglip2-base": "cpu"},
        )

    def test_gpu_slot_does_not_block_cpu(self):
        gpu = threading.Event()
        cpu = threading.Event()

        def run(device, event):
            with gpu_slot(device):
                event.set()

        with ThreadPoolExecutor(max_workers=2) as pool:
            with gpu_slot("cuda"):
                g = pool.submit(run, "cuda", gpu)
                c = pool.submit(run, "cpu", cpu)
                self.assertTrue(cpu.wait(1))
                self.assertFalse(gpu.is_set())
            g.result(timeout=1)
            c.result(timeout=1)
            self.assertTrue(gpu.is_set())

    def test_active_torch_devices_read_weights_not_target_config(self):
        orch = SimpleNamespace(_embedder=SimpleNamespace(
            _text=SimpleNamespace(device="cuda", _model=SimpleNamespace(device="cpu")),
            _image=SimpleNamespace(device="cpu", _model=SimpleNamespace(device="cuda:0")),
        ))
        self.assertEqual(active_devices(orch)["bge-m3"], "cpu")
        self.assertEqual(active_devices(orch)["siglip2-base"], "cuda")
        orch._embedder._text._model = None
        self.assertEqual(active_devices(orch)["bge-m3"], "未加载")

    def test_settings_endpoint_live_report_mode_and_pending_devices(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.web.ui_page import create_ui_router

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = load_app_config(root=root)
            app = FastAPI()
            app.state.config = cfg
            app.state.orchestrator = SimpleNamespace(
                _asr=SimpleNamespace(active_device=("cpu", "int8")), _embedder=None
            )
            app.include_router(create_ui_router())
            caps = {
                "torch_cuda": True,
                "asr_cuda": True,
                "gpu_name": "fixture",
                "warnings": [],
            }
            with (
                patch("backend.web.ui_page.device_capabilities", return_value=caps),
                patch("backend.web.ui_page.apply_local_devices", side_effect=lambda orch,cfg,persist: (persist(),[])[1]),
                TestClient(app) as client,
            ):
                response = client.post(
                    "/internal/settings",
                    json={
                        "asr.device": "cuda",
                        "asr.compute_type": "float16",
                        "embedding.text_device": "cpu",
                        "embedding.image_device": "cuda",
                        "ui.report_mode": "detailed",
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()
                self.assertEqual(data["effective"]["ui.report_mode"], "detailed")
                self.assertEqual(data["effective"]["embedding.text_device"], "cpu")
                self.assertEqual(data["effective"]["embedding.image_device"], "cuda")
                self.assertEqual(data["active_devices"]["faster-whisper-small"], "cpu")
                self.assertTrue(data["hot_switch_supported"])
                self.assertFalse(data["restart_required"])
                self.assertEqual(
                    client.get("/internal/settings").json()["effective"][
                        "ui.report_mode"
                    ],
                    "detailed",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
