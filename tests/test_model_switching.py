"""设备热切换的事务、并发等待与旧模型引用回归，不加载真实权重。"""

import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.config import load_app_config
from backend.models.asr import AsrConfig, AsrEngine
from backend.models.embed import EmbeddingService
from backend.models.devices import active_devices
from backend.models.switching import apply_local_devices, DeviceSwitchError
from backend.user_settings import preview_settings, save_settings, settings_file


class Tower:
    dim = 2

    def __init__(self, device, value=1):
        self.device = device
        self._model = SimpleNamespace(device=device)
        self.value = value

    def embed_texts(self, texts):
        assert self._model is not None, "旧模型已卸载，不能继续推理"
        return [[self.value, 0] for _ in texts]

    embed_images = embed_texts

    def unload(self):
        self._model = None


class SwitchingTests(unittest.TestCase):
    def config(self, **values):
        return load_app_config(root=ROOT, ignore_user_settings=True, overrides=values)

    def make_asr(self, config, model_manager=None):
        obj = AsrEngine(config, model_manager=model_manager)
        obj._model = SimpleNamespace()
        obj._loaded_key = ("fixture", config.device, config.compute_type)
        obj.load = Mock(return_value=obj._model)
        return obj

    def make_embed(self, config, manager=None):
        return EmbeddingService(
            Tower(config.embedding.text_device or "cpu"),
            Tower(config.embedding.image_device or "cpu"),
        )

    def make_orch(self):
        cfg = self.config()
        asr = self.make_asr(AsrConfig.from_config(cfg), object())
        return SimpleNamespace(_cfg=cfg, _asr=asr, _embedder=self.make_embed(cfg))

    def mocks(self):
        return (
            patch("backend.models.switching.EmbeddingService.from_config", side_effect=self.make_embed),
            patch("backend.models.switching.AsrEngine", side_effect=self.make_asr),
        )

    def test_independent_switch_keeps_service_identity_and_noop_does_not_reload(self):
        orch = self.make_orch()
        services = (orch._asr, orch._embedder)
        old_image = orch._embedder._image
        cfg = self.config(**{"embedding.text_device":"cuda"})
        a,b = self.mocks()
        with a as build, b, patch("backend.models.switching.detect_device", side_effect=lambda x:x):
            persist = Mock()
            self.assertEqual(apply_local_devices(orch,cfg,persist), ["bge-m3"])
            self.assertEqual((orch._asr,orch._embedder),services)
            self.assertIs(orch._embedder._image,old_image)
            self.assertEqual(active_devices(orch)["bge-m3"],"cuda")
            self.assertEqual(apply_local_devices(orch,cfg,persist), [])
            build.assert_called_once()

    def test_model_or_disk_failure_keeps_old_models_and_config(self):
        for failure in ("model", "disk"):
            orch = self.make_orch()
            old = (orch._cfg,orch._asr._model,orch._embedder._text)
            cfg = self.config(**{"embedding.text_device":"cuda"})
            persist = Mock(side_effect=OSError("fixture") if failure=="disk" else None)
            a,b = self.mocks()
            with a as build,b,patch("backend.models.switching.detect_device",side_effect=lambda x:x):
                if failure=="model":
                    build.side_effect=RuntimeError("设备不可用")
                with self.assertRaises(DeviceSwitchError):
                    apply_local_devices(orch,cfg,persist)
            self.assertEqual((orch._cfg,orch._asr._model,orch._embedder._text),old)
            self.assertIsNotNone(old[2]._model)
            if failure=="model":persist.assert_not_called()

    def test_switch_waits_for_running_model(self):
        orch = self.make_orch()
        entered = threading.Event()
        persisted = threading.Event()
        cfg = self.config(**{"embedding.text_device":"cuda"})
        a,b = self.mocks()
        with a,b,patch("backend.models.switching.detect_device",side_effect=lambda x:x):
            with ThreadPoolExecutor(max_workers=1) as pool:
                with orch._embedder._text_lock:
                    def switch():
                        entered.set()
                        apply_local_devices(orch,cfg,persisted.set)
                    future=pool.submit(switch)
                    self.assertTrue(entered.wait(1))
                    self.assertFalse(persisted.wait(0.05))
                    self.assertEqual(active_devices(orch)["bge-m3"],"cpu")
                future.result(timeout=2)
        self.assertTrue(persisted.is_set())

    def test_cross_modal_waiter_reads_new_model_after_switch(self):
        embed=EmbeddingService(Tower("cpu"),Tower("cpu"))
        with ThreadPoolExecutor(max_workers=1) as pool:
            with embed._image_lock:
                future=pool.submit(embed.embed_cross_modal,"问题")
                old=embed._image
                embed._image=Tower("cpu",2)
                old.unload()
            self.assertEqual(future.result(timeout=2),[2,0])

    def test_preview_is_read_only_preserves_keys_and_env_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            save_settings({"llm.cloud.api_key":"sk-complete-fixture-key"},root=root)
            before=settings_file(root).read_bytes()
            with patch.dict("os.environ",{"ZH_YING__EMBEDDING__TEXT_DEVICE":"cpu"}):
                clean,cfg=preview_settings({"embedding.text_device":"cuda"},root=root)
            self.assertEqual(cfg.embedding.text_device,"cpu")
            self.assertEqual(clean["embedding.text_device"],"cuda")
            self.assertEqual(cfg.llm.cloud.api_key,"sk-complete-fixture-key")
            self.assertEqual(settings_file(root).read_bytes(),before)

    def test_web_endpoint_applies_devices_and_failure_keeps_saved_settings(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.web.ui_page import create_ui_router

        with tempfile.TemporaryDirectory() as tmp:
            orch=self.make_orch()
            cfg=load_app_config(root=Path(tmp))
            orch._cfg=cfg
            app=FastAPI()
            app.state.config=cfg
            app.state.orchestrator=orch
            app.include_router(create_ui_router())
            a,b=self.mocks()
            caps={"torch_cuda":True,"asr_cuda":True,"warnings":[],"gpu_name":"fixture"}
            with a as build,b,patch("backend.web.ui_page.device_capabilities",return_value=caps),TestClient(app) as client:
                response=client.post("/internal/settings",json={
                    "asr.device":"cuda","asr.compute_type":"float16",
                    "embedding.text_device":"cuda","embedding.image_device":"cuda",
                    "ui.report_mode":"detailed",
                })
                self.assertEqual(response.status_code,200,response.text)
                result=response.json()
                self.assertTrue(result["hot_switch_supported"])
                self.assertFalse(result["restart_required"])
                self.assertTrue(all(d=="cuda" for d in result["active_devices"].values()))
                self.assertEqual(len(result["switched_models"]),3)
                before=settings_file(cfg.root_dir).read_bytes()
                build.side_effect=RuntimeError("fixture load failure")
                response=client.post("/internal/settings",json={"embedding.image_device":"cpu","ui.report_mode":"user"})
                self.assertEqual(response.status_code,409,response.text)
                self.assertEqual(settings_file(cfg.root_dir).read_bytes(),before)
                self.assertEqual(orch._cfg.ui.report_mode,"detailed")


if __name__=="__main__":
    unittest.main(verbosity=2)
