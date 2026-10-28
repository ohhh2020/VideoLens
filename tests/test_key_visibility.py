"""密钥仅显式请求时读取；普通响应不泄露内容或长度。测试使用虚构密钥。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from backend.config import load_app_config
from backend.user_settings import save_settings, settings_file
from backend.web.main import create_app


class KeyVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.keys = {"llm.cloud.api_key": "sk-example-short", "vl.cloud_api_key": "sk-example-much-longer-credential-123456"}
        save_settings(self.keys, root=self.root)
        self.cfg = load_app_config(root=self.root)
        self.orch = SimpleNamespace(_cfg=self.cfg)
        self.client = TestClient(create_app(self.cfg, orchestrator=self.orch))
        self.addCleanup(self.client.close)
        for name, value in [("device_capabilities", {"torch_cuda": False}), ("describe_backends", [])]:
            mock = patch(f"backend.web.ui_page.{name}", return_value=value)
            mock.start()
            self.addCleanup(mock.stop)

    def test_default_response_masks_both_keys_without_length(self):
        response = self.client.get("/internal/settings")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        for key, value in self.keys.items():
            self.assertEqual(response.json()["values"][key], "●" * 8)
            self.assertTrue(response.json()["secret_present"][key])
            self.assertNotIn(value, response.text)

    def test_reveal_is_explicit_whitelisted_and_not_cached(self):
        before = settings_file(self.root).read_bytes()
        for key, value in self.keys.items():
            response = self.client.post("/internal/settings/reveal-key", json={"key": key})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"value": value})
            self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(settings_file(self.root).read_bytes(), before)
        self.assertEqual(self.client.get("/internal/settings/reveal-key").status_code, 405)
        for payload in [{"key": "other.secret"}, {"key": "llm.cloud.api_key", "path": "other"}]:
            self.assertEqual(self.client.post("/internal/settings/reveal-key", json=payload).status_code, 422)

    def test_cross_origin_and_remote_access_are_rejected(self):
        for headers in [{"Origin": "https://example.com"}, {"Sec-Fetch-Site": "cross-site"}, {"Host": "remote.example"}]:
            response = self.client.post("/internal/settings/reveal-key", json={"key": "llm.cloud.api_key"}, headers=headers)
            self.assertEqual(response.status_code, 403)
            self.assertNotIn(self.keys["llm.cloud.api_key"], response.text)

    def test_environment_fallback_and_empty_key(self):
        save_settings({key: "" for key in self.keys}, root=self.root)
        fresh = load_app_config(root=self.root)
        with patch.dict("os.environ", {fresh.llm.cloud.api_key_env: "sk-environment-example", fresh.vl.cloud_api_key_env: ""}):
            settings = self.client.get("/internal/settings").json()
            self.assertTrue(settings["secret_present"]["llm.cloud.api_key"])
            self.assertFalse(settings["secret_present"]["vl.cloud_api_key"])
            for key, value in [("llm.cloud.api_key", "sk-environment-example"), ("vl.cloud_api_key", "")]:
                response = self.client.post("/internal/settings/reveal-key", json={"key": key})
                self.assertEqual(response.json()["value"], value)

    def test_unrelated_save_preserves_keys_and_returns_only_masks(self):
        def apply(orch, fresh, commit):
            commit()
            orch._cfg = fresh
            return []
        with patch("backend.web.ui_page.apply_local_devices", side_effect=apply):
            response = self.client.post("/internal/settings", json={"ui.report_mode": "detailed"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")
        raw = json.loads(settings_file(self.root).read_text(encoding="utf-8"))
        for key, value in self.keys.items():
            self.assertEqual(raw[key], value)
            self.assertNotIn(value, response.text)
            self.assertEqual(response.json()["values"][key], "●" * 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
