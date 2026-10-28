"""网页服务菜单仅转发白名单操作，使用当前监听端口的本机控制台。"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.config import load_app_config
from backend.web.routes.system import create_router
from backend.runtime import runtime_path


class ServiceMenuTests(unittest.TestCase):
    def test_actions_are_forwarded_to_current_console_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=load_app_config(root=Path(tmp))
            app=FastAPI();app.state.config=cfg
            app.include_router(create_router(),prefix="/internal")
            with TestClient(app,base_url="http://127.0.0.1:8123") as client,patch("backend.web.routes.system.send_console_command",return_value={"accepted":True}) as command:
                for action in ("restart","stop"):
                    response=client.post("/internal/service/control",json={"action":action})
                    self.assertEqual(response.status_code,202,response.text)
                    self.assertEqual(response.json()["pid"],os.getpid())
                    command.assert_called_with(action,state_path=runtime_path("state/console-8123.json",cfg.root_dir))
                command.reset_mock()
                self.assertEqual(client.post("/internal/service/control",json={"action":"shell"}).status_code,422)
                self.assertEqual(client.post("/internal/service/control",json={"action":"restart","path":"C:/Windows"}).status_code,422)
                command.assert_not_called()

    def test_unavailable_console_does_not_report_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            app=FastAPI();app.state.config=load_app_config(root=Path(tmp))
            app.include_router(create_router(),prefix="/internal")
            with TestClient(app) as client,patch("backend.web.routes.system.send_console_command",side_effect=FileNotFoundError):
                response=client.post("/internal/service/control",json={"action":"restart"})
                self.assertEqual(response.status_code,503)
                self.assertIn("控制台",response.json()["detail"])


if __name__=="__main__":unittest.main(verbosity=2)
