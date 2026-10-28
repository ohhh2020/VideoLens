"""缓存入口仅打开视频记录对应目录，不允许客户端指定路径。"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.web.routes.videos import create_router


@unittest.skipUnless(os.name=="nt","Windows 缓存文件夹入口")
class VideoCacheTests(unittest.TestCase):
    def test_opens_record_folder_and_ignores_client_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp)/"中文 视频";cache.mkdir()
            video=cache/"source.mp4";video.write_bytes(b"fixture")
            repo=Mock();repo.find_video.return_value=SimpleNamespace(source_kind="url",local_path=str(video))
            app=FastAPI();app.state.orchestrator=SimpleNamespace(repository=repo)
            app.include_router(create_router(),prefix="/internal")
            with TestClient(app) as client,patch("backend.web.routes.videos.os.startfile") as launch:
                response=client.post("/internal/videos/v1/open-cache",json={"path":"C:/Windows"})
                self.assertEqual(response.status_code,200,response.text)
                launch.assert_called_once_with(str(cache.resolve()))
                self.assertEqual(response.json()["directory"],str(cache.resolve()))

    def test_missing_or_local_video_never_opens_folder(self):
        repo=Mock();app=FastAPI();app.state.orchestrator=SimpleNamespace(repository=repo)
        app.include_router(create_router(),prefix="/internal")
        with TestClient(app) as client,patch("backend.web.routes.videos.os.startfile") as launch:
            for record,status in (
                (None,404),
                (SimpleNamespace(source_kind="local",local_path="C:/Windows"),409),
                (SimpleNamespace(source_kind="url",local_path=""),409),
                (SimpleNamespace(source_kind="url",local_path=str(ROOT/"missing-cache.mp4")),404),
            ):
                repo.find_video.return_value=record
                self.assertEqual(client.post("/internal/videos/v1/open-cache").status_code,status)
            launch.assert_not_called()


if __name__=="__main__":unittest.main(verbosity=2)
