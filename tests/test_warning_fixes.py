"""上游默认配置假告警和 yt-dlp 路径/匿名字幕回归，不连接外部站点。"""
import json
import logging
import sys
import tempfile
import threading
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.models.siglip_compat import load_siglip_config
from backend.media.downloader import DownloadConfig, MediaDownloader
from backend.media.parser import parse_source


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages=[]
    def emit(self,record):self.messages.append(record.getMessage())


class WarningFixTests(unittest.TestCase):
    def test_siglip_serialization_does_not_instantiate_invalid_defaults(self):
        log=logging.getLogger("transformers.configuration_utils")
        capture=Capture();log.addHandler(capture)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)
                for model_type in ("siglip","siglip2"):
                    raw={"model_type":model_type,"text_config":{"vocab_size":256000}}
                    (path/"config.json").write_text(json.dumps(raw),encoding="utf-8")
                    config,_,_=load_siglip_config(path)
                    repr(config);config.to_json_string();config.to_diff_dict()
                    self.assertEqual(config.text_config.vocab_size,256000)
                    self.assertEqual(config.text_config.bos_token_id,49406)
                    self.assertEqual(config.text_config.eos_token_id,49407)
                self.assertFalse(any("within the vocabulary" in msg for msg in capture.messages))
                # 真正的模型配置错误仍然可见，未屏蔽 Transformers 的日志。
                raw={"model_type":"siglip","text_config":{"vocab_size":10}}
                (path/"config.json").write_text(json.dumps(raw),encoding="utf-8")
                load_siglip_config(path)
                self.assertTrue(any("within the vocabulary" in msg for msg in capture.messages))
        finally:log.removeHandler(capture)

    def test_download_paths_and_anonymous_bilibili_subtitles(self):
        from yt_dlp import YoutubeDL
        from yt_dlp.extractor.bilibili import BiliBiliIE
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dl=MediaDownloader(DownloadConfig(videos_dir=root/"videos",temp_dir=root/"temp"))
            source=parse_source("https://www.bilibili.com/video/BV1bxcfzeEab")
            target=root/"videos"/source.video_id
            target.mkdir(parents=True)
            with patch("backend.media.downloader.logger.warning") as warning:
                options=dl._build_options(source,target,1,None)
                self.assertFalse(options["writesubtitles"])
                self.assertFalse(options["writeautomaticsub"])
                self.assertFalse(options["no_warnings"],"保留其他真正告警")
                self.assertFalse(Path(options["outtmpl"]["default"]).is_absolute())
                with YoutubeDL(options) as ydl:
                    info={"id":"example","title":"example","ext":"mp4"}
                    self.assertEqual(Path(ydl.prepare_filename(info,warn=True)),target.resolve()/"source.mp4")
                    self.assertEqual(Path(ydl.prepare_filename(info,"temp",warn=True)),root.resolve()/"temp"/source.video_id/"source.mp4")
                    extractor=BiliBiliIE(ydl)
                    with patch.object(extractor,"_download_json",side_effect=AssertionError("匿名状态不请求受限字幕")):
                        self.assertEqual(extractor.extract_subtitles("example",123),{})
                warning.assert_not_called()
            second=dl._build_options(source,root/"videos"/"another-video",1,None)
            self.assertNotEqual(options["paths"]["temp"],second["paths"]["temp"])
            (target/"source.mp4.part").write_bytes(b"old partial")
            self.assertEqual(dl._build_options(source,target,1,None)["paths"]["temp"],str(target.resolve()))

    def test_cookie_configuration_and_other_sites_keep_subtitles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cookie=root/"cookies.txt";cookie.write_text("# Netscape HTTP Cookie File\n",encoding="utf-8")
            bili=parse_source("https://www.bilibili.com/video/BV1bxcfzeEab")
            for kwargs in ({"cookie_file":cookie},{"cookies_from_browser":"chrome"}):
                dl=MediaDownloader(DownloadConfig(videos_dir=root/"videos",**kwargs))
                options=dl._build_options(bili,root/"videos"/bili.video_id,1,None)
                self.assertTrue(options["writesubtitles"])
                self.assertTrue(options["writeautomaticsub"])
            dl=MediaDownloader(DownloadConfig(videos_dir=root/"videos"))
            options=dl._build_options(parse_source("https://www.youtube.com/watch?v=abcdefghijk"),root/"videos"/"youtube",1,None)
            self.assertTrue(options["writesubtitles"])

    def test_real_ytdlp_download_reaches_final_directory(self):
        # 只验证下载与文件移动，测试字节不作为有效视频交给 FFmpeg/模型。
        class Handler(SimpleHTTPRequestHandler):
            def log_message(self,*args):pass
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);origin=root/"origin";origin.mkdir()
            content=b"test download bytes"*256
            (origin/"sample.mp4").write_bytes(content)
            server=ThreadingHTTPServer(("127.0.0.1",0),partial(Handler,directory=str(origin)))
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            try:
                source=parse_source(f"http://127.0.0.1:{server.server_port}/sample.mp4")
                dl=MediaDownloader(DownloadConfig(videos_dir=root/"videos",temp_dir=root/"temp",write_subtitles=False))
                target=root/"videos"/source.video_id;target.mkdir(parents=True)
                # 隔离本机代理环境；本用例仅连接临时 localhost 服务。
                original=dl._build_options
                def options(*args):return {**original(*args),"proxy":""}
                with patch.object(dl,"_build_options",side_effect=options),patch("backend.media.downloader.logger.warning") as warning:
                    result=dl._attempt(source,target,1,None)
                    warning.assert_not_called()
                self.assertEqual(result.path,target/"source.mp4")
                self.assertEqual(result.path.read_bytes(),content)
                self.assertTrue((target/"source.info.json").is_file())
            finally:
                server.shutdown();server.server_close();worker.join(timeout=2)


if __name__=="__main__":unittest.main(verbosity=2)
