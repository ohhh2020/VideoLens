"""真实FFmpeg、文件软链接、原文件保护和视频库会话标题回归。"""
from pathlib import Path
from types import SimpleNamespace
import ctypes
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI,HTTPException
from fastapi.testclient import TestClient
from backend.config import load_app_config
from backend.media.ffmpeg_utils import resolve_binary
from backend.media.local_video import attach_local_video,describe_local_file,select_native_video
from backend.media.parser import parse_local
from backend.storage.repository import SqliteRepository
from backend.storage.conversations import ConversationStore
from backend.pipeline.orchestrator import Orchestrator
from backend.assistant.uploads import cleanup_uploads
from backend.analysis.resources import inventory
from backend.web.routes.videos import create_router
from backend.web.routes.conversations import create_router as create_conversation_router


class LocalLinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples=tempfile.TemporaryDirectory()
        cls.sample=Path(cls.samples.name)/'sample.mp4'
        subprocess.run([resolve_binary('ffmpeg'),'-y','-loglevel','error','-f','lavfi','-i','color=c=navy:s=160x90:r=4',
            '-t','1.1','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(cls.sample)],check=True,capture_output=True)

    @classmethod
    def tearDownClass(cls):cls.samples.cleanup()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.cfg=load_app_config(root=self.root)
        self.repo=SqliteRepository(self.cfg.path('sqlite'));self.store=ConversationStore(self.repo)
        self.cid=self.store.create()['id'];self.service=SimpleNamespace(config=self.cfg,store=self.store)
        self.original=self.root/'原始 视频.mp4';self.original.write_bytes(self.sample.read_bytes())

    def tearDown(self):self.repo.close();self.temp.cleanup()

    def attach(self):return attach_local_video(self.service,self.cid,str(self.original))

    def test_real_link_analysis_path_and_delete_preserve_original(self):
        original=self.original.read_bytes()
        with patch('shutil.copy2',side_effect=AssertionError('禁止复制视频')):
            result=self.attach()
        link=Path(result['link_path'])
        self.assertTrue(link.is_symlink());self.assertEqual(link.resolve(),self.original)
        self.assertEqual(link.read_bytes(),original)
        media=parse_local(link);direct=parse_local(self.original)
        self.assertEqual(media.path,link)
        self.assertEqual(media.video_id,direct.video_id)
        self.repo.add_video(video_id=media.video_id,source_kind='local',local_path=str(link))
        self.store.update(self.cid,video_id=media.video_id)
        worker=Orchestrator(repository=self.repo,config=self.cfg);worker._use_integrity=True
        removed=worker.delete(media.video_id)
        self.assertNotIn('db_error',removed)
        self.assertFalse(link.is_symlink())
        self.assertEqual(self.original.read_bytes(),original)

    def test_failed_symlink_never_falls_back_to_copy(self):
        with patch('pathlib.Path.symlink_to',side_effect=OSError('permission denied')):
            with self.assertRaises(HTTPException) as context:self.attach()
        self.assertEqual(context.exception.status_code,507)
        self.assertTrue(self.original.exists())
        self.assertEqual(list((self.cfg.path('data')/'uploads').iterdir()),[])

    def test_stale_selection_invalid_path_and_media_limits(self):
        descriptor=describe_local_file(str(self.original),self.cfg)
        with self.assertRaises(HTTPException) as context:
            attach_local_video(self.service,self.cid,str(self.original),expected_size=descriptor['size_bytes']+1)
        self.assertEqual(context.exception.status_code,409)
        with self.assertRaises(HTTPException):describe_local_file('relative.mp4',self.cfg)
        with self.assertRaises(HTTPException):describe_local_file('https://example.org/video.mp4',self.cfg)
        broken=self.root/'broken.mp4';broken.write_bytes(b'broken')
        with self.assertRaises(Exception):attach_local_video(self.service,self.cid,str(broken))
        self.assertEqual(self.repo._query('SELECT * FROM uploaded_videos'),[])

    def test_inventory_counts_link_itself_not_full_video(self):
        self.attach()
        stats=inventory(self.cfg,refresh=True)
        self.assertLess(stats['source_bytes'],self.original.stat().st_size)

    def test_link_reference_protects_original_even_inside_old_upload_cache(self):
        directory=self.cfg.path('data')/'uploads';directory.mkdir(parents=True,exist_ok=True)
        target=directory/'legacy-cache.mp4';self.original.replace(target);self.original=target
        old=time.time()-48*3600;os.utime(target,(old,old))
        result=self.attach();cleanup_uploads(self.service)
        self.assertTrue(target.exists())
        self.assertTrue(Path(result['link_path']).is_file())

    def test_local_endpoint_cancel_preview_then_explicit_attach(self):
        app=FastAPI();app.state.config=self.cfg
        self.service.orch=SimpleNamespace(_cfg=self.cfg)
        app.state.assistant=self.service
        app.include_router(create_conversation_router(),prefix='/internal')
        with TestClient(app) as client,patch('backend.media.local_video.select_native_video',return_value=None):
            self.assertTrue(client.post('/internal/local-videos/select').json()['cancelled'])
            descriptor=client.post('/internal/local-videos/inspect',json={'path':str(self.original)}).json()
            self.assertEqual(self.repo._query('SELECT * FROM uploaded_videos'),[])
            response=client.post('/internal/conversations/'+self.cid+'/local-video',json={
                'path':descriptor['path'],'size_bytes':descriptor['size_bytes'],'last_modified':descriptor['last_modified']})
            self.assertEqual(response.status_code,201,response.text)
            self.assertTrue(Path(response.json()['link_path']).is_symlink())

    def test_expired_unclaimed_broken_link_removed_without_touching_source(self):
        result=self.attach();link=Path(result['link_path'])
        moved=self.original.with_name('已移动.mp4');self.original.rename(moved)
        future=time.time()+48*3600
        with patch('backend.assistant.uploads.time.time',return_value=future):
            cleanup_uploads(self.service)
        self.assertFalse(link.is_symlink());self.assertTrue(moved.exists())

    def test_library_title_follows_named_conversation_and_shows_real_local_path(self):
        result=self.attach();link=Path(result['link_path'])
        self.repo.add_video(video_id='vlocal',title='原文件名',source_kind='local',local_path=str(link))
        self.store.update(self.cid,video_id='vlocal',title='本地对话标题')
        self.store.create(video_id='vlocal')
        self.repo.add_video(video_id='vcloud',title='原视频标题',source_url='https://example.org/watch?x=1',source_kind='url')
        cloud=self.store.create(video_id='vcloud',title='云端对话标题')
        app=FastAPI();app.state.config=self.cfg;app.state.orchestrator=SimpleNamespace(repository=self.repo)
        app.include_router(create_router(),prefix='/internal')
        with TestClient(app) as client:
            rows={row['video_id']:row for row in client.get('/internal/videos').json()}
            self.assertEqual(rows['vlocal']['title'],'本地对话标题')
            self.assertEqual(rows['vlocal']['conversation_id'],self.cid)
            self.assertEqual(rows['vlocal']['local_path'],str(self.original))
            self.assertEqual(rows['vcloud']['source_url'],'https://example.org/watch?x=1')
            self.assertEqual(rows['vcloud']['conversation_id'],cloud['id'])
            self.store.update(self.cid,title='已改名')
            self.assertEqual(client.get('/internal/videos/vlocal').json()['video']['title'],'已改名')
        self.assertEqual(self.repo.find_video('vlocal').title,'原文件名')

    @unittest.skipUnless(os.name=='nt','Windows本机文件选择器')
    def test_native_dialog_protocol_and_cancel_without_opening_desktop(self):
        dialog=SimpleNamespace(GetOpenFileNameW=Mock(return_value=0),CommDlgExtendedError=Mock(return_value=0))
        user=SimpleNamespace(GetForegroundWindow=Mock(return_value=None))
        with patch('ctypes.WinDLL',side_effect=lambda name,**kw:dialog if name=='comdlg32' else user):
            self.assertIsNone(select_native_video())
            def choose(pointer):
                options=pointer._obj
                self.assertTrue(options.Flags&0x8)
                self.assertTrue(options.Flags&0x1000)
                address=ctypes.c_void_p.from_address(ctypes.addressof(options)+type(options).lpstrFile.offset).value
                value=ctypes.create_unicode_buffer(str(self.original))
                ctypes.memmove(address,value,ctypes.sizeof(value))
                return 1
            dialog.GetOpenFileNameW.side_effect=choose
            self.assertEqual(select_native_video(),str(self.original))


if __name__=='__main__':unittest.main(verbosity=2)
