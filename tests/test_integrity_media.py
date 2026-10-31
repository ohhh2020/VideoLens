"""真实FFmpeg的时间/分辨率取证与可控OOM，不调用云端。"""
import io
import base64
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import load_app_config
from backend.analysis.runs import RunLedger
from backend.analysis.sampling import extract, metrics
from backend.media.ffmpeg_utils import resolve_binary, probe_media
from backend.models.vl import QwenVideoRunner, _image_to_data_uri, VlError
from backend.models.batching import AdaptiveBatch


class IntegrityMediaTests(unittest.TestCase):
    def test_json_generation_abort_splits_regions_and_motion_keeps_tail(self):
        from backend.analysis.vision import describe as recognize
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = load_app_config(root=root, overrides={'integrity.disk_reserve_bytes': 0})
            ledger = RunLedger(root/'run', 'run')
            ledger.put('motion_windows', [[0, 16000]])
            source = root/'frame.png'; Image.new('RGB', (200,200), 'white').save(source)
            frame = SimpleNamespace(path=source, timestamp_ms=1000, keyframe_id='k', scene_index=0, clip_id='c')
            attempts, windows = [], []
            def describe(paths, prompt, **kwargs):
                attempts.append((len(paths), kwargs.get('native')))
                if kwargs.get('native') and len([a for a in attempts if a[1]]) == 1:
                    exc = VlError('Model output became abnormal while generating a JSON response');exc.code=400
                    raise exc
                text = '{"images":[{"index":0,"text":"数值123"}]}' if kwargs.get('native') else '静态画面'
                return SimpleNamespace(text=text,finish_reason='stop')
            def segment(path, lo, hi, prompt):
                windows.append((lo,hi))
                return SimpleNamespace(text='连续动作', finish_reason='stop',actual_start_ms=lo,actual_end_ms=hi)
            runner=SimpleNamespace(endpoint='test',describe_multi=describe)
            vl=SimpleNamespace(_model_name='test',_runner=runner,supports_video_segments=True,
                               _segment_runner=SimpleNamespace(describe_segment=segment))
            orch=SimpleNamespace(_cfg=cfg,_vl=vl,_analysis_run=ledger)
            with patch('backend.analysis.vision.probe_media',return_value=SimpleNamespace(width=200,height=200)), \
                 patch('backend.analysis.vision.regions',return_value=([[0,0,200,200]],(200,200))), \
                 patch('backend.analysis.vision.crop_native',return_value=(source,[0,0,200,200])):
                result=recognize(orch,source,'v',[frame],[],SimpleNamespace(profile={}))
            self.assertEqual(len(result),1)
            self.assertEqual(windows,[(0,15000),(15000,16000)])
            records=ledger.get('visual_results')
            self.assertEqual(len([r for r in records if r['kind']=='ocr']),2)
            self.assertEqual([r['actual_end_ms'] for r in records if r['kind']=='motion'],[15000,16000])

    def test_actual_frames_native_resolution_and_precise_segments(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source.mp4'
            subprocess.run([resolve_binary('ffmpeg'),'-y','-nostdin','-loglevel','error','-f','lavfi','-i',
                'color=c=white:s=1920x1080:r=24:d=16.15','-c:v','libx264','-preset','ultrafast','-threads','1',str(source)],check=True)
            cfg = load_app_config(root=root, overrides={'integrity.disk_reserve_bytes': 0})
            orch = SimpleNamespace(_cfg=cfg, _analysis_run=RunLedger(root/'run', 'run'))
            clip = SimpleNamespace(clip_id='c', core_start_ms=0, core_end_ms=16200)
            report = SimpleNamespace(profile={})
            frames = extract(orch, source, 'v', [clip], report, SimpleNamespace(max_keyframes=None))
            self.assertTrue(all(frame.width == 1920 and frame.height == 1080 for frame in frames))
            self.assertLessEqual(frames[0].timestamp_ms, 50)
            self.assertGreater(frames[-1].timestamp_ms, 15000)
            self.assertTrue(all(0 <= frame.timestamp_ms < 16200 for frame in frames))
            self.assertTrue(all(abs(a.timestamp_ms-b.timestamp_ms) <= 10050 for a,b in zip(frames,frames[1:])))
            frame = root/'native.png';Image.new('RGB',(1920,1080),'white').save(frame)
            native = _image_to_data_uri(frame,max_side=0)
            self.assertTrue(native.startswith('data:image/png;'))
            with Image.open(io.BytesIO(base64.b64decode(native.split(',')[1]))) as image:
                self.assertEqual(image.size,(1920,1080))
            runner = QwenVideoRunner('https://invalid','model','key')
            runner.encode_segment(source, 0, 15000, root/'first.mp4')
            runner.encode_segment(source, 15000, 16150, root/'tail.mp4')
            self.assertLess(abs(probe_media(root/'tail.mp4').duration_ms-1150),120)
            with self.assertRaises(VlError):
                runner.encode_segment(source,0,16000,root/'too-long.mp4')

    def test_small_local_text_change_is_visible_to_scanner(self):
        import numpy as np
        before = np.zeros((360,640),dtype='uint8')
        after = before.copy();after[120:135,200:215] = 255
        delta, edge, motion = metrics(before,after)
        self.assertTrue(delta >= .04 or edge >= .08)
        self.assertLess(motion,.25)

    def test_oom_halves_batch_and_preserves_sorted_result_order(self):
        cuda = SimpleNamespace(mem_get_info=lambda *a: (4*1024**3,8*1024**3), memory_allocated=lambda *a: 100,
            reset_peak_memory_stats=lambda *a: None, max_memory_allocated=lambda *a: 1000, empty_cache=lambda: None)
        fake_torch = SimpleNamespace(cuda=cuda)
        calls = []
        def encode(items):
            calls.append(len(items))
            if len(items)>2:
                raise RuntimeError('CUDA out of memory')
            return [[item] for item in items]
        with patch.dict(sys.modules, {'torch': fake_torch}):
            result = AdaptiveBatch().run(list(range(10)),encode,'cuda',lengths=list(reversed(range(10))))
            self.assertEqual(result,[[item] for item in range(10)])
            self.assertEqual(calls[0],4)
            self.assertTrue(all(size<=2 for size in calls[1:]))
            with self.assertRaises(RuntimeError):
                AdaptiveBatch(1).run([1],lambda items: (_ for _ in ()).throw(RuntimeError('CUDA out of memory')),'cuda')


if __name__ == '__main__':
    unittest.main(verbosity=2)
