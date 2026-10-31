"""回退策略的来源、预算、调用次数与旧版本兼容回归。"""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.analysis import fast
from backend.analysis.policy import IntegritySection
from backend.analysis.runs import RunLedger, snapshot, restore
from backend.config import load_app_config


class FastTests(unittest.TestCase):
    def test_fixed_sampling_has_no_change_refinement_and_keeps_tail(self):
        points = fast.sample_points(686544, 10)
        self.assertEqual(len(points), 70)
        self.assertEqual(points[:3], [0,10000,20000])
        self.assertEqual(points[-1], 686444)
        self.assertEqual(fast.sample_points(40,10), [0])

    def test_snapshot_uses_fast_but_old_snapshot_keeps_old_execution(self):
        config = load_app_config()
        values = snapshot(config)
        self.assertEqual(values['_analysis_strategy'], 'fast-v3')
        restore(config, values)
        with tempfile.TemporaryDirectory() as directory:
            ledger = RunLedger(directory, 'strategy')
            orch = SimpleNamespace(_analysis_run=ledger)
            ledger.put('policy', values)
            self.assertTrue(fast.enabled(orch))
            values.pop('_analysis_strategy')
            ledger.put('policy', values)
            self.assertFalse(fast.enabled(orch))

    def test_one_report_call_spreads_over_whole_timeline_and_keeps_modalities(self):
        clips = [{'clip_id':str(i),'start_ms':i*20000,'end_ms':(i+1)*20000,'text':'语音数字123。'*30} for i in range(108)]
        repo = SimpleNamespace(iter_clip_payloads=lambda v: iter(clips), get_segments=lambda v: [], get_keyframes=lambda v: [])
        calls = []
        class Model:
            def complete(self, messages, **kwargs):
                calls.append(messages)
                return SimpleNamespace(text='## 内容概览\n报告\n## 关键结论\n数字123',finish_reason='stop')
        with tempfile.TemporaryDirectory() as directory:
            ledger = RunLedger(directory, 'single')
            ledger.put('visual_results', [{'kind':kind,'start_ms':i*20000,'end_ms':i*20000,
                'text':('画面内容ABC' if kind=='visual' else 'OCR数字456')*10} for i in range(108) for kind in ('visual','ocr')])
            fast.summarize(repo,'v','视频',Model(),IntegritySection(),ledger)
            self.assertEqual(len(calls),1)
            content=json.loads(calls[0][-1]['content'])['representative_evidence']
            self.assertEqual(len(content),32)
            self.assertEqual(content[0]['clip_id'],'0')
            self.assertEqual(content[-1]['clip_id'],'107')
            self.assertTrue(all(row.get('speech') and row.get('visual') and row.get('ocr') for row in content))
            self.assertLess(ledger.get('coverage')['ratio'],1)
            self.assertTrue(ledger.get('coverage')['representative'])
            self.assertEqual(ledger.get('summary_tree'),[])

    def test_frame_batch_never_uses_native_crops(self):
        calls=[]
        class Runner:
            endpoint='https://test.invalid'; max_image_side=1280
            def describe_multi(self,paths,prompt,**kwargs):
                calls.append((paths,kwargs))
                # 真实兼容端点会返回顶层数组，不能因此拆批重试。
                return SimpleNamespace(text=json.dumps([{'index':i,'description':'示意图','ocr':'123'} for i in range(len(paths))]),finish_reason='stop')
        with tempfile.TemporaryDirectory() as directory:
            ledger=RunLedger(directory,'frames'); frames=[]
            for i in range(7):
                path=Path(directory)/f'{i}.jpg'; path.write_bytes(bytes([i]))
                frames.append(SimpleNamespace(path=path,keyframe_id=str(i),video_id='v',timestamp_ms=i*1000,
                    scene_index=i,clip_id='c',width=1920,height=1080,bytes=1,notes=()))
            orch=SimpleNamespace(_analysis_run=ledger,_cfg=SimpleNamespace(integrity=IntegritySection(),vl=SimpleNamespace(cloud_sequence_frames=8)),
                _vl=SimpleNamespace(_runner=Runner(),_model_name='test'))
            with patch('backend.analysis.vision.crop_native',side_effect=AssertionError('禁止裁图')):
                payloads=fast.describe(orch,None,'v',frames,[],SimpleNamespace(profile={}))
            self.assertEqual(len(calls),2)
            self.assertEqual(len(payloads),7)
            self.assertTrue(all('native' not in kwargs for _,kwargs in calls))
            self.assertEqual(len(ledger.get('visual_results')),14)

    def test_numeric_comparison_remains_verbatim_when_many_frames_share_one_clip(self):
        clips=[{'clip_id':'c','start_ms':0,'end_ms':20000,'text':'性能对比'}]
        repo=SimpleNamespace(iter_clip_payloads=lambda v: iter(clips),get_segments=lambda v: [],get_keyframes=lambda v: [])
        class Model:
            def complete(self,messages,**options):
                data=json.loads(messages[-1]['content'])
                assert any(row['text']=='生产力方面，56XT落后12%（输多赢少）' for row in data['numeric_quotes'])
                return SimpleNamespace(text='## 关键结论\n按原文保留比较方向',finish_reason='stop')
        with tempfile.TemporaryDirectory() as directory:
            ledger=RunLedger(directory,'quotes')
            ledger.put('visual_results',[{'kind':'ocr','start_ms':i*1000,'end_ms':i*1000,
                'text':'测试总结\n生产力方面，56XT落后12%（输多赢少）\n字幕变化'+str(i)} for i in range(20)])
            fast.summarize(repo,'v','对比',Model(),IntegritySection(),ledger)
            self.assertEqual(len(ledger.get('report_selection')['numeric_quotes']),1)


if __name__=='__main__':unittest.main(verbosity=2)
