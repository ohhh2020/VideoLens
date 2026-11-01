"""fast-v4正确性、完整事实预算、时间、缓存与并发故障边界。"""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.analysis.packing import atomic_units, select
from backend.analysis.decoding import stream_frames
from backend.analysis.runs import RunLedger, snapshot
from backend.analysis.policy import estimate_tokens
from backend.analysis import transcripts
from backend.config import load_app_config
from backend.media.ffmpeg_utils import probe_media, resolve_binary
from backend.models.asr import AsrConfig, Transcript, TranscriptSegment, Word
from backend.pipeline.domain import AnalyzeOptions
from backend.pipeline.orchestrator import Orchestrator
from backend.storage.repository import SqliteRepository


class OptimizedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = load_app_config(root=self.root, overrides={'integrity.disk_reserve_bytes': 0,
            'integrity.max_sample_gap_s': 1, 'media.scene_detect.max_workers': 2})
        self.ledger = RunLedger(self.root / 'run', 'run')
        self.ledger.put('policy', snapshot(self.cfg))

    def tearDown(self):
        self.temp.cleanup()

    def make_video(self, audio=False, extra=()):
        path = self.root / 'video.mp4'
        args = [resolve_binary('ffmpeg'), '-y', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=160x90:r=10']
        if audio:
            args += ['-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=16000']
        args += ['-t', '2.3', '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', *extra, str(path)]
        subprocess.run(args, check=True, capture_output=True)
        return path

    def test_atomic_units_preserve_code_tables_and_negation(self):
        text = '必须先安装驱动。不能使用DDR4。\n\n| 型号 | 价格 |\n|---|---|\n| A1 | 750元 |\n| A2 | 890元 |\n\n```python\nif x:\n    print(1)\n```'
        units = atomic_units(text)
        self.assertIn('不能使用DDR4。', units)
        self.assertTrue(any('| 型号 | 价格 |' in row and '| A2 | 890元 |' in row for row in units))
        self.assertTrue(any('    print(1)' in row and row.endswith('```') for row in units))

    def test_same_budget_whole_units_ocr_dedup_all_occurrences_and_keywords(self):
        evidence = [{'id': f'e{i}', 'clip_id': f'c{i}', 'kind': 'ocr', 'start_ms': i * 1000, 'end_ms': i * 1000,
            'text': 'i5-13600KF售价1116元，仅限盒装。\n不能使用DDR4。'} for i in range(40)]
        evidence += [{'id': 'last', 'clip_id': 'c40', 'kind': 'speech', 'start_ms': 40000, 'end_ms': 41000,
                      'text': '生产力测试中5600XT落后12%，网游领先32%。'}]
        make = lambda rows: [{'role': 'user', 'content': json.dumps(rows, ensure_ascii=False)}]
        rows, meta = select(evidence, make, 3000)
        self.assertLessEqual(estimate_tokens(json.dumps(make(rows), ensure_ascii=False)), 3000)
        self.assertEqual(meta['total_clips'], 41)
        self.assertEqual(len(meta['selected_clips']), 41)
        self.assertTrue(any(row['repeats'] == 40 for row in rows))
        self.assertIn('生产力测试中5600XT落后12%，网游领先32%。', [row['text'] for row in rows])
        self.assertTrue(meta['keywords'])
        for row in rows:
            self.assertTrue(row['text'].endswith('。'))
        self.assertEqual(sum(len(row['occurrences']) for row in meta['unit_provenance']), 81)

    def test_oversized_unit_is_visible_not_prefix_truncated(self):
        body = '特别长的连续原句' * 200 + '末尾型号A5'
        evidence = [{'id':'e1','clip_id':'c1','kind':'speech','start_ms':0,'end_ms':1000,'text':body},
                    {'id':'e2','clip_id':'c2','kind':'ocr','start_ms':2000,'end_ms':2000,'text':'售价750元。'}]
        rows, meta = select(evidence, lambda rows: [{'role':'user','content':json.dumps(rows,ensure_ascii=False)}], 1200)
        self.assertEqual([row['text'] for row in rows], ['售价750元。'])
        self.assertEqual(meta['omitted_count'], 1)
        self.assertEqual(meta['unit_provenance'][0]['text'], body)

    def test_decoder_reuses_context_real_pts_and_does_not_detect_scenes(self):
        path = self.make_video()
        orch = SimpleNamespace(_analysis_run=self.ledger, _cfg=self.cfg)
        report = SimpleNamespace(profile={})
        with patch('backend.segmentation.scenes.SceneDetector.detect_scenes', side_effect=AssertionError('禁止镜头检测')):
            frames = list(stream_frames(orch, path, 'video', probe_media(path), report, AnalyzeOptions()))
        self.assertEqual(len(frames), 4)
        self.assertLessEqual(report.profile['decoder_instances'], 2)
        self.assertTrue(all(frame.requested_ms <= frame.timestamp_ms <= frame.requested_ms + 101 for frame in frames))
        self.assertEqual(len(self.ledger.get('visual_coverage')['sampling_points_ms']), 4)
        self.assertTrue(all(frame.width == 160 and frame.height == 90 for frame in frames))

    def test_transcript_cache_params_audio_and_weights_invalidate(self):
        directory = self.root / 'weights'; directory.mkdir(); (directory / 'model.bin').write_bytes(b'weights')
        audio = self.root / 'audio.wav'; audio.write_bytes(b'audio-content')
        asr = SimpleNamespace(config=AsrConfig(device='cuda'), _model_path=lambda: directory)
        orch = SimpleNamespace(_asr=asr, _cfg=self.cfg)
        key, identity = transcripts.identity(orch, audio)
        value = Transcript(video_id='old', segments=(TranscriptSegment(0,0,1000,'价格750元。',
            words=(Word('750元', 100, 900),)),), device='cuda')
        transcripts.write(orch, key, identity, value)
        restored = transcripts.read(orch, key, 'new')
        self.assertEqual(restored.video_id, 'new'); self.assertEqual(restored.segments[0].words, value.segments[0].words)
        self.assertEqual(transcripts.purge(orch, 'old'), 0)
        self.assertIsNotNone(transcripts.read(orch, key, 'new'))
        self.assertEqual(transcripts.purge(orch, 'new'), 1)
        self.assertIsNone(transcripts.read(orch, key, 'new'))
        asr.config = replace(asr.config, beam_size=1)
        self.assertNotEqual(key, transcripts.identity(orch, audio)[0])
        asr.config = replace(asr.config, beam_size=5); audio.write_bytes(b'changed-audio')
        self.assertNotEqual(key, transcripts.identity(orch, audio)[0])
        audio.write_bytes(b'audio-content'); (directory / 'model.bin').write_bytes(b'changed-weights')
        self.assertNotEqual(key, transcripts.identity(orch, audio)[0])

    def test_subtitles_only_match_original_name_through_local_symlink(self):
        path = self.make_video()
        matching = path.with_suffix('.zh-Hans.srt'); matching.write_text('1\n00:00:00,000 --> 00:00:01,000\n正确字幕\n', encoding='utf-8')
        (self.root / 'unrelated.srt').write_text('其他视频', encoding='utf-8')
        link = self.root / 'linked.mp4'; link.symlink_to(path)
        orch = Orchestrator(repository=None, config=self.cfg)
        self.assertEqual(orch._find_subtitles(link), [matching])

    def test_stream_visual_starts_before_producer_finishes_and_retry_reuses(self):
        from backend.analysis.fast import describe_stream
        seen = threading.Event(); calls = []
        class Runner:
            endpoint = 'https://test.invalid'; max_image_side = 1280
            def describe_multi(self, paths, prompt, **kwargs):
                calls.append(len(paths)); seen.set()
                return SimpleNamespace(text=json.dumps({'images':[{'index':i,'description':'画面','ocr':'750元'} for i in range(len(paths))]}),finish_reason='stop')
        frames = []
        for i in range(8):
            path = self.root / f'{i}.jpg'; path.write_bytes(bytes([i]))
            frames.append(SimpleNamespace(path=path,keyframe_id=str(i),video_id='v',timestamp_ms=i*1000,
                scene_index=0,clip_id=None,width=100,height=100,bytes=1,notes=()))
        def produced():
            yield from frames[:4]
            self.assertTrue(seen.wait(2), '必须边抽帧边请求，不等生成器完全结束')
            yield from frames[4:]
        orch = SimpleNamespace(_analysis_run=self.ledger, _cfg=SimpleNamespace(integrity=self.cfg.integrity,
            vl=SimpleNamespace(cloud_sequence_frames=8)),_vl=SimpleNamespace(_runner=Runner(),_model_name='test'))
        result = describe_stream(orch, produced(), SimpleNamespace(profile={}))
        self.assertEqual(len(result), 8); self.assertEqual(len(calls), 2)
        describe_stream(orch, iter(frames), SimpleNamespace(profile={}))
        self.assertEqual(len(calls), 2, '只补未完成帧')

    def test_pooled_transport_preserves_status_retry_after_and_sse(self):
        import httpx
        from backend.models.http_transport import PooledTransport
        pool = PooledTransport.__new__(PooledTransport)
        def handle(request):
            if request.url.path == '/limited':
                return httpx.Response(429, headers={'Retry-After':'2'}, text='limited')
            return httpx.Response(200, text='data: {"text":"回答"}\n\ndata: [DONE]\n')
        pool.client = httpx.Client(transport=httpx.MockTransport(handle))
        try:
            request = urllib.request.Request('https://test.invalid/events', data=b'{}')
            with pool.open(request, 2) as response:
                self.assertTrue(any(b'[DONE]' in line for line in response))
            with self.assertRaises(urllib.error.HTTPError) as error:
                with pool.open(urllib.request.Request('https://test.invalid/limited'),2):pass
            self.assertEqual(error.exception.code,429)
            self.assertEqual(error.exception.headers['Retry-After'],'2')
        finally:pool.close()

    def test_successful_visual_batch_survives_another_batch_failure(self):
        from backend.analysis.fast import describe_stream
        good = threading.Event()
        calls = []
        class Runner:
            endpoint = 'https://test.invalid'; max_image_side = 1280
            failing = True
            def describe_multi(self, paths, prompt, **kwargs):
                calls.append([path.name for path in paths])
                if paths[0].name == '4.jpg' and self.failing:
                    if not good.wait(3):
                        raise AssertionError('第一批未完成')
                    raise RuntimeError('模拟另一个批次失败')
                good.set()
                return SimpleNamespace(text=json.dumps({'images':[
                    {'index': i, 'description': '画面', 'ocr': '价格750元。'}
                    for i in range(len(paths))]}), finish_reason='stop')
        runner = Runner()
        frames = []
        for i in range(8):
            path = self.root / f'{i}.jpg'; path.write_bytes(bytes([i]))
            frames.append(SimpleNamespace(path=path, keyframe_id=str(i), video_id='v',
                timestamp_ms=i*1000, scene_index=0, clip_id=None, width=100, height=100, bytes=1, notes=()))
        orch = SimpleNamespace(_analysis_run=self.ledger, _cfg=SimpleNamespace(integrity=self.cfg.integrity,
            vl=SimpleNamespace(cloud_sequence_frames=4)), _vl=SimpleNamespace(_runner=runner, _model_name='fixture'))
        with self.assertRaises(RuntimeError):
            describe_stream(orch, iter(frames), SimpleNamespace(profile={}))
        runner.failing = False
        describe_stream(orch, iter(frames), SimpleNamespace(profile={}))
        self.assertEqual(sum('0.jpg' in batch for batch in calls), 1, '已完成第一批不得重跑')
        self.assertEqual(sum('4.jpg' in batch for batch in calls), 2)

    def test_connection_pool_respects_no_proxy_for_local_endpoint(self):
        import http.server
        import os
        from backend.models.http_transport import PooledTransport
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.end_headers(); self.wfile.write(b'local-endpoint')
            def log_message(self, *args):
                pass
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with patch.dict(os.environ, {'HTTP_PROXY': 'http://127.0.0.1:1',
                'HTTPS_PROXY': 'http://127.0.0.1:1', 'ALL_PROXY': 'http://127.0.0.1:1',
                'NO_PROXY': '127.0.0.1'}):
                pool = PooledTransport()
                try:
                    with pool.open(urllib.request.Request(f'http://127.0.0.1:{server.server_port}'), 2) as response:
                        self.assertEqual(response.read(), b'local-endpoint')
                finally:
                    pool.close()
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_decoder_preserves_rotated_display_and_nonzero_timestamps(self):
        original = self.make_video()
        path = self.root / 'rotated.mp4'
        subprocess.run([resolve_binary('ffmpeg'), '-y', '-loglevel', 'error', '-display_rotation:v:0', '90', '-i', str(original),
            '-c', 'copy', '-output_ts_offset', '5', str(path)],
            check=True, capture_output=True)
        info = probe_media(path)
        self.assertEqual(info.display_width, 90)
        orch = SimpleNamespace(_analysis_run=self.ledger, _cfg=self.cfg)
        frames = list(stream_frames(orch, path, 'rotated', info, SimpleNamespace(profile={}), AnalyzeOptions()))
        self.assertTrue(all(frame.width == 90 and frame.height == 160 for frame in frames))
        self.assertTrue(all(0 <= frame.timestamp_ms < 2400 for frame in frames),
                        '实际时间必须减去容器起始偏移，不把5秒偏移当作内容时间')

    def test_real_media_pipeline_has_no_scene_and_waits_for_index_and_report(self):
        path = self.make_video(audio=True)
        directory = self.root / 'model'; directory.mkdir(); (directory / 'model.bin').write_bytes(b'fixture')
        events = []
        class Encoder:
            def embed_texts(self, texts, **kwargs):return [[1., 0., 0., 0.] for text in texts]
            def embed_images(self, paths, **kwargs):return [[1., 0., 0., 0.] for path in paths]
        class Vectors:
            def with_generation(self, value):return self
            def upsert_clips(self, **kwargs):return len(kwargs['ids'])
            def upsert_frames(self, **kwargs):return len(kwargs['ids'])
            def drop_generation(self, value):pass
        class Asr:
            config = AsrConfig(device='cpu')
            def _model_path(self):return directory
            def transcribe(self, *args, **kwargs):
                return Transcript(video_id=kwargs['video_id'], segments=(TranscriptSegment(0, 0, 2200,
                    '型号A1价格750元。不能使用DDR4。', words=(Word('型号A1价格750元。不能使用DDR4。', 0, 2200),)),))
        class Model:
            def complete(self, messages, **kwargs):
                self.started = time.monotonic()
                time.sleep(.03)
                return SimpleNamespace(text='## 内容概览\n型号A1\n## 关键结论\n价格750元。', finish_reason='stop')
        class Runner:
            endpoint = 'https://test.invalid'; max_image_side = 1280
            def describe_multi(self, paths, prompt, **kwargs):
                return SimpleNamespace(text=json.dumps({'images':[{'index':i,'description':'测试画面','ocr':'价格750元。'}
                    for i in range(len(paths))]}),finish_reason='stop')
        runner = Runner()
        repo = SqliteRepository(self.cfg.path('sqlite'))
        from backend.pipeline.factory import _build_chunker
        orch = Orchestrator(repository=repo,config=self.cfg,embedder=Encoder(),asr=Asr(),
            vectors=Vectors(),llm=Model(),vl=SimpleNamespace(_runner=runner,_model_name='fixture'),
            chunker=_build_chunker(self.cfg))
        orch._analysis_run=self.ledger
        try:
            with self.assertLogs('zhiying', level='INFO') as logs:
                with patch.object(orch,'_detect_scenes',side_effect=AssertionError('禁止镜头检测')):
                    result=orch._analyze(str(path),AnalyzeOptions(progress=lambda stage, ratio, message:events.append((stage,ratio))))
            summaries = [line for line in logs.output if '分析耗时汇总' in line]
            self.assertEqual(len(summaries), 1)
            for label in ('语音转写', '画面理解', '抽帧', '建立向量索引', '生成总结', '总耗时', '并行流程'):
                self.assertIn(label, summaries[0])
            self.assertNotIn('镜头检测', summaries[0])
            self.assertTrue(result.ok, result.errors)
            self.assertNotIn('scene',[item.stage.value for item in result.stages])
            self.assertEqual(result.keyframe_count,4)
            self.assertTrue(result.summary)
            self.assertGreater(result.vector_count,0)
            self.assertEqual([ratio for _,ratio in events], sorted(ratio for _,ratio in events))
            stages={item.stage.value:item for item in result.stages}
            self.assertLess(abs(stages['summary'].started_at-stages['index'].started_at),.1)
            self.assertTrue(all(item.clip_id for item in repo.get_keyframes(result.video_id)))
        finally:
            repo.close()
            from backend.models.http_transport import close_pool
            close_pool(runner)

    def test_failed_parallel_analysis_still_logs_total(self):
        orch = Orchestrator(repository=None, config=self.cfg)
        orch._analysis_run = self.ledger
        with self.assertLogs('zhiying', level='INFO') as logs:
            with patch.object(orch, '_require_cloud_ready', side_effect=RuntimeError('模拟配置错误')):
                report = orch._analyze('fixture', AnalyzeOptions())
        self.assertEqual(report.status, 'failed')
        summaries = [line for line in logs.output if '分析耗时汇总' in line]
        self.assertEqual(len(summaries), 1)
        self.assertIn('status=failed', summaries[0]); self.assertIn('总耗时', summaries[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
