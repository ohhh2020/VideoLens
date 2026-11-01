"""完整性版本：真实SQLite、边界分窗、全量摘要、版本发布与分块上传。"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from dataclasses import replace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.analysis.evidence import compose, split_text, video_windows
from backend.analysis.policy import IntegritySection
from backend.analysis.runs import RunLedger, publish
from backend.analysis.summary import summarize
from backend.config import load_app_config
from backend.storage.repository import SqliteRepository
from backend.storage.conversations import ConversationStore
from backend.assistant.resumable import initialize, write_chunk, finalize, state


class Cloud:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, **options):
        self.calls += 1
        data = json.loads(messages[-1]['content'])
        if isinstance(data, list):
            output = {'items': [{'id': row['id'], 'summary': row.get('text', '')[:120],
                                 'facts': [row.get('text', '')[-80:]]} for row in data]}
            text = json.dumps(output, ensure_ascii=False)
        else:
            text = '## 内容概览\n全部章节均已处理\n## 主要脉络\n已覆盖全片\n## 关键结论\n保留原文事实'
        return SimpleNamespace(text=text, finish_reason='stop', prompt_tokens=100, completion_tokens=50)


class IntegrityTests(unittest.TestCase):
    def test_upper_summary_carries_all_outlines_without_repeating_fact_inventory(self):
        class VerboseFacts(Cloud):
            def complete(self,messages,**options):
                data=json.loads(messages[-1]['content'])
                if isinstance(data,list):
                    return SimpleNamespace(text=json.dumps({'items':[{'id':r['id'],'summary':'章节核心数字123，按步骤完成',
                        'facts':['详细事实数字123。'*80]} for r in data]},ensure_ascii=False),finish_reason='stop')
                return super().complete(messages,**options)
        with tempfile.TemporaryDirectory() as temp:
            ledger=RunLedger(temp,'outline')
            clips=[{'clip_id':str(i),'start_ms':i*1000,'end_ms':(i+1)*1000,'text':'原始唯一事实'+str(i)} for i in range(33)]
            repo=SimpleNamespace(iter_clip_payloads=lambda v: iter(clips),get_segments=lambda v: [],get_keyframes=lambda v: [])
            summarize(repo,'v','层级',VerboseFacts(),IntegritySection(),ledger)
            tree=ledger.get('summary_tree')
            self.assertEqual(ledger.get('coverage')['covered_clips'],33)
            self.assertTrue(any(n['level']>0 for n in tree))
            self.assertTrue(all(n['facts'] for n in tree))
            self.assertLess(max(n['level'] for n in tree),4)

    def test_single_evidence_truncation_splits_text_and_preserves_source(self):
        class LimitedCloud(Cloud):
            def complete(self, messages, **options):
                data=json.loads(messages[-1]['content'])
                if isinstance(data,list) and any(len(row['text'].encode())>128 for row in data):
                    return SimpleNamespace(text='{}',finish_reason='length')
                return super().complete(messages,**options)
        with tempfile.TemporaryDirectory() as temp:
            ledger=RunLedger(temp,'single')
            text='参数123和单位毫秒。'*20+'末尾唯一数值987654'
            repo=SimpleNamespace(iter_clip_payloads=lambda v: iter([{'clip_id':'c','start_ms':0,'end_ms':1000,'text':text}]),
                                 get_segments=lambda v: [],get_keyframes=lambda v: [])
            summarize(repo,'v','单证据',LimitedCloud(),IntegritySection(),ledger)
            self.assertEqual(ledger.get('coverage')['covered_evidence_count'],1)
            self.assertTrue(any('987654' in str(n) for n in ledger.get('summary_tree')))

    def test_structured_mode_is_sent_in_http_body_not_response_serialization(self):
        from backend.models.llm import OpenAICompatibleBackend, LlmMessage, LlmResponse
        client=OpenAICompatibleBackend('https://invalid','test','key')
        body=json.loads(client._payload([LlmMessage('user','json')],2048,0.0,(),False,json_mode=True))
        self.assertEqual(body['response_format'],{'type':'json_object'})
        self.assertNotIn('response_format',json.loads(client._payload([],100,0.0,(),False)))
        self.assertEqual(LlmResponse(text='ok',backend='cloud').to_dict()['text'],'ok')

    def test_word_alignment_preserves_original_spaces_numbers_and_tail(self):
        from backend.analysis.summary import aligned_words, collect
        for text, tokens in [('Item 1: value 100137. Tail!', ['Item', '1', 'value', '100137', 'Tail']),
                             ('中文：数值100137，尾部。', ['中文', '数值', '100137', '尾部'])]:
            segment = {'start_ms': 0, 'end_ms': len(tokens)*1000, 'text': text,
                       'words': [{'word': token, 'start_ms': i*1000, 'end_ms': (i+1)*1000} for i, token in enumerate(tokens)]}
            self.assertEqual(''.join(w['word'] for w in aligned_words(segment)), text)
            with tempfile.TemporaryDirectory() as temp:
                ledger = RunLedger(temp, 'alignment')
                ledger.put('transcript_raw', {'segments': [segment]})
                clips = [{'clip_id': str(i), 'start_ms': i*1000, 'end_ms': (i+1)*1000} for i in range(len(tokens))]
                repo = SimpleNamespace(iter_clip_payloads=lambda v: iter(clips), get_segments=lambda v: [], get_keyframes=lambda v: [])
                _, rows = collect(repo, 'v', ledger, IntegritySection())
                # 子证据保留原文片段的空格、格式与尾部。
                self.assertTrue(all(row['text'] in text for row in rows if row['kind'] == 'speech'))
                self.assertTrue(any('尾部。' in row['text'] or 'Tail!' in row['text'] for row in rows))
                self.assertFalse(any(row['kind'] == 'speech_unaligned' for row in rows))
        self.assertEqual(aligned_words({'text': '原文', 'words': [{'word': '错误'}]}), [])
        self.assertEqual(aligned_words({'text': 'value 100137. Tail fact', 'words': [{'word': 'value'}, {'word': '100137'}]}), [])

    def test_upload_boundaries_missing_hash_expiry_and_reopen(self):
        from fastapi import HTTPException
        from backend.assistant.resumable import clean
        with tempfile.TemporaryDirectory() as temp:
            cfg = load_app_config(root=Path(temp), overrides={'integrity.disk_reserve_bytes': 0})
            repo = SqliteRepository(cfg.path('sqlite'))
            store = ConversationStore(repo)
            cid = store.create()['id']
            service = SimpleNamespace(store=store, config=cfg)
            body = b'test-video'
            digest = hashlib.sha256(body).hexdigest()
            uid = initialize(service, cid, 'test.mp4', len(body), digest)['upload_id']
            with self.assertRaises(HTTPException) as err:
                finalize(service, cid, uid, digest)
            self.assertEqual(err.exception.status_code, 409)
            with self.assertRaises(HTTPException) as err:
                write_chunk(service, cid, uid, 0, body, '0'*64)
            self.assertEqual(err.exception.status_code, 422)
            write_chunk(service, cid, uid, 0, body, digest)
            repo.close()
            with SqliteRepository(cfg.path('sqlite')) as reopened:
                service.store = ConversationStore(reopened)
                self.assertEqual(json.loads(state(service, cid, uid)['chunks']), {'0': digest})
                exact = initialize(service, cid, 'large.mp4', 8*1024**3, '1'*64)
                self.assertEqual(exact['size_bytes'], 8*1024**3)
                with self.assertRaises(HTTPException) as err:
                    initialize(service, cid, 'large.mp4', 8*1024**3+1, '2'*64)
                self.assertEqual(err.exception.status_code, 413)
                part = Path(state(service, cid, uid)['path'])
                with reopened._tx() as conn:
                    conn.execute('UPDATE upload_sessions SET updated_ts=0 WHERE id=?', (uid,))
                clean(service)
                self.assertFalse(part.exists())
                self.assertEqual(state(service, cid, exact['upload_id'])['size_bytes'], 8*1024**3)

    def test_retry_parse_does_not_erase_completed_stage_checkpoint(self):
        from backend.pipeline.orchestrator import Orchestrator
        from backend.pipeline.domain import AnalysisReport, AnalyzeOptions, Stage
        with tempfile.TemporaryDirectory() as temp:
            ledger = RunLedger(temp, 'retry')
            ledger.put('progress', {'stage': 'vision'})
            orch = SimpleNamespace(_analysis_run=ledger, _emit=lambda *args: None)
            Orchestrator._begin(orch, AnalysisReport(), Stage.PARSE, None)
            self.assertEqual(Orchestrator._resume_point(orch, 'v', AnalyzeOptions()), Stage.VISION)

    def test_summary_boundaries_and_long_tail_and_failure_are_visible(self):
        for count in (1, 32, 33, 161):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temp:
                ledger = RunLedger(temp, 'boundary')
                clips = [{'clip_id': f'c{i}', 'start_ms': i*1000, 'end_ms': (i+1)*1000,
                          'text': ('很长的原始证据。'*1400+'独有尾部数字987654') if i == count-1 else f'事实{i}'} for i in range(count)]
                repo = SimpleNamespace(iter_clip_payloads=lambda vid: iter(clips), get_segments=lambda vid: [], get_keyframes=lambda vid: [])
                summarize(repo, 'v', '边界', Cloud(), IntegritySection(), ledger)
                self.assertEqual(ledger.get('coverage')['covered_clips'], count)
                self.assertTrue(any('987654' in row['text'] for row in ledger.all_evidence()))
                bad = SimpleNamespace(complete=lambda *a, **k: SimpleNamespace(text='{}', finish_reason='length'))
                failed = RunLedger(Path(temp)/'failed', 'failed')
                with self.assertRaises(ValueError):
                    summarize(repo, 'v', '失败', bad, IntegritySection(), failed)
                self.assertIsNone(failed.get('coverage'))

    def test_migration_backup_and_existing_ready_result_are_read_only(self):
        import sqlite3
        from backend.analysis.runs import execute
        from backend.pipeline.domain import AnalyzeOptions
        with tempfile.TemporaryDirectory() as temp:
            cfg = load_app_config(root=Path(temp))
            path = cfg.path('sqlite')
            repo = SqliteRepository(path)
            repo.add_video(video_id='legacy', title='旧报告')
            repo.close()
            from contextlib import closing
            with closing(sqlite3.connect(path)) as conn:
                conn.execute('DROP TABLE upload_sessions')
                conn.execute('DROP TABLE analysis_runs')
                conn.execute('PRAGMA user_version=7')
                conn.commit()
            with SqliteRepository(path) as repo:
                backup = path.with_name(path.name+'.before-v8.bak')
                self.assertTrue(backup.exists())
                with closing(sqlite3.connect(backup)) as conn:
                    self.assertEqual(conn.execute('PRAGMA user_version').fetchone()[0], 7)
                self.assertEqual(repo.find_video('legacy').title, '旧报告')
                file = Path(temp)/'video.mp4'
                file.write_bytes(b'cached-local-source')
                from backend.media.parser import parse_source, ParseOptions
                vid = parse_source(file, ParseOptions(do_probe=False)).video_id
                repo.add_video(video_id=vid, status='ready')
                repo.set_video_report(vid, {'video_id': vid, 'status': 'ready', 'summary': '既有报告', 'profile': {}})
                timings = []
                parent = SimpleNamespace(_cfg=cfg, _repo=repo, _require_cloud_ready=lambda: None,
                    _log_timings=timings.append,
                    _parse_source=lambda *a: self.fail('旧报告复用不得重新探测或触发分析'))
                report = execute(parent, str(file), AnalyzeOptions())
                self.assertTrue(report.ok, report.errors)
                self.assertEqual(report.summary, '既有报告')
                self.assertEqual(len(timings), 1)
                self.assertTrue(timings[0].stages[0].skipped)
                self.assertEqual(repo._query_one('SELECT COUNT(*) FROM analysis_runs')[0], 0)

    def test_cloud_limit_retry_and_stream_metadata(self):
        from backend.analysis.cloud import request, http_error
        from backend.models.llm import LlmError, OpenAICompatibleBackend, LlmMessage
        from urllib.error import HTTPError
        from email.message import Message
        import io
        headers = Message(); headers['Retry-After'] = '0'
        attempts = []
        def transient():
            attempts.append(1)
            raise http_error(LlmError('HTTP 429'), HTTPError('https://invalid', 429, '', headers, None))
        with patch('backend.analysis.cloud.time.sleep') as wait:
            with self.assertRaises(LlmError):
                request(None, IntegritySection(), '重试', {}, transient)
            self.assertEqual(len(attempts), 4)
            self.assertEqual(wait.call_count, 3)
        body = ('data: {"choices":[{"delta":{"content":"答案"}}]}\n\n'
                'data: {"choices":[{"finish_reason":"length","delta":{}}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":123,"completion_tokens":456}}\n\n'
                'data: [DONE]\n\n').encode()
        backend = OpenAICompatibleBackend('https://invalid', 'test', 'test-key')
        with patch('urllib.request.urlopen', return_value=io.BytesIO(body)):
            events = list(backend.stream_events([LlmMessage('user', '测试')]))
        self.assertEqual(events[0], ('delta', '答案'))
        self.assertEqual(events[-1][1]['finish_reason'], 'length')
        self.assertEqual(events[-1][1]['completion_tokens'], 456)

    def test_quality_validation_and_disk_inventory(self):
        from backend.analysis.resources import guard, inventory
        with tempfile.TemporaryDirectory() as temp:
            cfg = load_app_config(root=Path(temp), overrides={'integrity.disk_reserve_bytes': 0})
            frame = cfg.path('data')/'analysis-runs'/'test'/'ocr'/'frame.png'
            frame.parent.mkdir(parents=True);frame.write_bytes(b'abc')
            self.assertEqual(inventory(cfg, refresh=True)['frame_bytes'], 3)
            guard(cfg, 'frames', 100)
            limited = replace(cfg, storage=replace(cfg.storage, limits=replace(cfg.storage.limits, max_keyframe_bytes=2)))
            with self.assertRaises(OSError):
                guard(limited, 'frames', 100)
            for key, value in [('scan_fps', 0), ('change_threshold', float('nan')), ('cloud_total', 33), ('gpu_batch_max', 17)]:
                with self.assertRaises(ValueError):
                    replace(cfg.integrity, **{key: value}).validate()

    def test_sources_and_long_text_are_not_lost(self):
        text = compose({'core_text': '语音', 'text': '语音', 'visual_text': '独有画面', 'ocr_text': 'X=42'})
        self.assertIn('独有画面', text)
        self.assertIn('X=42', text)
        original = ('多字节🙂；' * 3000) + '最后一个独有事实'
        parts = split_text(original, 300)
        self.assertEqual(''.join(parts), original)
        self.assertTrue(all(len(part.encode()) <= 300 for part in parts))

    def test_windows_never_truncate_long_scenes_or_tail(self):
        for seconds in (14, 15, 16, 45, 60, 64):
            windows = video_windows([SimpleNamespace(start_ms=0, end_ms=seconds*1000)], 15)
            self.assertEqual(windows[0].start_ms, 0)
            self.assertEqual(windows[-1].end_ms, seconds*1000)
            self.assertTrue(all(0 < row.end_ms-row.start_ms <= 15000 for row in windows))
            self.assertTrue(all(a.end_ms == b.start_ms for a, b in zip(windows, windows[1:])))

    def test_all_161_clips_enter_summary_and_cached_batches_are_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = RunLedger(temp, 'test-run')
            clips = [{'clip_id': f'c{i}', 'start_ms': i*20000, 'end_ms': (i+1)*20000,
                      'core_text': f'第{i}段证据，独有数据值{i}，最后步骤{i}'} for i in range(161)]
            repo = SimpleNamespace(iter_clip_payloads=lambda vid: iter(clips), get_segments=lambda vid: [], get_keyframes=lambda vid: [])
            cloud = Cloud()
            summary = summarize(repo, 'v1', '测试', cloud, IntegritySection(), ledger)
            self.assertIn('关键结论', summary)
            self.assertEqual(ledger.get('coverage')['covered_clips'], 161)
            self.assertEqual({cid for row in ledger.get('summary_tree') for cid in row['clip_ids']}, {row['clip_id'] for row in clips})
            self.assertTrue(any('最后步骤160' in str(row) for row in ledger.get('summary_tree')))
            calls = cloud.calls
            summarize(repo, 'v1', '测试', cloud, IntegritySection(), ledger)
            self.assertEqual(cloud.calls, calls)
            self.assertGreater(calls, 2)

    def test_publication_preserves_conversations_and_replaces_atomically(self):
        with tempfile.TemporaryDirectory() as temp:
            with SqliteRepository(Path(temp)/'main.db') as main, SqliteRepository(Path(temp)/'stage.db') as stage:
                main.add_video(video_id='v1', title='旧报告')
                main.set_video_report('v1', {'summary': '原报告'})
                conversation = ConversationStore(main).create(video_id='v1')
                main.add_qa('v1', question='原问题', answer='原回答', session_id=conversation['id'])
                stage.add_video(video_id='v1', title='新报告')
                stage.set_video_report('v1', {'summary': '新报告'})
                stage.add_clips('v1', [{'clip_id': 'c-new', 'text': '新证据'}])
                with patch.object(main, '_assert_capacity', side_effect=RuntimeError('容量不足')):
                    with self.assertRaises(RuntimeError):
                        publish(main, stage, 'v1')
                self.assertEqual(main.get_video_report('v1')['summary'], '原报告')
                publish(main, stage, 'v1')
                self.assertEqual(main.get_video_report('v1')['summary'], '新报告')
                self.assertEqual(len(main.list_qa('v1')), 1)
                self.assertEqual(ConversationStore(main).get(conversation['id'])['video_id'], 'v1')

    def test_resumable_chunks_survive_reopen_and_are_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = load_app_config(root=Path(temp), overrides={'integrity.disk_reserve_bytes': 0})
            with SqliteRepository(cfg.path('sqlite')) as repo:
                store = ConversationStore(repo)
                cid = store.create()['id']
                service = SimpleNamespace(store=store, config=cfg)
                chunk = 8 * 1024**2
                body = b'0' * (chunk * 2 + 10)
                digest = hashlib.sha256(body).hexdigest()
                upload = initialize(service, cid, 'test.mp4', len(body), digest)
                uid = upload['upload_id']
                for index in (2, 0, 1):
                    block = body[index*chunk:index*chunk+chunk]
                    sha = hashlib.sha256(block).hexdigest()
                    write_chunk(service, cid, uid, index, block, sha)
                    write_chunk(service, cid, uid, index, block, sha)
                self.assertEqual(len(json.loads(state(service, cid, uid)['chunks'])), 3)
                self.assertEqual(initialize(service, cid, 'test.mp4', len(body), digest)['upload_id'], uid)
                probe = SimpleNamespace(has_video=True, width=1920, height=1080, duration_ms=600000)
                with patch('backend.assistant.resumable.probe_media', return_value=probe):
                    result = finalize(service, cid, uid, digest)
                    self.assertEqual(finalize(service, cid, uid, digest), result)
                self.assertEqual(Path(store.attachment(result['id'], cid)['path']).read_bytes(), body)


if __name__ == '__main__':
    unittest.main(verbosity=2)
