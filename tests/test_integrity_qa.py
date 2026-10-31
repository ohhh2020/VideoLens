"""新版问答、预算、局部/全局检索和旧版本只读边界。"""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.analysis.qa import ask, fit_messages, scope_of
from backend.analysis.policy import estimate_tokens
from backend.config import load_app_config
from backend.storage.repository import SqliteRepository


class FakeCloud:
    def __init__(self, sufficient=True):
        self.sufficient = sufficient
        self.messages = []

    def complete(self, messages, **kwargs):
        self.messages.append(messages)
        return SimpleNamespace(text=json.dumps({'sufficient': self.sufficient, 'query': '补充数字'}), finish_reason='stop')

    def stream_events(self, messages, **kwargs):
        self.messages.append(messages)
        yield 'delta', '视频中的参数为9。[1]'
        yield 'metadata', {'finish_reason': 'stop', 'prompt_tokens': 123, 'completion_tokens': 12}


class QaIntegrityTests(unittest.TestCase):
    def test_budgeted_evidence_reading_shrinks_without_losing_source_batches(self):
        with tempfile.TemporaryDirectory() as temp:
            policy = load_app_config(root=Path(temp)).integrity
            rows = [{'index': i + 1, 'text': f'独有事实{i}。' + '无关重复描述。' * 80 + f'尾部价格{i}元。'}
                    for i in range(12)]
            seen, caps = set(), []
            def read(messages, **kwargs):
                import re
                source = json.loads(messages[-1]['content'])['本轮证据']
                seen.update(row['index'] for row in source)
                caps.append(kwargs['max_tokens'])
                self.assertRegex(messages[-2]['content'], r'正文最多\d+个字符')
                tail = [part for row in source for part in re.findall(r'尾部价格\d+元。', row['text'])]
                return SimpleNamespace(text=''.join(tail) or '无直接相关事实。', finish_reason='stop')
            result = fit_messages(SimpleNamespace(complete=read), '分别说明价格', '测试视频', rows, [],
                                  policy, None, 8000, 2048)
            self.assertEqual(seen, set(range(1, 13)))
            self.assertTrue(all(value <= 2048 for value in caps))
            serialized = json.dumps(result, ensure_ascii=False)
            self.assertLessEqual(estimate_tokens(serialized) + 2048 + 1536, 8000)
            self.assertIn('尾部价格11元', serialized)

    def test_expanded_or_truncated_map_retries_with_tighter_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            policy = load_app_config(root=Path(temp)).integrity
            calls = []
            def read(messages, **kwargs):
                calls.append(kwargs['max_tokens'])
                if len(calls) % 2:
                    return SimpleNamespace(text='扩写而不是压缩。' * 200, finish_reason='length')
                return SimpleNamespace(text='价格降至750元。[1]', finish_reason='stop')
            result = fit_messages(SimpleNamespace(complete=read), '价格是多少', '视频',
                [{'index': 1, 'text': '价格750元。' * 300}], [], policy, None, 8000, 2048)
            self.assertIn('750元', result[-1]['content'])
            self.assertTrue(all(calls[i + 1] <= calls[i] for i in range(0, len(calls), 2)))

    def test_map_does_not_accept_overlong_text_or_slice_source_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            policy = load_app_config(root=Path(temp)).integrity
            original = '原始证据。' * 1000 + '尾部关键事实'
            rows = [{'index': 1, 'text': original}]
            cloud = SimpleNamespace(complete=lambda *args, **kwargs:
                SimpleNamespace(text='不受约束的扩写。' * 1000, finish_reason='stop'))
            with self.assertRaisesRegex(ValueError, '超出压缩预算'):
                fit_messages(cloud, '事实', '视频', rows, [], policy, None, 8000, 2048)
            self.assertEqual(rows[0]['text'], original)

    def test_overview_questions_use_global_chapter_index(self):
        for text in ('请说明这段视频的主要内容。','视频讲了什么？','请概括核心观点'):
            self.assertEqual(scope_of(text),'global')
        self.assertEqual(scope_of('第15秒的参数是多少？'),'local')

    def test_fact_index_visits_all_leaves_and_retains_provenance(self):
        from backend.analysis.qa import chapter_facts
        from backend.analysis.runs import RunLedger
        with tempfile.TemporaryDirectory() as temp:
            ledger=RunLedger(temp,'run')
            ledger.put('summary_tree',[
                {'id':str(i),'level':0,'clip_ids':['c'+str(i//10)],'summary':'说明',
                 'facts':['重复事实','独有数字'+str(i)],'evidence_ids':['e'+str(i)],
                 'start_ms':i*1000,'end_ms':(i+1)*1000} for i in range(161)])
            rows=chapter_facts(ledger,'v')
            self.assertEqual(len(rows),17)
            self.assertEqual(sum(r['text'].count('重复事实') for r in rows),17)
            self.assertTrue(all('独有数字'+str(i) in '\n'.join(r['text'] for r in rows) for i in range(161)))
            p=ledger.get('qa_fact_provenance')
            self.assertEqual({s for r in p for f in r['facts'] for s in f['source_ids']},{'e'+str(i) for i in range(161)})
            self.assertEqual(len(ledger.all_evidence()),17)

    def test_global_traverses_every_legacy_clip_and_local_keeps_constraints(self):
        for question, expected_k in [('请列出全部步骤与数字', 20), ('参数9是什么？'+'保留这个限制。'*150, 6)]:
            with self.subTest(question=question[:20]), tempfile.TemporaryDirectory() as temp:
                cfg = load_app_config(root=Path(temp), overrides={'integrity.disk_reserve_bytes': 0})
                with SqliteRepository(cfg.path('sqlite')) as repo:
                    repo.add_video(video_id='old', status='ready', title='旧视频')
                    repo.add_clips('old', [{'clip_id': f'c{i}', 'start_ms': i*1000, 'end_ms': (i+1)*1000,
                                           'text': f'参数9。独有事实{i}'} for i in range(35)])
                    cloud, retrieved = FakeCloud(), []
                    def retrieve(vid, query, k):
                        retrieved.append((query, k))
                        return [{'clip_id': 'c10'}]
                    orch = SimpleNamespace(_cfg=cfg, _repo=repo, _llm=cloud, _retrieve=retrieve,
                        _attach_visuals=lambda vid, rows: rows, _rewrite_query=lambda *args: '改写主题',
                        _preference_messages=lambda messages, text: messages, _notify_memory=lambda: None)
                    result = list(ask(orch, 'old', question, stream=True))[-1][1]
                    self.assertTrue(result.ok)
                    self.assertEqual(retrieved[0][1], expected_k)
                    self.assertEqual(result.usage['completion_tokens'], 12)
                    actual = json.loads(cloud.messages[-1][-1]['content'])
                    self.assertEqual(actual['原始问题'], question)
                    self.assertEqual(len(actual['本轮证据']), 35 if expected_k == 20 else 3)
                    self.assertTrue(result.refs)
                    self.assertEqual(result.refs[0].video_id, 'old')
                    self.assertEqual(len(repo.list_qa('old')), 1)

    def test_legacy_insufficient_evidence_never_opens_source(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = load_app_config(root=Path(temp))
            with SqliteRepository(cfg.path('sqlite')) as repo:
                repo.add_video(video_id='old', status='ready', local_path='must-not-open.mp4')
                repo.add_clips('old', [{'clip_id': 'c', 'text': '参数9', 'end_ms': 1000}])
                queries = []
                def retrieve(vid, query, k):
                    queries.append(query)
                    return [{'clip_id': 'c'}]
                orch = SimpleNamespace(_cfg=cfg, _repo=repo, _llm=FakeCloud(False), _retrieve=retrieve,
                    _attach_visuals=lambda vid, rows: rows, _rewrite_query=lambda *args: '',
                    _preference_messages=lambda messages, text: messages, _notify_memory=lambda: None)
                with patch('backend.analysis.supplement.supplement', side_effect=AssertionError('旧视频禁止补帧')):
                    result = list(ask(orch, 'old', '参数9是否大于8？'))[-1][1]
                self.assertTrue(result.ok)
                self.assertEqual(len(queries), 3)
                self.assertTrue(all('参数9是否大于8？' in value for value in queries))

    def test_attachment_followup_is_not_lost_without_then(self):
        from backend.assistant.intent import understand
        llm = SimpleNamespace(complete=lambda *a, **k: SimpleNamespace(text=json.dumps({'action':'analyze','source_index':0,'question':'','reply':''})))
        text = '请核对 value_001 和 limit_001 的具体数值。'
        intent = understand(llm,text,urls=[],attachment={'filename':'test.mp4','path':'safe.mp4'})
        self.assertEqual(intent.question,text)
        only = understand(llm,'请分析这个视频',urls=[],attachment={'filename':'test.mp4','path':'safe.mp4'})
        self.assertEqual(only.question,'')

    def test_ocr_native_single_image_has_unambiguous_source(self):
        from backend.analysis.vision import normalize_ocr
        wrapped = normalize_ocr([{'images': [[{'index':0,'text':'wrapped','uncertain':False}]]}],1)
        self.assertEqual(wrapped[0]['text'],'wrapped')
        rows = normalize_ocr([{'text': '代码\n    value=123', 'bbox_2d': [0,0,500,1000]}], 1)
        self.assertEqual(rows[0]['index'], 0)
        self.assertIn('    value=123', rows[0]['text'])
        self.assertTrue(rows[0]['uncertain'])
        with self.assertRaises(ValueError):
            normalize_ocr([{'text': '不能猜多图归属'}], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
