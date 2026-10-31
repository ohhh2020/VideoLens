"""智能工作台：真实 SQLite/ASGI，注入模型验证编排与故障边界。"""
from __future__ import annotations

import json
import asyncio
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.assistant.intent import explicit_followup, extract_urls, understand
from backend.assistant.service import AssistantService
from backend.assistant.titles import generate_title
from backend.assistant.uploads import cleanup_uploads, receive_upload
from backend.config import load_app_config
from backend.media.ffmpeg_utils import resolve_binary
from backend.pipeline.domain import AnalysisReport
from backend.pipeline.orchestrator import Orchestrator
from backend.storage.conversations import ConversationStore
from backend.storage.repository import SqliteRepository, _MIGRATIONS, SCHEMA_VERSION
from backend.web.routes.conversations import create_router
from backend.web.routes.videos import create_router as create_video_router


class FakeLlm:
    def __init__(self):
        self.decision = None
        self.calls = 0
        self.gate = None
        self.title_calls = 0
        self.title_gate = None
        self.title_started = threading.Event()
        self.title_response = '{"title":"测试视频核心结论"}'
        self.title_error = False
        self.title_analysis = ''

    def complete(self, messages, **kwargs):
        self.calls += 1
        if messages[0]['role'] == 'system':
            self.title_calls += 1
            self.title_analysis = json.loads(messages[-1]['content'])['视频分析内容']
            self.title_started.set()
            if self.title_gate:
                self.title_gate.wait(5)
            if self.title_error:
                raise TimeoutError('命名调用超时')
            return SimpleNamespace(text=self.title_response)
        if self.gate:
            self.gate.wait(5)
        payload = json.loads(messages[0]['content'])
        if self.decision is not None:
            data = self.decision
        elif payload['当前视频']:
            data = {'action': 'ask', 'question': payload['用户原文']}
        elif payload['候选来源']:
            data = {'action': 'analyze', 'source_index': 0,
                    'question': '解释主要依据' if '依据' in payload['用户原文'] else ''}
        else:
            data = {'action': 'request_video', 'reply': '请上传视频或提供视频链接。'}
        return SimpleNamespace(text=json.dumps(data, ensure_ascii=False))


class FakeOrchestrator:
    def __init__(self, repo):
        self.repository = repo
        self._llm = FakeLlm()
        self.calls = 0
        self.ask_calls = 0
        self.fail_analysis = False
        self.fail_ask = False

    def _require_cloud_ready(self):
        pass

    def delete(self, video_id, *, missing_ok=False):
        worker=Orchestrator(repository=self.repository,config=self._cfg,vectors=getattr(self,'_delete_vectors',None))
        worker._use_integrity=True
        return worker.delete(video_id,missing_ok=missing_ok)

    def analyze(self, source, options):
        self.calls += 1
        options.progress('probe', .3, '视频探测完成')
        if self.fail_analysis:
            raise RuntimeError('分析故障')
        video = self.repository.find_video('vfixture')
        if not video:
            self.repository.add_video(video_id='vfixture', title='测试视频', source_url=source, duration_ms=1000)
        self.repository.update_video_status('vfixture', 'ready')
        report = AnalysisReport(video_id='vfixture', title='测试视频', duration_ms=1000, status='ready',
                                summary='## 内容概览\n测试视频\n## 关键结论\n测试结论')
        self.repository.set_video_report('vfixture', report.to_dict())
        return report

    def ask_stream(self, vid, question, session_id):
        self.ask_calls += 1
        if self.fail_ask:
            raise RuntimeError('问答故障')
        self.repository.add_qa(vid, question=question, answer='有证据的答案 [1]', session_id=session_id)
        yield 'delta', '有证据的'
        yield 'delta', '答案 [1]'
        yield 'final', SimpleNamespace(to_dict=lambda: {'answer': '有证据的答案 [1]', 'refs': [], 'elapsed_ms': 5})


class ConversationTests(unittest.TestCase):
    def test_bound_followup_bypasses_wrong_cloud_intent(self):
        self.orch._llm.decision={'action':'analyze','source_index':0}
        for question in ('视频中测试的指标有哪些','再总结一下这个视频','这个视频提到的另一个视频有哪些指标',
                         '为什么总叫我分析另一个视频','不要分析另一个视频，解释当前视频的指标',
                         '根据这个参考链接解释视频中的指标 https://example.org/reference'):
            intent=understand(self.orch._llm,question,urls=extract_urls(question),video_id='bound-video')
            self.assertEqual(intent.action,'ask')
            self.assertEqual(intent.question,question)
        self.assertEqual(self.orch._llm.calls,0)
        for question in ('请分析另一个视频','帮我分析这个视频 https://example.org/other-video','换个视频'):
            intent=understand(self.orch._llm,question,urls=extract_urls(question),video_id='bound-video')
            self.assertEqual(intent.action,'clarify')
        self.assertEqual(understand(self.orch._llm,'',urls=[],video_id='bound-video',
            attachment={'filename':'new.mp4','path':'safe.mp4'}).action,'clarify')
        with self.assertRaises(ValueError):
            understand(self.orch._llm,'问题'*1001,urls=[],video_id='bound-video')
        self.assertEqual(self.orch._llm.calls,0)

    def test_metrics_followup_answers_in_same_conversation_without_reanalysis(self):
        first=self.submit('https://example.org/video','video')
        self.wait(first)
        self.orch._llm.decision={'action':'analyze','source_index':0}
        before=self.orch._llm.calls
        oid=self.submit('视频中测试的指标有哪些','metrics')
        self.wait(oid)
        snapshot=self.client.get(f'/internal/conversations/{self.cid}').json()
        self.assertEqual(snapshot['operation']['status'],'done')
        self.assertEqual(snapshot['messages'][-1]['kind'],'answer')
        self.assertEqual(self.orch.calls,1)
        self.assertEqual(self.orch.ask_calls,1)
        self.assertEqual(self.orch._llm.calls,before)
        self.assertEqual(len(self.service.store.list()),1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = load_app_config(root=self.root)
        self.repo = SqliteRepository(self.cfg.path('sqlite'))
        self.orch = FakeOrchestrator(self.repo)
        self.orch._cfg = self.cfg
        self.service = AssistantService(self.orch, self.cfg)
        self.app = FastAPI()
        self.app.state.config = self.cfg
        self.app.state.orchestrator = self.orch
        self.app.state.assistant = self.service
        self.app.include_router(create_router(), prefix='/internal')
        self.app.include_router(create_video_router(), prefix='/internal')
        self.client = TestClient(self.app)
        self.cid = self.service.store.create()['id']

    def tearDown(self):
        if self.orch._llm.gate:
            self.orch._llm.gate.set()
        if self.orch._llm.title_gate:
            self.orch._llm.title_gate.set()
        self.service.close()
        self.client.close()
        self.repo.close()
        self.temp.cleanup()

    def submit(self, text, key='request1', **kwargs):
        response = self.client.post(f'/internal/conversations/{self.cid}/messages',
                                    json={'text': text, 'request_id': key, **kwargs})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()['operation_id']

    def wait(self, oid):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            op = self.service.store.operation(oid)
            if op['status'] not in {'queued', 'running'}:
                return op
            time.sleep(.01)
        self.fail('操作未在测试时限内完成')

    def test_title_generated_from_analysis_once_and_persisted(self):
        self.orch._llm.title_response = '{"title":"人工智能教学视频的完整核心内容总结与实践"}'
        operation = self.wait(self.submit('https://example.org/a.mp4'))
        self.assertEqual(operation['status'], 'done')
        conversation = self.service.store.get(self.cid)
        self.assertEqual(conversation['title'], '人工智能教学视频的完整核心内容')
        self.assertEqual(len(conversation['title']), 15)
        self.assertEqual(conversation['title_source'], 'generated')
        self.assertIn('关键结论', self.orch._llm.title_analysis)
        self.assertNotIn('example.org', self.orch._llm.title_analysis)
        self.assertEqual(self.orch._llm.title_calls, 1)
        self.wait(self.submit('为什么？', 'next'))
        self.wait(self.submit('为什么？', 'next'))
        self.assertEqual(self.orch._llm.title_calls, 1)
        self.service.close()
        self.service = AssistantService(self.orch, self.cfg)
        self.assertEqual(self.service.store.get(self.cid)['title'], conversation['title'])

    def test_manual_title_before_or_during_generation_is_preserved(self):
        self.service.store.update(self.cid, title='我的手动标题')
        self.wait(self.submit('https://example.org/a'))
        self.assertEqual(self.service.store.get(self.cid)['title'], '我的手动标题')
        self.assertEqual(self.orch._llm.title_calls, 0)
        self.cid = self.service.store.create()['id']
        self.orch._llm.title_gate = threading.Event()
        oid = self.submit('https://example.org/b', 'concurrent')
        self.assertTrue(self.orch._llm.title_started.wait(3))
        result = self.client.patch('/internal/conversations/' + self.cid, json={'title': '新会话'})
        self.assertEqual(result.status_code, 200)
        self.orch._llm.title_gate.set()
        self.assertEqual(self.wait(oid)['status'], 'done')
        self.assertEqual(self.service.store.get(self.cid)['title'], '新会话')
        self.assertEqual(self.service.store.get(self.cid)['title_source'], 'manual')

    def test_title_failure_preserves_report_and_question_and_can_retry_later(self):
        self.orch._llm.title_error = True
        op = self.wait(self.submit('https://example.org/a，解释依据'))
        self.assertEqual(op['status'], 'done')
        self.assertEqual([m['kind'] for m in self.service.store.messages(self.cid)], ['text', 'report', 'answer'])
        self.assertEqual(self.service.store.get(self.cid)['title_source'], 'pending')
        self.orch._llm.title_error = False
        self.wait(self.submit('为什么？', 'next'))
        self.assertEqual(self.orch._llm.title_calls, 2)
        self.assertEqual(self.orch.calls, 1)
        self.assertEqual(self.service.store.get(self.cid)['title_source'], 'generated')

    def test_new_session_for_saved_report_gets_model_title_without_reanalysis(self):
        self.wait(self.submit('https://example.org/a'))
        response = self.client.post('/internal/conversations', json={'video_id': 'vfixture'})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()['title'], '测试视频核心结论')
        self.assertEqual(response.json()['title_source'], 'generated')
        self.assertEqual(self.orch.calls, 1)
        self.assertEqual(self.orch._llm.title_calls, 2)

    def test_title_validation_and_skipped_summary_use_clip_evidence(self):
        for response in ['', '{}', '{"title":123}', '{"title":""}', '{"title":"https://example.org/a"}']:
            self.orch._llm.title_response = response
            with self.assertRaises(ValueError):
                generate_title(self.orch._llm, '实际视频分析')
        self.orch._llm.title_response = '```json\n{"title":"「 Python入门教程 」"}\n```'
        self.assertEqual(generate_title(self.orch._llm, '实际视频分析'), 'Python入门教程')
        self.repo.add_video(video_id='vclip', title='file.mp4')
        self.repo.add_clips('vclip', [{'clip_id': 'clip1', 'text': 'Python变量和函数使用说明'}])
        self.service.ensure_title(self.cid, {'video_id': 'vclip', 'summary': ''})
        self.assertEqual(self.orch._llm.title_analysis, 'Python变量和函数使用说明')
        self.assertEqual(self.service.store.get(self.cid)['title_source'], 'generated')

    def test_v6_migration_preserves_legacy_names_and_manual_rename_limit(self):
        path = self.root / 'v6.db'
        conn = sqlite3.connect(path)
        for version in range(1, 7):
            for statement in _MIGRATIONS[version]:
                conn.execute(statement)
        conn.execute("INSERT INTO conversations VALUES ('cold','原有的自定义标题','','cold',1,1,NULL)")
        conn.execute('PRAGMA user_version=6')
        conn.commit()
        conn.close()
        with SqliteRepository(path) as repository:
            conversation = ConversationStore(repository).get('cold')
            self.assertEqual(conversation['title'], '原有的自定义标题')
            self.assertEqual(conversation['title_source'], 'legacy')
        endpoint = '/internal/conversations/' + self.cid
        self.assertEqual(self.client.patch(endpoint, json={'title': '字' * 16}).status_code, 422)
        self.assertEqual(self.client.patch(endpoint, json={'title': '字' * 15}).status_code, 200)
        self.assertEqual(self.service.store.get(self.cid)['title_source'], 'manual')

    def test_url_extract_and_model_cannot_invent_source(self):
        self.assertEqual(extract_urls('分享 https://example.org/a?q=1，解释。 [视频](https://example.org/b)。'),
                         ['https://example.org/a?q=1', 'https://example.org/b'])
        self.assertEqual(extract_urls('https://user:secret@example.org/a file:///C:/x'), [])
        self.orch._llm.decision = {'action': 'analyze', 'source_index': 99, 'source': 'C:/Windows'}
        with self.assertRaisesRegex(ValueError, '不存在'):
            understand(self.orch._llm, '分析', urls=['https://example.org/a'])

    def test_explicit_followup_is_preserved_when_model_omits_it(self):
        self.orch._llm.decision = {'action': 'analyze', 'source_index': 0, 'question': ''}
        intent = understand(self.orch._llm, '分析 https://example.org/a，然后告诉我画面主体是什么。',
                            urls=['https://example.org/a'])
        self.assertEqual(intent.question, '告诉我画面主体是什么。')

    def test_shared_title_and_markdown_preserve_question_when_model_omits_it(self):
        url = 'https://www.bilibili.com/video/BV1pRxk6oEM3?vd_source=34523949c81717f3360d3a260d71234e'
        title = '【【26年11月CPU推荐】intel多款处理器来到历史最低最低价，小白选择这些型号不踩坑！】'
        question = '为什么这些cpu会出现历史最低点?'
        self.orch._llm.decision = {'action': 'analyze', 'source_index': 0, 'question': ''}
        cases = [
            (title + url + '\n\n' + question, question),
            (title + f'[{url}]({url}) ' + question, question),
            (f'[为什么CPU降价？]({url})\n' + question, question),
            (url + '，视频中推荐的型号有哪些？预算1000元，比较盒装与散片。',
             '视频中推荐的型号有哪些？预算1000元，比较盒装与散片。'),
            ('预算1000元，哪些型号值得购买？ ' + title + url, '预算1000元，哪些型号值得购买？'),
            (url + '【为什么降价？】', '【为什么降价？】'),
            (url + '\n请列出每款CPU的价格、平台和适用场景', '请列出每款CPU的价格、平台和适用场景'),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                urls = extract_urls(text)
                self.assertEqual(urls, [url])
                intent = understand(self.orch._llm, text, urls=urls)
                self.assertEqual(intent.source, url)
                self.assertEqual(intent.question, expected)
        self.assertEqual(explicit_followup('为什么降价？预算1000元', []), '为什么降价？预算1000元')

    def test_question_in_source_title_does_not_trigger_extra_answer(self):
        url = 'https://example.org/video'
        for text in ('【为什么CPU降价？】' + url, '帮我分析【为什么CPU降价？】' + url,
                     f'[为什么CPU降价？]({url})', f'[{url}]({url})', url,
                     '能否帮我分析这个视频？ ' + url, '《如何选CPU》' + url):
            with self.subTest(text=text):
                self.assertEqual(explicit_followup(text, extract_urls(text)), '')

    def test_local_video_question_is_kept_without_then_or_question_mark(self):
        self.orch._llm.decision = {'action': 'analyze', 'source_index': 0, 'question': ''}
        self.service.store.add_attachment(self.cid, 'u-local', 'CPU.mp4', self.root / 'CPU.mp4', 100)
        text = '帮我罗列视频中出现的cpu和推荐指数'
        self.assertEqual(self.wait(self.submit(text, attachment_id='u-local'))['status'], 'done')
        self.assertEqual(self.repo.list_qa('vfixture')[0]['question'], text)
        self.assertEqual((self.orch.calls, self.orch.ask_calls), (1, 1))

    def test_shared_video_automatic_question_idempotency_and_qa_only_retry(self):
        self.orch._llm.decision = {'action': 'analyze', 'source_index': 0, 'question': ''}
        text = '【【CPU推荐】历史最低价】https://example.org/cpu\n\n为什么这些cpu会出现历史最低点?'
        self.orch.fail_ask = True
        oid = self.submit(text, 'shared-cpu')
        self.assertEqual(self.wait(oid)['phase'], 'ask')
        self.assertEqual(self.service.store.operation(oid)['status'], 'failed')
        self.assertEqual(self.service.store.operation(oid)['payload']['question'], '为什么这些cpu会出现历史最低点?')
        self.assertEqual(self.submit(text, 'shared-cpu'), oid)
        self.orch.fail_ask = False
        self.service.retry(self.cid, oid)
        self.assertEqual(self.wait(oid)['status'], 'done')
        messages = self.service.store.messages(self.cid)
        self.assertEqual([m['kind'] for m in messages].count('report'), 1)
        self.assertEqual([m['kind'] for m in messages].count('answer'), 1)
        self.assertEqual(len([m for m in messages if m['role'] == 'user']), 1)
        self.assertEqual(self.orch.calls, 1)
        self.assertEqual(self.repo.list_qa('vfixture')[0]['question'], '为什么这些cpu会出现历史最低点?')

    def test_shared_question_survives_source_clarification_and_length_check(self):
        self.orch._llm.decision = {'action': 'analyze', 'source_index': 0, 'question': ''}
        text = '【CPU推荐】https://example.org/a https://example.org/b\n为什么这些cpu会出现历史最低点?'
        self.wait(self.submit(text))
        self.assertEqual(self.orch.calls, 0)
        self.wait(self.submit('第一个', 'choice'))
        self.assertEqual((self.orch.calls, self.orch.ask_calls), (1, 1))
        self.assertEqual(self.repo.list_qa('vfixture')[0]['question'], '为什么这些cpu会出现历史最低点?')
        with self.assertRaisesRegex(ValueError, '超过'):
            text = '【视频】https://example.org/a\n为什么' + '长' * 2000
            understand(self.orch._llm, text, urls=extract_urls(text))

    def test_report_then_automatic_question_and_followup(self):
        op = self.wait(self.submit('分析 https://example.org/a.mp4，然后解释依据'))
        self.assertEqual(op['status'], 'done', op)
        kinds = [m['kind'] for m in self.service.store.messages(self.cid)]
        self.assertEqual(kinds, ['text', 'report', 'answer'])
        self.assertEqual((self.orch.calls, self.orch.ask_calls), (1, 1))
        self.assertEqual(self.wait(self.submit('为什么？', 'request2'))['status'], 'done')
        self.assertEqual((self.orch.calls, self.orch.ask_calls), (1, 2))
        restored = self.client.get('/internal/conversations/' + self.cid).json()
        self.assertEqual(len(restored['messages']), 5)
        self.assertEqual(restored['conversation']['video_id'], 'vfixture')

    def test_video_only_does_not_ask_and_empty_input_guides(self):
        self.wait(self.submit('我想分析视频'))
        self.assertEqual(self.orch.calls, 0)
        self.assertEqual(self.service.store.messages(self.cid)[-1]['kind'], 'guidance')
        self.wait(self.submit('https://example.org/a.mp4', 'request2'))
        self.assertEqual((self.orch.calls, self.orch.ask_calls), (1, 0))

    def test_ambiguous_source_is_not_downloaded(self):
        self.wait(self.submit('https://example.org/a https://example.org/b，然后解释依据'))
        self.assertEqual(self.orch.calls, 0)
        self.assertEqual(self.service.store.messages(self.cid)[-1]['kind'], 'clarification')
        self.wait(self.submit('第一个', 'request2'))
        self.assertEqual(self.orch.calls, 1)
        self.assertEqual(self.orch.ask_calls, 1, '选择来源后仍保留原消息附带的问题')

    def test_busy_and_repeated_submission_are_idempotent(self):
        self.orch._llm.gate = threading.Event()
        first = self.submit('https://example.org/a')
        second = self.submit('https://example.org/a')
        self.assertEqual(first, second)
        response = self.client.post(f'/internal/conversations/{self.cid}/messages',
                                    json={'text': '另外的问题', 'request_id': 'different'})
        self.assertEqual(response.status_code, 409)
        self.orch._llm.gate.set()
        self.wait(first)
        self.assertEqual(self.orch.calls, 1)
        self.assertEqual(len(self.service.store.messages(self.cid)), 2)

    def test_failed_answer_retry_never_reanalyzes(self):
        self.orch.fail_ask = True
        oid = self.submit('https://example.org/a，解释依据')
        self.assertEqual(self.wait(oid)['phase'], 'ask')
        self.assertEqual(self.service.store.operation(oid)['status'], 'failed')
        self.assertIn('report', [m['kind'] for m in self.service.store.messages(self.cid)])
        self.orch.fail_ask = False
        self.service.retry(self.cid, oid)
        self.assertEqual(self.wait(oid)['status'], 'done')
        self.assertEqual(self.orch.calls, 1)
        self.assertEqual([m['kind'] for m in self.service.store.messages(self.cid)].count('report'), 1)

    def test_failed_analysis_never_asks(self):
        self.orch.fail_analysis = True
        op = self.wait(self.submit('https://example.org/a，解释依据'))
        self.assertEqual(op['status'], 'failed')
        self.assertEqual(self.orch.ask_calls, 0)

    def test_migration_from_v5_and_import_tombstone(self):
        path = self.root / 'legacy.db'
        conn = sqlite3.connect(path)
        for version in range(1, 6):
            for statement in _MIGRATIONS[version]:
                conn.execute(statement)
        conn.execute('PRAGMA user_version=5')
        conn.commit()
        conn.close()
        with SqliteRepository(path) as repo:
            self.assertEqual(repo.schema_version(), SCHEMA_VERSION)
            repo.add_video(video_id='vold', title='旧视频')
            repo.add_qa('vold', question='问题', answer='回答', session_id='old')
            store = ConversationStore(repo)
            store.import_history()
            store.import_history()
            self.assertEqual(len(store.list()), 1)
            cid = store.list()[0]['id']
            self.assertEqual(len(store.messages(cid)), 2)
            store.delete(cid)
            store.import_history()
            self.assertEqual(store.list(), [])
            self.assertIsNotNone(repo.find_video('vold'))

    def test_conversation_delete_removes_video_and_all_its_sessions(self):
        self.wait(self.submit('https://example.org/a，解释依据'))
        second = self.service.store.create(video_id='vfixture')
        self.repo.add_qa('vfixture', question='其他会话', answer='保留', session_id=second['session_id'])
        self.service.store.import_history()
        self.assertEqual(len(self.service.store.list()), 2)
        self.repo.add_video(video_id='unrelated',title='其他视频')
        response=self.client.delete('/internal/conversations/'+self.cid)
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(set(response.json()['conversation_ids']),{self.cid,second['id']})
        self.assertIsNone(self.repo.find_video('vfixture'))
        self.assertEqual(self.repo.list_qa('vfixture'),[])
        self.assertIsNotNone(self.repo.find_video('unrelated'))
        self.assertEqual(self.client.get('/internal/conversations/'+self.cid).status_code,404)
        self.assertEqual(self.client.get('/internal/conversations/'+second['id']).status_code,404)
        self.service.store.import_history()
        self.assertFalse(any(c['video_id']=='vfixture' for c in self.service.store.list()))

    def test_restart_marks_interrupted_without_dispatch(self):
        operation, _ = self.service.store.submit(self.cid, 'pending', 'https://example.org/a')
        self.service.close()
        self.service = AssistantService(self.orch, self.cfg)
        self.app.state.assistant = self.service
        self.assertEqual(self.service.store.operation(operation['id'])['status'], 'interrupted')
        self.assertEqual(self.orch.calls, 0)

    def test_upload_real_probe_limits_and_filename(self):
        video = self.root / 'test.mp4'
        subprocess.run([resolve_binary('ffmpeg'), '-y', '-f', 'lavfi', '-i', 'color=s=160x90:d=0.3',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)], capture_output=True, check=True)
        endpoint = f'/internal/conversations/{self.cid}/uploads'
        response = self.client.post(endpoint, content=video.read_bytes(), headers={'X-Filename': '..%2Ftest.mp4'})
        self.assertEqual(response.status_code, 201, response.text)
        attachment = response.json()
        self.assertEqual(attachment['filename'], 'test.mp4')
        stored = self.service.store.attachment(attachment['id'], self.cid)
        self.assertTrue(Path(stored['path']).is_file())
        self.assertNotEqual(Path(stored['path']).name, 'test.mp4')
        duplicate = self.client.post(endpoint, content=video.read_bytes(), headers={'X-Filename': 'second.mp4'}).json()
        self.assertNotEqual(attachment['id'], duplicate['id'])
        self.assertEqual(self.service.store.attachment(duplicate['id'], self.cid)['path'], stored['path'])
        self.assertEqual(self.client.post(endpoint, content=b'broken', headers={'X-Filename': 'test.mp4'}).status_code, 415)
        self.assertEqual(self.client.post(endpoint, content=b'broken', headers={'X-Filename': 'test.exe'}).status_code, 415)
        self.service.config = replace(self.cfg, limits=replace(self.cfg.limits, max_file_bytes=1))
        self.orch._cfg = self.service.config
        self.assertEqual(self.client.post(endpoint, content=b'xx', headers={'X-Filename': 'test.mp4'}).status_code, 413)
        other = self.service.store.create()['id']
        response = self.client.post(f'/internal/conversations/{other}/messages',
                                    json={'text': '', 'attachment_id': attachment['id'], 'request_id': 'foreign'})
        self.assertEqual(response.status_code, 404)

    def test_invalid_options_and_rename(self):
        endpoint = f'/internal/conversations/{self.cid}'
        self.assertEqual(self.client.patch(endpoint, json={'title': '新标题'}).status_code, 200)
        self.assertEqual(self.client.get(endpoint).json()['conversation']['title'], '新标题')
        self.assertEqual(self.client.post(endpoint + '/messages', json={
            'text': 'https://example.org/a', 'request_id': 'invalid', 'options': {'source': 'C:/bad'},
        }).status_code, 422)

    def test_idle_event_stream_finishes_and_does_not_block_restart(self):
        response = self.client.get(f'/internal/conversations/{self.cid}/events')
        self.assertEqual(response.status_code, 200)
        self.assertIn('"type": "snapshot"', response.text)
        self.assertEqual(self.service._listeners, {})

    def test_upload_disconnect_removes_partial_file(self):
        async def chunks():
            yield b'partial video'
            raise asyncio.CancelledError()
        request = SimpleNamespace(headers={'x-filename': 'test.mp4'}, stream=chunks)
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(receive_upload(request, self.service, self.cid))
        self.assertEqual(list((Path(self.cfg.path('data')) / 'uploads').glob('*.part')), [])

    def test_expired_unclaimed_upload_removed_but_video_retained(self):
        directory = Path(self.cfg.path('data')) / 'uploads'
        directory.mkdir(parents=True)
        orphan = directory / 'orphan.mp4'; orphan.write_bytes(b'old')
        keep = directory / 'keep.mp4'; keep.write_bytes(b'keep')
        self.repo.add_video(video_id='vkeep', local_path=str(keep))
        old = time.time() - 48 * 3600
        os.utime(orphan, (old, old)); os.utime(keep, (old, old))
        self.service.store.add_attachment(self.cid, 'old', 'orphan.mp4', orphan, 3)
        with self.repo._tx() as conn:
            conn.execute('UPDATE uploaded_videos SET created_ts=? WHERE id=?', (old, 'old'))
        cleanup_uploads(self.service)
        self.assertFalse(orphan.exists())
        self.assertTrue(keep.exists())

    def test_video_delete_cascades_history(self):
        self.wait(self.submit('https://example.org/a'))
        response=self.client.delete('/internal/videos/vfixture')
        self.assertEqual(response.status_code,200,response.text)
        self.assertIn(self.cid,response.json()['conversation_ids'])
        self.assertEqual(self.client.get('/internal/conversations/'+self.cid).status_code,404)
        response=self.client.post('/internal/conversations/'+self.cid+'/messages',json={'text':'内容是什么','request_id':'followup'})
        self.assertEqual(response.status_code,404)
        self.assertEqual(self.orch.ask_calls, 0)

    def test_linked_delete_refuses_busy_other_session_before_touching_vectors(self):
        self.wait(self.submit('https://example.org/a'))
        second=self.service.store.create(video_id='vfixture')
        op,_=self.service.store.submit(second['id'],'busy','正在问答')
        class Vectors:
            def delete_video(self,vid):raise AssertionError('忙碌时不得删除向量')
        self.orch._delete_vectors=Vectors()
        for endpoint in ('/internal/videos/vfixture','/internal/conversations/'+self.cid):
            response=self.client.delete(endpoint)
            self.assertEqual(response.status_code,409,response.text)
        self.assertIsNotNone(self.repo.find_video('vfixture'))
        self.assertEqual(len(self.service.store.list()),2)
        self.service.store.update_operation(op['id'],status='done')

    def test_linked_delete_purges_versions_upload_and_memory_but_keeps_external_original(self):
        from backend.pipeline.chat_sessions import get_chat_store,Turn
        self.wait(self.submit('https://example.org/a'))
        uploads=self.cfg.path('data')/'uploads';uploads.mkdir(exist_ok=True,parents=True)
        managed=uploads/'managed.mp4';managed.write_bytes(b'uploaded')
        original=self.root/'user-original.mp4';original.write_bytes(b'original')
        self.repo.update_video_meta('vfixture',local_path=str(original))
        self.service.store.add_attachment(self.cid,'managed','video.mp4',managed,8)
        directory=self.cfg.path('data')/'analysis-runs'/'r-delete-test';directory.mkdir(parents=True)
        (directory/'evidence.json').write_text('{}')
        with self.repo._tx() as conn:
            conn.execute('INSERT INTO analysis_runs VALUES (?,?,?,?,?,?,?,?,?,?)',('r-delete-test','vfixture','delete-test','fast-v3','ready','{}',str(directory),1,1,''))
        get_chat_store().remember('vfixture:'+self.cid,Turn('问题','回答'))
        response=self.client.delete('/internal/conversations/'+self.cid)
        self.assertEqual(response.status_code,200,response.text)
        self.assertFalse(managed.exists())
        self.assertFalse(directory.exists())
        self.assertTrue(original.exists())
        self.assertEqual(get_chat_store().history('vfixture:'+self.cid),[])
        self.assertEqual(self.repo._query('SELECT * FROM analysis_runs WHERE video_id=?',('vfixture',)),[])

    def test_shared_uploaded_source_survives_until_last_video_deleted(self):
        self.wait(self.submit('https://example.org/a'))
        directory=self.cfg.path('data')/'uploads';directory.mkdir(parents=True,exist_ok=True)
        shared=directory/'shared.mp4';shared.write_bytes(b'shared')
        self.repo.update_video_meta('vfixture',local_path=str(shared))
        self.repo.add_video(video_id='vsecond',local_path=str(shared))
        second=self.service.store.create(video_id='vsecond')
        self.assertEqual(self.client.delete('/internal/conversations/'+self.cid).status_code,200)
        self.assertTrue(shared.exists())
        self.assertEqual(self.client.delete('/internal/conversations/'+second['id']).status_code,200)
        self.assertFalse(shared.exists())

    def test_draft_delete_only_removes_own_upload(self):
        directory=self.cfg.path('data')/'uploads';directory.mkdir(parents=True,exist_ok=True)
        upload=directory/'draft.mp4';upload.write_bytes(b'draft')
        self.service.store.add_attachment(self.cid,'draft','draft.mp4',upload,5)
        self.repo.add_video(video_id='keep',title='保留')
        response=self.client.delete('/internal/conversations/'+self.cid)
        self.assertEqual(response.status_code,200,response.text)
        self.assertFalse(upload.exists())
        self.assertIsNotNone(self.repo.find_video('keep'))

    def test_delete_refuses_analysis_lock_and_surfaces_vector_failure(self):
        from backend.pipeline.domain import _video_lock
        self.wait(self.submit('https://example.org/a'))
        with _video_lock('vfixture'):
            self.assertEqual(self.client.delete('/internal/videos/vfixture').status_code,409)
        class Vectors:
            def delete_video(self,vid):raise RuntimeError('模拟向量删除失败')
        self.orch._delete_vectors=Vectors()
        response=self.client.delete('/internal/videos/vfixture')
        self.assertEqual(response.status_code,500,response.text)
        self.assertIsNotNone(self.repo.find_video('vfixture'))
        self.assertEqual(len(self.service.store.list()),1)

    def test_stale_browser_confirmation_cannot_trigger_new_cascade(self):
        self.wait(self.submit('https://example.org/a'))
        for endpoint in ('/internal/videos/vfixture','/internal/conversations/'+self.cid):
            response=self.client.delete(endpoint,headers={'Sec-Fetch-Mode':'cors'})
            self.assertEqual(response.status_code,409)
            self.assertIn('刷新页面',response.json()['detail'])
        self.assertIsNotNone(self.repo.find_video('vfixture'))
        response=self.client.delete('/internal/conversations/'+self.cid,headers={
            'Sec-Fetch-Mode':'cors','X-VideoLens-Linked-Delete':'1'})
        self.assertEqual(response.status_code,200,response.text)

    def test_submit_cannot_enter_while_linked_delete_is_in_progress(self):
        from concurrent.futures import ThreadPoolExecutor
        self.wait(self.submit('https://example.org/a'))
        started,release=threading.Event(),threading.Event()
        class Vectors:
            def delete_video(self,vid):started.set();release.wait(5);return {}
        self.orch._delete_vectors=Vectors()
        with ThreadPoolExecutor(max_workers=2) as pool:
            deleting=pool.submit(self.orch.delete,'vfixture')
            self.assertTrue(started.wait(3))
            submitting=pool.submit(self.service.store.submit,self.cid,'race','并发追问')
            try:
                time.sleep(.05)
                self.assertFalse(submitting.done())
            finally:release.set()
            self.assertNotIn('db_error',deleting.result(timeout=5))
            with self.assertRaises(KeyError):submitting.result(timeout=5)

    def test_video_delete_cleans_owned_upload_only(self):
        directory = Path(self.cfg.path('data')) / 'uploads'
        directory.mkdir(parents=True)
        managed = directory / 'managed.mp4'; managed.write_bytes(b'copy')
        original = self.root / 'user-original.mp4'; original.write_bytes(b'original')
        owner = SimpleNamespace(_cfg=self.cfg)
        Orchestrator.purge_files(owner, 'vfixture', local_path=str(managed))
        Orchestrator.purge_files(owner, 'vfixture', local_path=str(original))
        self.assertFalse(managed.exists())
        self.assertTrue(original.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
