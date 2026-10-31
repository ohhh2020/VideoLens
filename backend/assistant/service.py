"""持久化任务编排；浏览器仅订阅，分析后自动问答不依赖页面存活。"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from backend.assistant.titles import generate_title
from backend.assistant.intent import extract_urls, explicit_followup, understand
from backend.pipeline.domain import AnalyzeOptions
from backend.pipeline.factory import _build_llm
from backend.storage.conversations import ConversationStore

logger = logging.getLogger('zhiying.assistant')


class DefaultAnalysisStrategy:
    """后续分类策略的接入点；当前只复用原视频分析流水线。"""
    def analyze(self, orchestrator, source, options):
        return orchestrator.analyze(source, options)


class AssistantService:
    def __init__(self, orchestrator, config, *, strategy=None):
        self.orch, self.config = orchestrator, config
        self.store = ConversationStore(orchestrator.repository)
        self.strategy = strategy or DefaultAnalysisStrategy()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='video-assistant')
        self._listeners = {}
        self._lock = threading.Lock()
        self._closed = False
        self.store.interrupt_pending()
        self.store.import_history()

    def close(self):
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=True)

    def snapshot(self, cid):
        return {'conversation': self.store.get(cid), 'messages': self.store.messages(cid),
                'operation': self.store.active(cid) or self.store.latest(cid)}

    def subscribe(self, cid, loop, queue):
        self.store.get(cid)
        with self._lock:
            self._listeners.setdefault(cid, []).append((loop, queue))

    def unsubscribe(self, cid, queue):
        with self._lock:
            self._listeners[cid] = [(lp, q) for lp, q in self._listeners.get(cid, []) if q is not queue]
            if not self._listeners[cid]:
                self._listeners.pop(cid, None)

    def publish(self, cid, event):
        with self._lock:
            listeners = list(self._listeners.get(cid, []))
        for loop, queue in listeners:
            def put(q=queue, value=event):
                if q.full():
                    q.get_nowait()
                q.put_nowait(value)
            if not loop.is_closed():
                loop.call_soon_threadsafe(put)

    def _snapshot_event(self, cid):
        try:
            self.publish(cid, {'type': 'snapshot', **self.snapshot(cid)})
        except KeyError:
            pass  # 终态落库后，用户可能已经删除该会话。

    def submit(self, cid, request_id, text, attachment_id='', options=None):
        self.store.get(cid)
        self.orch._require_cloud_ready()
        with self._lock:
            if self._closed:
                raise RuntimeError('服务正在关闭')
            from backend.analysis.runs import snapshot
            options = dict(options or {})
            options["policy_snapshot"] = snapshot(getattr(self.orch, "_cfg", self.config))
            operation, created = self.store.submit(cid, request_id, text, attachment_id, options)
            if created:
                self._pool.submit(self._run, operation['id'])
        self._snapshot_event(cid)
        return operation

    def retry(self, cid, oid):
        self.orch._require_cloud_ready()
        with self._lock:
            if self._closed:
                raise RuntimeError('服务正在关闭')
            operation = self.store.retry(cid, oid)
            self._pool.submit(self._run, oid)
        self._snapshot_event(cid)
        return operation

    def delete_conversation(self, cid):
        """已绑定视频时联动删除该视频全部会话；未绑定草稿仅清理自身。"""
        from backend.pipeline.deletion import attachment_paths, purge_uploads
        from backend.pipeline.domain import PipelineError
        item = self.store.get(cid)
        if item['video_id']:
            removed = self.orch.delete(item['video_id'], missing_ok=True)
            failures = [str(removed[key]) for key in ('db_error','vectors_error') if removed.get(key)]
            if failures:
                raise RuntimeError('联动删除未全部完成：' + '；'.join(failures))
            return {'deleted': True, 'video_id': item['video_id'], 'conversation_ids': removed.get('conversation_ids', [cid])}
        paths = attachment_paths(self.store.repo, conversation_id=cid)
        try:
            self.store.delete(cid)
        except ValueError as exc:
            raise PipelineError(str(exc)) from exc
        purge_uploads(self.store.repo, self.config, paths)
        return {'deleted': True, 'video_id': '', 'conversation_ids': [cid]}

    def _progress(self, oid, cid, phase, ratio, message):
        self.store.update_operation(oid, phase=phase, progress=ratio, message=message)
        self.publish(cid, {'type': 'progress', 'operation_id': oid, 'phase': phase,
                           'progress': ratio, 'message': message})

    def ensure_title(self, cid, report):
        """首次识别内容后命名；命名失败不使已经完成的分析或问答失败。"""
        if self.store.get(cid)['title_source'] != 'pending':
            return
        summary = str(report.get('summary') or '').strip()
        if not summary:
            # 跳过总结时，仍可根据已提取的切片证据命名。
            parts = []
            length = 0
            for clip in self.orch.repository.iter_clip_payloads(report.get('video_id', '')):
                text = str(clip.get('text') or clip.get('summary') or '').strip()
                if text:
                    parts.append(text)
                    length += len(text)
                    if length >= 8000:
                        break
            summary = '\n'.join(parts)[:8000]
        if not summary:
            return
        try:
            llm = self.orch._llm or _build_llm(self.config, None)
            title = generate_title(llm, summary)
            if self.store.set_generated_title(cid, title):
                self._snapshot_event(cid)
        except Exception as exc:
            # 后续操作可再尝试命名，成功前保留待命名状态，不冒充模型生成。
            logger.warning('会话标题生成失败，已保留视频分析结果 cid=%s error=%s',
                           cid, type(exc).__name__)

    def _run(self, oid):
        operation = self.store.operation(oid)
        cid, payload = operation['conversation_id'], operation['payload']
        phase = operation['phase']
        self.store.update_operation(oid, status='running', error='')
        try:
            conversation = self.store.get(cid)
            if phase == 'intent':
                self._progress(oid, cid, 'intent', 0, '正在理解视频请求…')
                attachment = self.store.attachment(payload['attachment_id'], cid) if payload.get('attachment_id') else None
                urls = extract_urls(payload['text'])
                history = self.store.messages(cid)[:-1][-8:]
                # 用户明确回复候选编号时，从先前澄清消息恢复候选，仍不允许模型构造来源。
                prior = next((m for m in reversed(history) if m['kind'] == 'clarification'), None)
                pending_question = prior['payload'].get('question', '') if prior and not conversation['video_id'] else ''
                if not urls and not attachment and prior and not conversation['video_id']:
                    urls = prior['payload'].get('urls', [])
                    aid = prior['payload'].get('attachment_id')
                    attachment = self.store.attachment(aid, cid) if aid else None
                if not conversation['video_id'] and len(urls) + bool(attachment) > 1 and (extract_urls(payload['text']) or payload.get('attachment_id')):
                    self.store.add_message(cid, 'clarification', '检测到多个视频来源，请明确选择一个（例如“第一个”“第二个”或“附件”）。',
                                           payload={'urls': urls, 'attachment_id': attachment['id'] if attachment else '',
                                                    'question': explicit_followup(payload['text'], urls)}, operation_id=oid)
                    self.store.update_operation(oid, status='done', message='等待选择视频', progress=1)
                    return
                llm = self.orch._llm or _build_llm(self.config, None)
                intent = understand(llm, payload['text'], urls=urls, attachment=attachment,
                                    video_id=conversation['video_id'],
                                    history=[{'role': m['role'], 'text': m['text']} for m in history],
                                    max_question_chars=self.config.limits.max_question_chars)
                if conversation['video_id'] and attachment:
                    self.store.add_message(cid, 'guidance', '每个会话只绑定一个视频，请点击“新建会话”上传另一视频。',
                                           payload={'new_conversation': True}, operation_id=oid)
                    self.store.update_operation(oid, status='done', progress=1, message='请新建会话')
                    return
                if intent.action in {'request_video', 'clarify'}:
                    self.store.add_message(cid, 'clarification' if sources_present(urls, attachment) else 'guidance', intent.reply,
                                           payload={'urls': urls, 'attachment_id': attachment['id'] if attachment else '',
                                                    'question': explicit_followup(payload['text'], urls) or pending_question,
                                                    'new_conversation': bool(conversation['video_id'])}, operation_id=oid)
                    self.store.update_operation(oid, status='done', message='等待补充信息', progress=1)
                    return
                payload['question'] = intent.question or pending_question
                payload['source'] = intent.source
                if intent.source and attachment and intent.source == attachment['path']:
                    payload['selected_attachment_id'] = attachment['id']
                    if not payload['options'].get('title'):
                        payload['options']['title'] = attachment['filename'][:200]
                phase = 'analysis' if intent.action == 'analyze' else 'ask'
                self.store.update_operation(oid, phase=phase, payload=payload)
            if phase == 'analysis':
                options = AnalyzeOptions(**payload.get('options', {}))
                options.task_id = oid
                options.progress = lambda stage, ratio, message: self._progress(oid, cid, 'analysis', ratio, message)
                report = self.strategy.analyze(self.orch, payload['source'], options)
                if not report.ok:
                    raise RuntimeError('；'.join(report.errors) or '视频分析失败')
                self.store.update(cid, video_id=report.video_id)
                if payload.get('selected_attachment_id'):
                    with self.store.repo._tx() as conn:
                        conn.execute('UPDATE uploaded_videos SET claimed=1 WHERE id=?', (payload['selected_attachment_id'],))
                self.store.add_message(cid, 'report', report.summary, payload=report.to_dict(), operation_id=oid,
                                       import_key=oid + ':report')
                # 先写报告与阶段再发事件，刷新或断流不会重新执行已完成的分析。
                phase = 'ask' if payload.get('question') else 'complete'
                self.store.update_operation(oid, phase=phase, payload=payload)
                self._snapshot_event(cid)
            if phase in {'ask', 'complete'}:
                conversation = self.store.get(cid)
                if conversation['title_source'] == 'pending' and self.orch.repository.find_video(conversation['video_id']):
                    saved_report = self.orch.repository.get_video_report(conversation['video_id'])
                    if saved_report:
                        self.ensure_title(cid, saved_report)
            if phase == 'ask':
                conversation = self.store.get(cid)
                if not self.orch.repository.find_video(conversation['video_id']):
                    raise RuntimeError('视频已从视频库删除，历史报告仍可查看；请新建会话重新提供视频')
                question = payload['question']
                self._progress(oid, cid, 'ask', 0, '正在检索视频证据并回答…')
                # 若崩溃发生在问答落库后、会话消息落库前，直接接回已有答案。
                existing = [row for row in self.orch.repository.list_qa(conversation['video_id'], session_id=conversation['session_id'], newest=True, limit=1)
                            if row['question'] == question and row['created_ts'] >= operation['created_ts']]
                result_data = None
                if existing:
                    row = existing[-1]
                    result_data = {'answer': row['answer'], 'refs': row['refs'], 'elapsed_ms': row['elapsed_ms']}
                else:
                    for kind, data in self.orch.ask_stream(conversation['video_id'], question, session_id=conversation['session_id']):
                        if kind == 'delta':
                            self.publish(cid, {'type': 'delta', 'operation_id': oid, 'text': data})
                        elif kind == 'final':
                            result_data = data.to_dict()
                if result_data is None:
                    raise RuntimeError('视频问答未返回完整答案，请重试')
                self.store.add_message(cid, 'answer', result_data.get('answer', ''), payload=result_data, operation_id=oid,
                                       import_key=oid + ':answer')
            self.store.update_operation(oid, status='done', phase='complete', message='已完成', progress=1)
        except Exception as exc:
            logger.exception('视频助手操作失败 operation=%s', oid)
            self.store.update_operation(oid, status='failed', error=str(exc), message='处理失败，可重试')
            self.store.add_message(cid, 'error', str(exc), operation_id=oid)
        finally:
            self._snapshot_event(cid)


def sources_present(urls, attachment):
    return bool(urls or attachment)
