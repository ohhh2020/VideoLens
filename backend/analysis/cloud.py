from backend.analysis.prompts import get_prompt
"""任务预算、请求缓存、统一公平调度与可见的截断状态。"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import random
import threading
import time
from types import SimpleNamespace


class CloudScheduler:
    def __init__(self):
        self.condition = threading.Condition()
        self.pending = []
        self.sequence = 0
        self.total = 0
        self.active = {'text': 0, 'vision': 0}

    @contextmanager
    def slot(self, policy, channel='text', interactive=False):
        with self.condition:
            self.sequence += 1
            ticket = [time.monotonic(), self.sequence, interactive, channel]
            self.pending.append(ticket)
            while True:
                now = time.monotonic()
                # 后台等待超过5秒后按到达顺序竞争，避免持续问答饿死分析。
                ordered = sorted(self.pending, key=lambda item: (0 if item[2] or now - item[0] > 5 else 1, item[1]))
                eligible = next((item for item in ordered if self.active[item[3]] < getattr(policy, 'cloud_' + item[3])), None)
                if eligible is ticket and self.total < policy.cloud_total:
                    self.pending.remove(ticket)
                    self.total += 1
                    self.active[channel] += 1
                    break
                self.condition.wait(.2)
        try:
            yield
        finally:
            with self.condition:
                self.total -= 1
                self.active[channel] -= 1
                self.condition.notify_all()


SCHEDULER = CloudScheduler()
MANAGED = ContextVar('cloud_managed', default=False)


def http_error(error, original):
    """保留限流状态和 Retry-After（秒或 HTTP 日期），供唯一重试层处理。"""
    from email.utils import parsedate_to_datetime
    error.code = original.code
    value = original.headers.get('Retry-After') if original.headers else None
    try:
        error.retry_after = max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            error.retry_after = max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            error.retry_after = None
    return error


def parse_json(text):
    text = str(text).strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[-1].rsplit('```', 1)[0]
    return json.loads(text)


def retry_delay(exc, attempt):
    status = getattr(exc, 'code', None)
    retryable = status in {429, 500, 502, 503, 504} or isinstance(exc, (TimeoutError, ConnectionError)) or any(
        term in str(exc) for term in ('超时', 'timeout', '网络错误'))
    if not retryable:
        return None
    value = getattr(exc, 'retry_after', None)
    return max(0, float(value)) if value is not None else min(30, 2**attempt) + random.random() * .25


def request(ledger, policy, task, identity, fn, *, channel='text', interactive=False):
    key = hashlib.sha256(json.dumps({'version': 2, 'task': task, 'identity': identity},
                                   sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
    cached = ledger.cached(key) if ledger else None
    if cached:
        return SimpleNamespace(**cached)
    for attempt in range(policy.cloud_retries + 1):
        try:
            with SCHEDULER.slot(policy, channel, interactive):
                token = MANAGED.set(True)
                started = time.monotonic()
                try:
                    response = fn()
                finally:
                    MANAGED.reset(token)
            break
        except Exception as exc:
            if ledger:
                ledger.call(task, 'failed', {'code': getattr(exc, 'code', None), 'type': type(exc).__name__})
            status = getattr(exc, 'code', None)
            message = str(exc)
            retryable = status in {429, 500, 502, 503, 504} or isinstance(exc, (TimeoutError, ConnectionError)) or any(
                term in message for term in ('429', '503', '超时', 'timeout', '网络错误'))
            if attempt == policy.cloud_retries or not retryable:
                raise
            retry_after = getattr(exc, 'retry_after', None)
            time.sleep(max(0, float(retry_after)) if retry_after is not None else min(30, 2**attempt) + random.random() * .25)
    text = str(getattr(response, 'text', response))
    data = {'text': text, 'finish_reason': str(getattr(response, 'finish_reason', '') or ''),
            'cache_key': key,
            'prompt_tokens': int(getattr(response, 'prompt_tokens', 0)),
            'completion_tokens': int(getattr(response, 'completion_tokens', 0)),
            'usage_reported': bool(getattr(response, 'prompt_tokens', 0) or getattr(response, 'completion_tokens', 0))}
    data.update({name: getattr(response, name, default) for name, default in
                 [('backend', 'cloud'), ('model', ''), ('degraded', False), ('attempts', [])]})
    data['elapsed_ms'] = int((time.monotonic() - started)*1000)
    for name in ('actual_start_ms', 'actual_end_ms', 'encoded_duration_ms'):
        if hasattr(response, name):
            data[name] = getattr(response, name)
    if not text.strip():
        raise ValueError('云端返回空结果')
    if ledger:
        ledger.call(task, 'success', data)
        ledger.cache(key, task, data)
    return SimpleNamespace(**data)


def model_identity(llm):
    return [{'model': getattr(backend, 'model', ''), 'endpoint': getattr(backend, 'endpoint', ''),
             'credential_digest': hashlib.sha256(getattr(backend, 'api_key', '').encode()).hexdigest()}
            for backend in getattr(llm, '_backends', [])] or getattr(llm, 'model', '')


def complete(llm, messages, policy, ledger=None, *, task, output_tokens, interactive=False):
    identity = {'messages': messages, 'model': model_identity(llm),
                'output_tokens': output_tokens, 'temperature': 0.0, 'json_mode': task in {'章节证据摘要', '问答证据充分性'}}
    return request(ledger, policy, task, identity,
                   lambda: llm.complete(messages, max_tokens=output_tokens, temperature=0.0, **({'json_mode': True} if identity['json_mode'] else {})), interactive=interactive)


def continuation_messages(messages, text, budget, output_tokens, ledger=None):
    from backend.analysis.policy import estimate_tokens
    instruction = {'role': 'user', 'content': get_prompt('integrity_continue', ledger)}
    available = budget - output_tokens - 512 - estimate_tokens(json.dumps([*messages, instruction], ensure_ascii=False)) - 128
    if available < 64:
        raise ValueError('续写总预算不足；未静默截掉原始证据')
    tail = text.encode('utf-8')[-min(available // 2, 2048):].decode('utf-8', errors='ignore')
    return [*messages, {'role': 'assistant', 'content': tail}, instruction]


def prose(llm, messages, policy, ledger=None, *, task, output_tokens, interactive=False, input_budget=None):
    input_budget = input_budget or policy.report_input_tokens
    result = complete(llm, messages, policy, ledger, task=task, output_tokens=output_tokens, interactive=interactive)
    text, count = result.text, 0
    while result.finish_reason == 'length' and count < policy.max_continuations:
        count += 1
        continuation = continuation_messages(messages, text, input_budget, output_tokens, ledger)
        result = complete(llm, continuation, policy, ledger, task=task + f':续{count}',
                          output_tokens=output_tokens, interactive=interactive)
        overlap = 0
        for length in range(min(512, len(text), len(result.text)), 0, -1):
            if text.endswith(result.text[:length]):
                overlap = length
                break
        text += ('\n\n' if not overlap else '') + result.text[overlap:]
    return text, result.finish_reason == 'length'
