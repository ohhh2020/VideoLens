"""动态问答：局部检索、全片章节遍历、有限补检索与版本内补充取证。"""
from __future__ import annotations
from backend.analysis.prompts import get_prompt

import json
from pathlib import Path
import re
import time
from types import SimpleNamespace

from backend.analysis.cloud import complete, parse_json
from backend.analysis.evidence import compose, split_text
from backend.analysis.policy import VERSION, estimate_tokens
from backend.analysis.runs import RunLedger
from backend.pipeline.domain import AskResult, extract_citations
from backend.pipeline.helpers import _enforce_answer_claims


def scope_of(question):
    return 'global' if re.search(r'全部|所有|全片|整[个部段]|总结|汇总|比较|对比|异同|步骤|流程|梳理|有哪些|分别|总体|整体|主要内容|核心内容|讲了什么|主要讲|内容概览|中心思想|核心观点', question, re.I) else 'local'


def active_ledger(orch, video_id):
    report = orch._repo.get_video_report(video_id) or {}
    profile = report.get('profile') or {}
    if profile.get('analysis_version') not in {VERSION, 'integrity-v2'}:
        return None
    row = orch._repo._query_one('SELECT * FROM analysis_runs WHERE run_id=? AND status=?',
                                (profile.get('analysis_run_id'), 'ready'))
    if not row or not Path(row['work_dir'], 'evidence.sqlite3').is_file():
        return None
    return RunLedger(row['work_dir'], row['run_id'])


def item_id(item):
    return item.get('clip_id') if isinstance(item, dict) else getattr(item, 'clip_id', '')


def evidence_items(orch, vid, ids, ledger):
    if ledger:
        rows = ledger.all_evidence()
        return [row for row in rows if row['clip_id'] in ids]
    clips = [p for p in orch._repo.iter_clip_payloads(vid) if p['clip_id'] in ids]
    attached = orch._attach_visuals(vid, clips)
    return [{**p, 'text': compose(p), 'kind': 'legacy'} for p in attached]


def neighbours(orch, vid, ids):
    clips = sorted(orch._repo.iter_clip_payloads(vid), key=lambda p: p['start_ms'])
    positions = {p['clip_id']: n for n, p in enumerate(clips)}
    selected = set(ids)
    for cid in ids:
        pos = positions.get(cid)
        if pos is not None:
            selected.update(clips[n]['clip_id'] for n in (pos - 1, pos + 1) if 0 <= n < len(clips))
    return selected


def messages_for(question, title, rows, prefix, ledger=None):
    return [*prefix, {'role': 'system', 'content':
        get_prompt('integrity_qa', ledger)},
        {'role': 'user', 'content': json.dumps({'视频': title, '原始问题': question, '本轮证据': rows}, ensure_ascii=False)}]


def chapter_facts(ledger, video_id):
    """遍历全部叶摘要，按切片归并逐字相同事实；来源仍可逐条追溯。"""
    from backend.analysis.evidence import make_evidence
    groups = {}
    for node in ledger.get('summary_tree') or []:
        if node['level'] != 0:
            continue
        key = tuple(node['clip_ids'])
        group = groups.setdefault(key, {'facts': {}, 'start_ms': node['start_ms'], 'end_ms': node['end_ms']})
        group['start_ms'] = min(group['start_ms'], node['start_ms'])
        group['end_ms'] = max(group['end_ms'], node['end_ms'])
        for fact in node['facts'] or [node['summary']]:
            sources = group['facts'].setdefault(fact, set())
            sources.update(node['evidence_ids'])
    rows, provenance = [], []
    for clip_ids, group in groups.items():
        text = '\n'.join(group['facts'])
        evidence = make_evidence(ledger.run_id, video_id, clip_ids[0], 'chapter_fact_index',
                                group['start_ms'], group['end_ms'], text,
                                source_ids=sorted({s for sources in group['facts'].values() for s in sources}),
                                observation='derived_fact_index')
        ledger.evidence(evidence)
        rows.append({'evidence_id': evidence.evidence_id, 'clip_id': clip_ids[0], 'video_id': video_id,
                     'start_ms': group['start_ms'], 'end_ms': group['end_ms'], 'text': text, 'kind': 'chapter'})
        provenance.append({'evidence_id': evidence.evidence_id, 'clip_ids': list(clip_ids),
                           'facts': [{'text': fact, 'source_ids': sorted(sources)} for fact, sources in group['facts'].items()]})
    ledger.put('qa_fact_provenance', provenance)
    return rows


def fit_messages(llm, question, title, rows, prefix, policy, ledger, budget, output):
    """超过预算时逐批阅读全部输入，不能靠切片/字符上限隐式丢尾。"""
    safety = 1536
    for level in range(12):
        messages = messages_for(question, title, rows, prefix, ledger)
        if estimate_tokens(json.dumps(messages, ensure_ascii=False)) + output + safety <= budget:
            return messages
        available = budget - output - safety - estimate_tokens(json.dumps(messages_for(question, title, [], prefix, ledger), ensure_ascii=False)) - estimate_tokens(get_prompt('integrity_qa_map', ledger)) - 256
        if available < 256:
            if prefix:
                prefix = prefix[2:]
                continue
            raise ValueError('问答输入预算不足以容纳问题、提示词和输出预留')
        units = []
        for row in rows:
            for part in split_text(row['text'], max_bytes=max(128, available // 2)):
                units.append({**row, 'text': part})
        groups, batch, used = [], [], 0
        for row in units:
            cost = estimate_tokens(json.dumps(row, ensure_ascii=False))
            if batch and used + cost > available:
                groups.append(batch)
                batch, used = [], 0
            batch.append(row)
            used += cost
        if batch:
            groups.append(batch)
        reduced = []
        for group in groups:
            # 应用估算使用UTF-8字节，服务商max_tokens却是真实token。
            # 旧实现只设token上限，短证据可被扩写成更长的“答案”，反复压缩仍不收敛。
            # 每批同时约束正文字符数与实际字节，并校验完成状态；不续写扩张压缩结果。
            source_bytes = estimate_tokens(json.dumps(group, ensure_ascii=False))
            target_bytes = max(96, min(source_bytes // 3, available // 2))
            for attempt in range(2):
                target_chars = max(16, target_bytes // (3 * (attempt + 1)))
                prompt = messages_for(question, title, group, prefix, ledger)
                prompt[-2]['content'] += '\n' + get_prompt('integrity_qa_map', ledger) + (
                    f'\n这是证据提炼，不是最终回答。正文最多{target_chars}个字符（含数字引用）。'
                    '只写与原始问题直接相关的简短事实及其原编号引用；保留数字、单位、否定和必要步骤。'
                    '不要标题、开场白、重复问题或推测原因；无直接相关事实时简短说明。')
                response = complete(llm, prompt, policy, ledger, task='问答证据提炼-v2',
                    output_tokens=min(output, max(64, target_chars * 2)), interactive=True)
                body = response.text.strip()
                if response.finish_reason != 'length' and estimate_tokens(body) <= target_bytes:
                    break
            else:
                raise ValueError('问答证据提炼超出压缩预算或被截断，请重试；原始证据未被裁切')
            reduced.append({'index': '', 'text': body})
        if estimate_tokens(json.dumps(reduced, ensure_ascii=False)) >= estimate_tokens(json.dumps(rows, ensure_ascii=False)):
            raise ValueError('问答证据压缩未收敛，请提高预算；未静默丢弃证据')
        rows = reduced
    raise ValueError('问答层级超过限制')


def stream_answer(llm, messages, policy, ledger, output, budget):
    """保留流式结束元数据；已输出后断流不自动从头生成。"""
    from backend.analysis.cloud import continuation_messages
    original = messages
    collected, usage, final = '', {'prompt_tokens': 0, 'completion_tokens': 0}, ''
    for continuation in range(policy.max_continuations + 1):
        events = getattr(llm, 'stream_events', None)
        if callable(events):
            metadata = None
            piece_text = ''
            # 只在首字之前重试临时故障，首字之后断流明确失败。
            for attempt in range(policy.cloud_retries + 1):
                try:
                    for kind, data in events(messages, max_tokens=output, temperature=0.0):
                        if kind == 'delta':
                            piece_text += data
                            if continuation == 0:
                                yield 'delta', data
                        elif kind == 'metadata':
                            metadata = data
                    break
                except Exception as exc:
                    if ledger:
                        ledger.call('流式问答', 'failed', {'code': getattr(exc, 'code', None), 'type': type(exc).__name__})
                    from backend.analysis.cloud import retry_delay
                    if piece_text or attempt == policy.cloud_retries:
                        raise
                    delay = retry_delay(exc, attempt)
                    if delay is None:
                        raise
                    time.sleep(delay)
            if ledger and metadata:
                ledger.call('流式问答', 'success', {**metadata, 'usage_reported': bool(metadata.get('prompt_tokens') or metadata.get('completion_tokens'))})
            if not metadata or not piece_text.strip():
                raise ValueError('云端回答缺少完成元数据或正文')
        else:
            value = complete(llm, messages, policy, ledger, task='视频问答', output_tokens=output, interactive=True)
            piece_text, metadata = value.text, vars(value)
            if continuation == 0:
                yield 'delta', piece_text
        for key in usage:
            usage[key] += metadata.get(key, 0)
        final = metadata.get('finish_reason', '')
        overlap = 0
        for size in range(min(len(collected), len(piece_text), 512), 0, -1):
            if collected.endswith(piece_text[:size]):
                overlap = size
                break
        collected += ('\n\n' if continuation and not overlap else '') + piece_text[overlap:]
        if continuation:
            yield 'delta', '\n\n' + piece_text[overlap:]
        if final != 'length':
            break
        if final == 'length' and continuation < policy.max_continuations:
            messages = continuation_messages(original, collected, budget, output, ledger)
    yield 'completed', {'answer': collected, 'truncated': final == 'length', 'finish_reason': final, **usage}


def ask(orch, video_id, question, *, top_k=None, max_context_tokens=None, stream=False, session_id=''):
    started = time.monotonic()
    text = (question or '').strip()
    result = AskResult(question=question, video_id=video_id, session_id=session_id)
    if not text or len(text) > orch._cfg.limits.max_question_chars:
        result.notes.append('问题不能为空且最多2000字符')
        yield 'final', result
        return
    video = orch._repo.find_video(video_id)
    if not video:
        result.notes.append('视频不存在')
        yield 'final', result
        return
    policy, llm = orch._cfg.integrity, orch._llm
    ledger = active_ledger(orch, video_id)
    global_scope = scope_of(text) == 'global'
    k = top_k or (policy.qa_global_k if global_scope else policy.qa_local_k)
    history = []
    if orch._cfg.qa.multi_turn:
        for turn in orch._repo.list_qa(video_id, session_id=session_id, limit=200)[-orch._cfg.qa.history_window:]:
            history.extend([{'role': 'user', 'content': turn['question']}, {'role': 'assistant', 'content': turn['answer']}])
    rewritten = orch._rewrite_query(text, history) if history and orch._cfg.qa.query_rewrite else ''
    query = text + ('\n补充检索主题：' + rewritten if rewritten else '')
    yield 'status', {'phase': 'retrieval', 'message': '正在遍历章节与事实索引…' if global_scope else '正在检索并补充相邻证据…'}
    retrieved = orch._retrieve(video_id, query, k)
    ids = {item_id(item) for item in retrieved}
    rows = []
    if global_scope:
        if ledger and ledger.get('summary_tree'):
            rows = chapter_facts(ledger, video_id)
        else:
            ids = {p['clip_id'] for p in orch._repo.iter_clip_payloads(video_id)}
    if not rows:
        rows = evidence_items(orch, video_id, neighbours(orch, video_id, ids), ledger)
    if not global_scope:
        for round_no in range(policy.qa_extra_rounds):
            overview = [{'id': row.get('evidence_id', row['clip_id']), 'text': row['text'][:400]} for row in rows]
            decision = complete(llm, [{'role': 'user', 'content': json.dumps({'任务': get_prompt('integrity_sufficiency', ledger),
                    '原始问题': text, '证据概览': overview}, ensure_ascii=False)}], policy, ledger,
                    task='问答证据充分性', output_tokens=256, interactive=True)
            data = parse_json(decision.text)
            if decision.finish_reason == 'length' or not isinstance(data.get('sufficient'), bool):
                raise ValueError('证据充分性判断格式无效，请重试')
            if data['sufficient']:
                break
            extra = str(data.get('query') or '')[:500]
            fresh = orch._retrieve(video_id, text + '\n补充检索主题：' + extra, k)
            ids.update(item_id(item) for item in fresh)
            rows = evidence_items(orch, video_id, neighbours(orch, video_id, ids), ledger)
            result.notes.append(f'已完成第{round_no + 1}轮补充检索')
            if round_no + 1 == policy.qa_extra_rounds and ledger and rows:
                from backend.analysis.supplement import supplement
                rows.extend(supplement(orch, video, rows, ledger))
    # 每个编号只指向本轮实际使用的原始证据或章节节点。
    refs, numbered = [], []
    for row in rows:
        if not str(row.get('text', '')).strip():
            continue
        n = len(refs) + 1
        lo, hi = row['start_ms'], row['end_ms']
        refs.append({'index': n, 'clip_id': row['clip_id'], 'video_id': video_id,
                     'start_ms': lo, 'end_ms': hi, 'time_range': f'{lo/1000:.1f}–{hi/1000:.1f}s',
                     'text': row['text'], 'keyframe_paths': [row['frame_path']] if row.get('frame_path') else [],
                     'evidence_id': row.get('evidence_id', ''), 'kind': row.get('kind', '')})
        numbered.append({'index': n, 'start_ms': lo, 'end_ms': hi, 'text': row['text'], 'kind': row.get('kind', '')})
    if not numbered:
        result.notes.append('没有可用证据，请检查视频分析状态')
        yield 'final', result
        return
    result.candidates = len(numbered)
    budget = max_context_tokens or (policy.qa_global_input_tokens if global_scope else policy.qa_local_input_tokens)
    output = policy.qa_global_output_tokens if global_scope else policy.qa_local_output_tokens
    prefix = orch._preference_messages(history, text)
    messages = fit_messages(llm, text, video.title or '该视频', numbered, prefix, policy, ledger, budget, output)
    result.context_chars = len(json.dumps(messages, ensure_ascii=False))
    yield 'meta', {'video_id': video_id, 'candidates': result.candidates, 'scope': 'global' if global_scope else 'local'}
    yield 'status', {'phase': 'generation', 'message': '证据已就绪，正在回答…'}
    for kind, value in stream_answer(llm, messages, policy, ledger, output, budget):
        if kind == 'delta' and stream:
            yield kind, value
        elif kind == 'completed':
            result.answer = value['answer']
            result.usage = {key: value[key] for key in ('prompt_tokens', 'completion_tokens', 'finish_reason')}
            result.truncated = value['truncated']
    if result.truncated:
        result.notes.append('回答达到长度上限，续写2次后仍未完成；内容不是完整枚举')
        result.answer += '\n\n> 回答尚未完成：已达到本次输出长度预算，请缩小问题范围继续提问。'
    result.answer = _enforce_answer_claims(result, SimpleNamespace(refs=refs))
    result.refs = extract_citations(result.answer, refs)
    result.backend = 'cloud'
    result.elapsed_ms = int((time.monotonic() - started) * 1000)
    orch._repo.add_qa(video_id, question=text, answer=result.answer, refs=[r.to_dict() for r in result.refs],
                      rewritten=rewritten, elapsed_ms=result.elapsed_ms, session_id=session_id)
    orch._notify_memory()
    yield 'final', result
