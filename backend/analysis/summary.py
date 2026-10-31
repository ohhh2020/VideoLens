"""全量证据的分层摘要树；任何叶节点遗漏都阻止完成状态。"""
from __future__ import annotations
from backend.analysis.prompts import get_prompt

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

from backend.analysis.cloud import complete, parse_json, prose
from backend.analysis.evidence import make_evidence, split_text
from backend.analysis.policy import estimate_tokens


def batches(items, policy, budget):
    output, current, used, units = [], [], 0, set()
    for item in items:
        unit = item.get('clip_id') or item['id']
        cost = estimate_tokens(json.dumps({key: item[key] for key in ('id', 'text', 'start_ms', 'end_ms', 'kind') if key in item}, ensure_ascii=False))
        if current and ((unit not in units and len(units) >= policy.summary_batch_target) or used + cost > budget or len(units) >= policy.summary_batch_max):
            output.append(current)
            current, used, units = [], 0, set()
        current.append(item)
        units.add(unit)
        used += cost
    if current:
        output.append(current)
    return output


def aligned_words(segment):
    """用字级时间定位原文，保留英文空格、标点；无法可靠对齐时保留整段。"""
    text = segment['text']
    words = segment.get('words') or []
    cursor, result = 0, []
    for word in words:
        token = word.get('word', '')
        if not token:
            return []
        pos = text.find(token, cursor)
        if pos < 0:
            return []
        if any(char.isalnum() for char in text[cursor:pos]):
            return []
        stop = pos + len(token)
        result.append({**word, 'word': text[cursor:stop]})
        cursor = stop
    if result:
        if any(char.isalnum() for char in text[cursor:]):
            return []
        result[-1]['word'] += text[cursor:]
    return result


def collect(repo, video_id, ledger, policy):
    """从原始转写重建正文，旧的切片截断不会变成新总结的截断。"""
    clips = list(repo.iter_clip_payloads(video_id))
    segments = repo.get_segments(video_id)
    frames = repo.get_keyframes(video_id)
    visual = ledger.get('visual_results') or []
    raw = ledger.get('transcript_raw')
    source_segments = raw['segments'] if raw else [{'start_ms': row.start_ms, 'end_ms': row.end_ms, 'text': row.text} for row in segments]
    prepared = [(segment, aligned_words(segment)) for segment in source_segments]
    output = []
    for clip in clips:
        cid, start, end = clip['clip_id'], clip.get('core_start_ms', clip['start_ms']), clip.get('core_end_ms', clip['end_ms'])
        components = []
        for segment, words in prepared:
            if words:
                selected = [word for word in words if start <= word['start_ms'] < end]
                if selected:
                    components.append(('speech', selected[0]['start_ms'], selected[-1]['end_ms'],
                        ''.join(word.get('word', '') for word in selected), {'observation': 'word_aligned_transcript'}))
            elif start <= segment['start_ms'] < end:
                components.append(('speech', segment['start_ms'], segment['end_ms'], segment['text'], {}))
        if not source_segments and (clip.get('core_text') or clip.get('text')):
            components.append(('speech', start, end, clip.get('core_text') or clip.get('text'), {}))
        # 新版视觉结果保留独立来源及准确观察时间，不拼成同一段重复叙述。
        relevant = [row for row in visual if (row['start_ms'] < end and row['end_ms'] > start)
                    or (start <= row['start_ms'] < end)]
        for row in relevant:
            components.append((row['kind'], row['start_ms'], row['end_ms'], (('【识别不确定】' if row.get('uncertain') else '') + row['text']),
                               {'frame_path': row.get('frame_path', ''), 'region': row.get('region', []),
                                'source_ids': row.get('source_ids', []), 'observation': row.get('observation', 'frame_sample')}))
        if not visual:
            for frame in frames:
                if start <= frame.timestamp_ms < end:
                    for kind, text_value in [('visual', frame.description), ('ocr', frame.ocr_text)]:
                        if text_value:
                            components.append((kind, frame.timestamp_ms, frame.timestamp_ms, text_value,
                                               {'frame_path': frame.path, 'observation': 'frame_sample'}))
        if not components:
            components = [('empty', start, end, '此区间未提取到可用语音或视觉文字，不能推断内容。', {})]
        for kind, lo, hi, body, extra in components:
            for part in split_text(body, max_bytes=min(6000, max(256, (policy.summary_input_tokens - policy.summary_output_tokens - 1600) // 3))):
                evidence = make_evidence(ledger.run_id, video_id, cid, kind, lo, hi, part, **extra)
                ledger.evidence(evidence)
                output.append({'id': evidence.evidence_id, 'clip_id': cid, 'kind': kind,
                               'start_ms': lo, 'end_ms': hi, 'text': part})
    return clips, output


def summarize(repo, video_id, title, llm, policy, ledger, preference=lambda messages: messages):
    clips, evidence = collect(repo, video_id, ledger, policy)
    if not evidence:
        raise ValueError('没有有效切片可供总结')
    tree = []
    # 同一切片内完全重复的文字只总结一次，全部原始证据及离散时间仍保留。
    expected_evidence = {row['id'] for row in evidence}
    merged = {}
    for row in evidence:
        key = (row['clip_id'],row['kind'],row['text'])
        if key not in merged:
            merged[key] = {**row, 'evidence_ids': [row['id']]}
        else:
            group = merged[key]
            group['evidence_ids'].append(row['id'])
            group['start_ms'] = min(group['start_ms'],row['start_ms'])
            group['end_ms'] = max(group['end_ms'],row['end_ms'])
    evidence = list(merged.values())

    def node(items, level):
        identity = items if level == 0 else {'items': items, 'level': level, 'outline_version': 1}
        key = 'summary:' + hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        saved = ledger.get(key)
        if saved:
            tree.extend(saved['nodes'])
            return saved['result']
        prompt = [
            {'role': 'system', 'content': get_prompt('integrity_summary', ledger)},
            {'role': 'user', 'content': json.dumps([{**{key: row[key] for key in ('text', 'start_ms', 'end_ms', 'kind') if key in row}, 'id': str(index+1)} for index, row in enumerate(items)], ensure_ascii=False)},
        ]
        if level > 0:
            prompt[0]['content'] += (' 本层用于整片概览：summary应独立说明本章的主要结论、关键数字与单位、比较关系和必要步骤，'
                                    '不要只写“包含若干数据”，也不要重复罗列全部日志。facts仍保存完整重要明细，供本地章节索引追溯。')
        if estimate_tokens(json.dumps(prompt, ensure_ascii=False)) + policy.summary_output_tokens + 512 > policy.summary_input_tokens:
            if len(items) <= 1:
                raise ValueError('单条证据超过总预算，请提高章节输入预算')
            middle = len(items) // 2
            children = [node(items[:middle], level), node(items[middle:], level)]
            return {'id': key, 'text': '\n'.join(row['text'] for row in children),
                    'clip_ids': sorted({cid for row in children for cid in row['clip_ids']}),
                    'evidence_ids': sorted({eid for row in children for eid in row['evidence_ids']}),
                    'start_ms': min(row['start_ms'] for row in children), 'end_ms': max(row['end_ms'] for row in children)}
        result = complete(llm, prompt, policy, ledger, task='章节证据摘要', output_tokens=policy.summary_output_tokens)
        try:
            if result.finish_reason == 'length':
                raise ValueError('结构化摘要被截断')
            parsed = parse_json(result.text).get('items')
            expected = {str(index+1) for index in range(len(items))}
            if not isinstance(parsed, list) or len(parsed) != len(items) or {row.get('id') for row in parsed} != expected:
                raise ValueError('摘要未覆盖全部证据ID')
            if any(not isinstance(row.get('summary'), str) or not row['summary'].strip() or
                   not isinstance(row.get('facts'), list) or any(not isinstance(f, str) for f in row['facts']) for row in parsed):
                raise ValueError('摘要结构或正文无效')
        except (ValueError, TypeError, AttributeError, KeyError):
            if len(items) <= 1:
                ledger.forget_request(result.cache_key)
                original = items[0]
                size = len(original['text'].encode('utf-8'))
                if size <= 128:
                    raise ValueError('最小子证据摘要仍不完整，请重试或提高预算；原始证据已保留')
                # 一条证据也可能列出大量数字，不能仅按条数拆批后就停止拆分。
                parts = split_text(original['text'], max_bytes=max(64, size // 2))
                children = [{**original, 'id': original['id'] + f':part{index}', 'text': part,
                             'evidence_ids': original.get('evidence_ids', [original['id']])}
                            for index, part in enumerate(parts)]
                return node(children, level)
            middle = len(items) // 2
            children = [node(items[:middle], level), node(items[middle:], level)]
            return {'id': key, 'text': '\n'.join(row['text'] for row in children),
                    'clip_ids': sorted({cid for row in children for cid in row['clip_ids']}),
                    'evidence_ids': sorted({eid for row in children for eid in row['evidence_ids']}),
                    'start_ms': min(row['start_ms'] for row in children), 'end_ms': max(row['end_ms'] for row in children)}
        lookup = {str(index+1): row for index, row in enumerate(items)}
        rows = []
        for summary in parsed:
            original = lookup[summary['id']]
            rows.append({'id': original['id'], 'level': level, 'summary': summary['summary'], 'facts': summary['facts'],
                         'clip_ids': original.get('clip_ids', [original['clip_id']] if 'clip_id' in original else []),
                         'evidence_ids': original.get('evidence_ids', [original['id']]),
                         'start_ms': original['start_ms'], 'end_ms': original['end_ms']})
        tree.extend(rows)
        # 明细保存在每一层rows及源证据中；高层只传递所有章节的概述，防止反复搬运同一事实清单。
        combined = {'id': key, 'text': '\n'.join(f"[{row['start_ms']/1000:.1f}–{row['end_ms']/1000:.1f}s] {row['summary']} " + ('；'.join(row['facts']) if level == 0 else '') for row in rows),
                    'clip_ids': sorted({cid for row in rows for cid in row['clip_ids']}),
                    'evidence_ids': sorted({eid for row in rows for eid in row['evidence_ids']}),
                    'start_ms': min(row['start_ms'] for row in rows), 'end_ms': max(row['end_ms'] for row in rows)}
        ledger.put(key, {'result': combined, 'nodes': rows})
        return combined

    level = 0
    def summarize_groups(groups, current_level):
        with ThreadPoolExecutor(max_workers=policy.cloud_text) as pool:
            return list(pool.map(lambda group: node(group, current_level), groups))
    nodes = summarize_groups(batches(evidence, policy, policy.summary_input_tokens - policy.summary_output_tokens - 1600), level)
    # 最终输入超预算时继续汇总，而非取前N章。每一层保留所有子节点来源。
    while estimate_tokens(json.dumps([row['text'] for row in nodes], ensure_ascii=False)) > policy.report_input_tokens - policy.report_output_tokens - 2000:
        level += 1
        if level > 12:
            raise ValueError('摘要层级未收敛，请提高摘要预算；原始证据与已完成章节保留')
        prepared = []
        for row in nodes:
            for index, part in enumerate(split_text(row['text'], max_bytes=max(256, policy.summary_input_tokens // 3))):
                prepared.append({**row, 'id': row['id'] + f':{level}:{index}', 'text': part})
        nodes = summarize_groups(batches(prepared, policy, policy.summary_input_tokens - policy.summary_output_tokens - 1600), level)
    covered = {cid for row in nodes for cid in row['clip_ids']}
    covered_evidence = {eid for row in nodes for eid in row['evidence_ids']}
    expected = {clip['clip_id'] for clip in clips}
    if covered != expected or covered_evidence != expected_evidence:
        raise ValueError('切片覆盖校验失败，禁止标记全部完成')
    tree = sorted({row['id']: row for row in tree}.values(), key=lambda row: (row['level'], row['start_ms'], row['id']))
    ledger.put('summary_tree', tree)
    ledger.put('coverage', {'total_clips': len(expected), 'covered_clips': len(covered), 'ratio': 1.0,
                            'evidence_count': len(expected_evidence), 'summary_inputs': len(evidence),
                            'covered_evidence_count': len(covered_evidence),
                            'meaning': '切片处理覆盖率，不代表逐帧观看或语义召回率'})
    prompt = [
        {'role': 'system', 'content': get_prompt('integrity_report', ledger)},
        {'role': 'user', 'content': json.dumps({'title': title, 'chapters': [{'id': row['id'], 'start_ms': row['start_ms'], 'end_ms': row['end_ms'], 'text': row['text']} for row in nodes]}, ensure_ascii=False)},
    ]
    prompt = preference(prompt)
    if estimate_tokens(json.dumps(prompt, ensure_ascii=False)) + policy.report_output_tokens + 512 > policy.report_input_tokens:
        raise ValueError("最终报告总预算不足，未截掉尾部章节")
    text, truncated = prose(llm, prompt, policy, ledger, task='整片综合报告', output_tokens=policy.report_output_tokens)
    ledger.put('report_truncated', truncated)
    return text
