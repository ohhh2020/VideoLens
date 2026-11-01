"""同预算下完整事实选取：保留原句/表格/代码，OCR去重不丢时间关联。"""
from collections import Counter, defaultdict
import hashlib
import json
import logging
import math
import re

from backend.analysis.policy import estimate_tokens

logger = logging.getLogger('zhiying.analysis')
FACT = re.compile(r'\d|不支持|不能|不得|必须|仅限|除外|前提|注意|先.+再|领先|落后|提高|下降|增加|减少|步骤|限制|条件|never|must|only', re.I)
RELATION = re.compile(r'领先|落后|提高|下降|增加|减少|高于|低于|不支持|不能|必须|仅限|除外|前提|\d.*[%％]|步骤|第[一二三四五六七八九十\d]+步|先.+再', re.I)
STOP = {'用户','视频','这个','那个','我们','你们','他们','就是','可以','一个','然后','因为','所以','还有','进行','以及','一些','不是','这样','现在','已经','没有','时候','这次','大家','比较','内容','画面','文字','一下','可能','如果','对于'}


def atomic_units(text):
    """不按字符切句；表格行带列头，代码块保留缩进。超长单元留原文而非截尾。"""
    lines, result, normal, table, code = text.replace('\r\n', '\n').splitlines(), [], [], [], []
    def flush_normal():
        if normal:
            paragraph = '\n'.join(normal).strip()
            result.extend(part.strip() for part in re.split(r'(?<=[。！？；!?;])\s*', paragraph) if part.strip())
            normal.clear()
    def flush_table():
        if table:
            if len(table) <= 3:
                result.append('\n'.join(table))
            else:
                header = '\n'.join(table[:2])
                result.extend(header + '\n' + row for row in table[2:])
            table.clear()
    for line in lines:
        if line.lstrip().startswith('```'):
            flush_normal(); flush_table()
            code.append(line)
            if len(code) > 1:
                result.append('\n'.join(code)); code.clear()
        elif code:
            code.append(line)
        elif line.count('|') >= 2:
            flush_normal(); table.append(line)
        elif not line.strip():
            flush_normal(); flush_table()
        else:
            flush_table(); normal.append(line)
    flush_normal(); flush_table()
    if code:
        result.append('\n'.join(code))
    return result


def keywords(documents, limit=24):
    import jieba
    jieba.setLogLevel(logging.WARNING)
    tokenized = []
    for text in documents:
        words = [word.lower() for word in jieba.lcut(text)
                 if 2 <= len(word.strip()) <= 24 and word not in STOP and re.search(r'[\w\u4e00-\u9fff]', word)
                 and not word.isdecimal()]
        words += re.findall(r'\b[a-zA-Z][a-zA-Z0-9_.+-]*\d[a-zA-Z0-9_.+-]*\b', text)
        tokenized.append(words)
    df, scores = Counter(), Counter()
    for words in tokenized:
        df.update(set(words))
    for words in tokenized:
        for word, count in Counter(words).items():
            scores[word] += (1 + math.log(count)) * (1 + math.log((len(documents) + 1) / (df[word] + 1)))
    top = scores.most_common(limit)
    scale = top[0][1] if top else 1
    return [(word, value / scale) for word, value in top]


def candidates(evidence, semantic=None):
    """重复OCR仅打包一次；每个出现位置和源证据ID完整保存在关联清单。"""
    grouped = {}
    for row in evidence:
        for text in atomic_units(row['text']):
            key = (row['kind'], text)
            occurrence = {'evidence_id': row['id'], 'clip_id': row['clip_id'],
                          'start_ms': row['start_ms'], 'end_ms': row['end_ms']}
            if key in grouped:
                if occurrence not in grouped[key]['occurrences']:
                    grouped[key]['occurrences'].append(occurrence)
                continue
            uid = 'u' + hashlib.sha256((row['kind'] + ':' + text).encode()).hexdigest()[:16]
            grouped[key] = {'id': uid, 'kind': row['kind'], 'text': text, 'start_ms': row['start_ms'],
                'end_ms': row['end_ms'], 'occurrences': [occurrence], 'fact': bool(FACT.search(text))}
    units = list(grouped.values())
    tags = keywords([row['text'] for row in evidence])
    for unit in units:
        text = unit['text'].lower()
        unit['score'] = (1 + (3 if unit['fact'] else 0) + (4 if RELATION.search(text) else 0)
                        + sum(weight for word, weight in tags if word.lower() in text))
        # 仅在规则与关键词不足、现有BGE向量已可用时补充，不额外推理或等待GPU。
        if not tags and semantic:
            unit['score'] += max((semantic.get(row['evidence_id'], 0) for row in unit['occurrences']), default=0)
    return units, tags


def select(evidence, make_messages, budget, *, semantic=None):
    units, tags = candidates(evidence, semantic)
    all_clips = sorted({row['clip_id'] for row in evidence}, key=lambda cid: min(row['start_ms'] for row in evidence if row['clip_id'] == cid))
    selected, selected_ids, included_sources = [], set(), set()
    omitted = []
    def payload(unit):
        seconds = max(0, unit['start_ms'] // 1000)
        return {'id': unit['id'], 'kind': unit['kind'], 'time': f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}',
                'text': unit['text'], 'repeats': len(unit['occurrences'])}
    def add(unit, cap=budget):
        if unit['id'] in selected_ids:
            return True
        candidate = payload(unit)
        if estimate_tokens(json.dumps(make_messages([*selected, candidate]), ensure_ascii=False)) > cap:
            return False
        selected.append(candidate); selected_ids.add(unit['id'])
        included_sources.update(row['evidence_id'] for row in unit['occurrences'])
        return True
    ranked = sorted(units, key=lambda u: (-(u['score'] / max(80, len(u['text'].encode()))**.5), u['start_ms']))
    # 首尾和三类证据保底，其余按时间分层覆盖；不是把前N片塞满。
    for kind in ('speech', 'ocr', 'visual'):
        subset = [unit for unit in ranked if unit['kind'] == kind]
        if subset:
            add(subset[0])
    by_clip = defaultdict(list)
    for unit in ranked:
        for cid in {row['clip_id'] for row in unit['occurrences']}:
            by_clip[cid].append(unit)
    positions = []
    for n in (2, 4, 8, 16, 32, 64):
        positions.extend(round(i * (len(all_clips) - 1) / max(1, n - 1)) for i in range(min(n, len(all_clips))))
    for index in dict.fromkeys(positions):
        cid = all_clips[index]
        if by_clip[cid]:
            add(by_clip[cid][0], cap=max(estimate_tokens(json.dumps(make_messages([]), ensure_ascii=False)) + 512, int(budget * .55)))
    for unit in ranked:
        if not add(unit):
            omitted.append(unit['id'])
    selected.sort(key=lambda row: (row['time'], row['kind'], row['id']))
    selected_clips = {item['clip_id'] for unit in units if unit['id'] in selected_ids for item in unit['occurrences']}
    metadata = {'unit_count': len(units), 'selected_units': len(selected), 'omitted_units': omitted,
        'omitted_count': len(units) - len(selected), 'selected_clips': [cid for cid in all_clips if cid in selected_clips],
        'total_clips': len(all_clips), 'selected_evidence_ids': sorted(included_sources),
        'selected_characters': sum(len(row['text']) for row in selected),
        'raw_characters': sum(len(row['text']) for row in evidence), 'keywords': [word for word, _ in tags],
        'unit_provenance': [{**unit, 'selected': unit['id'] in selected_ids} for unit in units],
        'semantic_assist_used': bool(not tags and semantic)}
    if not selected:
        raise ValueError('报告预算不足以容纳任何完整证据，未用截断正文冒充完成')
    return selected, metadata


def summarize(repo, video_id, title, llm, policy, ledger, preference=lambda messages: messages):
    from backend.analysis.summary import collect
    from backend.analysis.cloud import prose
    clips, evidence = collect(repo, video_id, ledger, policy)
    from backend.analysis.prompts import get_prompt
    instruction = get_prompt('integrity_fast_report', ledger)
    def messages(rows):
        return preference([{'role': 'system', 'content': instruction}, {'role': 'user', 'content':
            json.dumps({'title': title, 'total_clips': len(clips), 'evidence_units': rows}, ensure_ascii=False)}])
    budget = policy.report_input_tokens - policy.report_output_tokens - 512
    rows, metadata = select(evidence, messages, budget, semantic=ledger.get('evidence_semantic_scores'))
    ledger.put('report_selection', metadata)
    ledger.put('report_provenance', metadata['unit_provenance'])
    ledger.put('report_keywords', metadata['keywords'])
    text, truncated = prose(llm, messages(rows), policy, ledger, task='完整事实单次报告-v4', output_tokens=policy.report_output_tokens)
    ledger.put('coverage', {'total_clips': len(clips), 'covered_clips': len(metadata['selected_clips']),
        'ratio': len(metadata['selected_clips']) / max(1, len(clips)), 'evidence_count': len(evidence),
        'covered_evidence_count': len(metadata['selected_evidence_ids']), 'representative': True,
        'selected_units': metadata['selected_units'], 'total_units': metadata['unit_count'],
        'omitted_units': metadata['omitted_count'], 'meaning': '同预算完整事实选取，未穷尽所有原始证据'})
    ledger.put('summary_tree', [])
    ledger.put('report_truncated', truncated)
    return text
