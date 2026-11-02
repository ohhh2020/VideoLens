"""只读重放真实证据，比较关键词/打包耗时；不重跑视频、不操作群聊。

基线为改动前 packing.py 的独立副本；输出保存在源码目录外。
默认离线，只有显式 --cloud 才调用当前已配置的真实文本模型。
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sqlite3
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.analysis import packing
from backend.analysis.policy import estimate_tokens


def readonly(path):
    return sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)


def measures(action, rounds):
    elapsed = []
    result = None
    for _ in range(rounds):
        started = time.perf_counter()
        result = action()
        elapsed.append((time.perf_counter() - started) * 1000)
    return result, {'median_ms': round(statistics.median(elapsed), 3),
                    'max_ms': round(max(elapsed), 3), 'samples_ms': [round(t, 3) for t in elapsed]}


def markers(rows):
    # 只能作为同预算保留情况的代理指标，不等于事实准确率/召回率。
    return set(re.findall(r'[a-zA-Z0-9]+(?:[-_.+][a-zA-Z0-9]+)*[%％]?',
                          '\n'.join(row['text'] for row in rows).lower()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='修改前的packing.py副本')
    parser.add_argument('--runtime', type=Path, default=Path('../VideoLens-runtime'))
    parser.add_argument('--run', action='append', required=True, help='分析run_id，可重复')
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cloud', action='store_true', help='真实调用文本模型，将产生云端用量')
    parser.add_argument('--cloud-rounds', type=int, default=1)
    parser.add_argument('--current-prompt', action='store_true', help='两组均使用当前报告提示词，默认使用保存的历史提示词')
    parser.add_argument('--candidate-mode', choices=('experiment', 'default'), default='experiment',
                        help='对照实验模型或最终默认配置；默认配置继续使用原关键词评分')
    args = parser.parse_args()
    if args.rounds < 1 or args.cloud_rounds < 1:
        parser.error('测试轮数必须大于0')
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root):
        parser.error('测试产物请放在源码目录外的运行目录')
    spec = importlib.util.spec_from_file_location('keyword_baseline', args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    import jieba
    jieba.setLogLevel(40)
    started = time.perf_counter()
    jieba.initialize()
    output = {'kind': '只读真实证据重放，不含云端生成或视频端到端测量',
              'rounds': args.rounds, 'jieba_initialization_ms': round((time.perf_counter()-started)*1000, 3), 'runs': []}
    llm = None
    if args.cloud:
        from backend.config import load_app_config
        from backend.models.llm import LlmGateway
        llm = LlmGateway.from_config(load_app_config())
        output['kind'] = '真实证据重放及独立云端报告对照，不是视频端到端计时'
    for run_id in args.run:
        if not re.fullmatch(r'r[a-f0-9]+', run_id):
            parser.error('run_id格式不正确')
        path = args.runtime / 'data' / 'analysis-runs' / run_id / 'evidence.sqlite3'
        with readonly(path) as conn:
            evidence = [json.loads(row[0]) for row in conn.execute('SELECT data FROM evidence ORDER BY rowid')]
            policy = json.loads(conn.execute("SELECT data FROM results WHERE key='policy'").fetchone()[0])
        evidence = [{**row, 'id': row['evidence_id']} for row in evidence]
        documents = [row['text'] for row in evidence]
        instruction = policy['_prompts']['integrity_fast_report']
        if args.current_prompt:
            from backend.analysis.prompts import get_prompt
            instruction = get_prompt('integrity_fast_report')
        def messages(rows):
            return [{'role': 'system', 'content': instruction}, {'role': 'user', 'content':
                    json.dumps({'total_clips': len({row['clip_id'] for row in evidence}),
                                'evidence_units': rows}, ensure_ascii=False)}]
        budget = policy['integrity']['report_input_tokens'] - policy['integrity']['report_output_tokens'] - 512
        record = {'run_id': run_id, 'evidence_count': len(evidence), 'characters': sum(map(len, documents)),
                  'budget': budget}
        # 交替顺序，减轻温度/调度变化造成的偏差。
        values = {'baseline': [], 'candidate': []}
        selected = {}
        for _ in range(args.rounds):
            for name, module in ([('baseline', baseline), ('candidate', packing)]
                                 if _ % 2 == 0 else [('candidate', packing), ('baseline', baseline)]):
                extractor = module.legacy_extract if name == 'candidate' and args.candidate_mode == 'default' else module.keywords
                tags, extraction = measures(lambda: extractor(documents), 1)
                options = {'incremental': True, 'keyword_assist': args.candidate_mode == 'experiment'} if name == 'candidate' else {}
                (rows, meta), selection = measures(lambda: module.select(evidence, messages, budget, **options), 1)
                values[name].append((extraction['median_ms'], selection['median_ms']))
                selected[name] = rows
                record[name] = {'keywords': tags, 'selected_units': len(rows),
                    'input_tokens_estimated': estimate_tokens(json.dumps(messages(rows), ensure_ascii=False)),
                    'marker_count': len(markers(rows)), 'selected_clips': len(meta['selected_clips']),
                    'selected_unit_ids': [row['id'] for row in rows]}
        for name, times in values.items():
            record[name]['keyword_median_ms'] = round(statistics.median(t[0] for t in times), 3)
            record[name]['packing_median_ms'] = round(statistics.median(t[1] for t in times), 3)
            record[name]['samples_ms'] = times
        record['packing_delta_ms'] = round(record['candidate']['packing_median_ms'] - record['baseline']['packing_median_ms'], 3)
        if llm is not None:
            record['cloud'] = []
            for round_index in range(args.cloud_rounds):
                for name in (('baseline', 'candidate') if round_index % 2 == 0 else ('candidate', 'baseline')):
                    started = time.perf_counter()
                    response = llm.complete(messages(selected[name]),
                        max_tokens=policy['integrity']['report_output_tokens'], temperature=0.0)
                    elapsed = time.perf_counter() - started
                    target = args.output.parent / args.output.stem / f'{run_id}-{name}-{round_index + 1}.md'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(response.text, encoding='utf-8')
                    call = {'variant': name, 'round': round_index + 1, 'seconds': round(elapsed, 3),
                        'prompt_tokens': response.prompt_tokens, 'completion_tokens': response.completion_tokens,
                        'finish_reason': response.finish_reason, 'file': str(target.relative_to(args.output.parent))}
                    record['cloud'].append(call)
                    print(json.dumps(call, ensure_ascii=False), flush=True)
        output['runs'].append(record)
        compact = {key: record[key] for key in ('run_id', 'characters', 'packing_delta_ms')}
        for name in values:
            compact[name] = {key: record[name][key] for key in ('keyword_median_ms', 'packing_median_ms',
                'selected_units', 'input_tokens_estimated', 'marker_count', 'selected_clips')}
        print(json.dumps(compact, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
