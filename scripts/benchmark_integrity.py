"""在隔离数据目录实测完整链路；复用现有三个模型与云端设置，不改业务库。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description='完整性分析实测，运行会调用已配置的真实云端模型')
    parser.add_argument('--source', action='append', required=True, help='本地视频或可下载URL，可重复')
    parser.add_argument('--output', required=True, help='独立验收目录，不得使用业务数据目录')
    parser.add_argument('--retry-failed', action='store_true', help='显式恢复同目录内最近失败运行；耗时仅代表恢复阶段')
    parser.add_argument('--force', action='store_true', help='显式创建新的隔离分析版本')
    parser.add_argument('--baseline', action='store_true', help='使用兼容旧链路作为对照')
    args = parser.parse_args()
    from backend.config import load_app_config
    from backend.pipeline.factory import build_orchestrator
    from backend.pipeline.domain import AnalyzeOptions
    from backend.models.devices import active_devices
    config = load_app_config()
    output = Path(args.output).resolve()
    if output == config.path('data').resolve() or config.path('sqlite').resolve().is_relative_to(output):
        raise ValueError('验收目录不能包含业务数据库')
    output.mkdir(parents=True, exist_ok=True)
    values = {key: str(output / key) for key in ('data', 'videos', 'audio', 'frames', 'temp', 'chroma', 'exports', 'logs')}
    values.update(sqlite=str(output / 'data' / 'benchmark.sqlite3'), models=str(config.path('models')))
    config = replace(config, paths=replace(config.paths, **values), memory=replace(config.memory, file=str(output/'memory.json')))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[
        logging.StreamHandler(), logging.FileHandler(output / 'benchmark.log', encoding='utf-8')])
    start = time.monotonic()
    orch = build_orchestrator(config)
    orch._memory.close()
    orch._memory = None
    if args.baseline:
        orch._use_integrity = False
    warmup_seconds = time.monotonic()-start
    results = []
    try:
        import torch
        for source in args.source:
            stop = threading.Event()
            peaks = {'gpu_device_used_peak_bytes': 0, 'gpu_process_torch_peak_bytes': 0}
            def monitor():
                while not stop.wait(.1):
                    if torch.cuda.is_available():
                        free, total = torch.cuda.mem_get_info()
                        peaks['gpu_device_used_peak_bytes'] = max(peaks['gpu_device_used_peak_bytes'], total-free)
                        peaks['gpu_process_torch_peak_bytes'] = max(peaks['gpu_process_torch_peak_bytes'], torch.cuda.max_memory_allocated())
            watcher = threading.Thread(target=monitor, daemon=True)
            watcher.start()
            started = time.monotonic()
            try:
                from uuid import uuid4
                operation = 'benchmark-' + (uuid4().hex if args.force else str(len(results)))
                if args.retry_failed:
                    from backend.media.parser import parse_source, ParseOptions
                    vid = parse_source(source, ParseOptions(do_probe=False)).video_id
                    failed = orch.repository._query_one("SELECT operation_key FROM analysis_runs WHERE video_id=? AND status IN ('failed','interrupted') ORDER BY created_ts DESC LIMIT 1", (vid,))
                    if failed:
                        operation = failed['operation_key']
                report = orch.analyze(source, AnalyzeOptions(task_id=operation, force=args.force))
                item = {'attempt_type': 'manual_retry' if args.retry_failed else 'fresh', 'source': source, 'wall_seconds': time.monotonic()-started, 'devices': active_devices(orch),
                        'report': report.to_dict(), 'gpu': peaks,
                        'gpu_note': 'device峰值包含其他进程；torch峰值不包含CTranslate2分配。',
                        'sample_note': '来源与人工标注需单独记录；自动覆盖率不是事实召回率。'}
                if report.ok:
                    try:
                        qa = orch.ask(report.video_id, '请梳理视频的全部关键步骤、数字和结论，保留时间引用。', session_id='benchmark')
                        item['qa'] = qa.to_dict()
                    except Exception as exc:
                        item['qa_error'] = f'{type(exc).__name__}: {exc}'
                results.append(item)
                (output / 'results.json').write_text(json.dumps({'startup_seconds': warmup_seconds, 'baseline': args.baseline, 'results': results},
                    ensure_ascii=False, indent=2), encoding='utf-8')
                print(json.dumps({'source': source, 'ok': report.ok, 'seconds': item['wall_seconds'],
                                  'coverage': report.profile.get('coverage'), 'errors': report.errors}, ensure_ascii=False), flush=True)
            finally:
                stop.set()
                watcher.join()
    finally:
        orch.close()
    return 0 if all(row['report']['ok'] and not row.get('qa_error') for row in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
