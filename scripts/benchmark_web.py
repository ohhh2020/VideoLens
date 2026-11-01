"""通过正在运行的网页服务真实计时，强制新分析；不创建隐藏服务或复制模型。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no-transcript-cache', action='store_true', help='强制执行ASR，防止转写缓存命中混入首次分析对照')
    args = parser.parse_args()
    base = f'http://127.0.0.1:{args.port}/internal'
    def api(path, payload=None):
        request = Request(base + path, data=None if payload is None else json.dumps(payload).encode(),
                          headers={'Content-Type':'application/json'})
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    health = api('/health')
    if health['jobs_running']:
        raise SystemExit('已有分析任务，未创建测试任务')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    conversation = api('/conversations', {})
    cid = conversation.get('conversation', conversation)['id']
    started, wall = time.monotonic(), time.time()
    accepted = api(f'/conversations/{cid}/messages', {'text':args.source,'request_id':'benchmark-'+uuid4().hex,
                                                   'options':{'force':True,'reuse_transcript':not args.no_transcript_cache}})
    events, previous, last_print = [], None, 0
    while True:
        current = api(f'/conversations/{cid}')
        operation = current['operation']
        elapsed = time.monotonic()-started
        state = (operation['status'], operation['phase'], operation['message'])
        if state != previous or elapsed-last_print >= 15:
            item = {'seconds':round(elapsed,3),'status':state[0],'phase':state[1],'message':state[2]}
            events.append(item)
            print(json.dumps(item,ensure_ascii=False),flush=True)
            previous, last_print = state, elapsed
        reports = [message['payload'] for message in current['messages'] if message['kind']=='report']
        result = {'started_at':datetime.fromtimestamp(wall).isoformat(), 'source':args.source,
                  'conversation_id':cid,'operation_id':accepted['operation_id'],'wall_seconds':elapsed,
                  'server_pid':health['pid'],'devices':health['model_devices'],'status':operation['status'],
                  'error':operation.get('error',''),'events':events,'report':reports[-1] if reports else None,
                  'timing_scope':'已启动且模型已加载的网页服务；从提交消息到分析、报告和会话标题完成，包含本轮下载，不含服务启动。'}
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        if operation['status'] not in ('running','queued'):
            print(json.dumps({'result':str(output),'seconds':round(elapsed,3),'status':operation['status']},ensure_ascii=False),flush=True)
            return 0 if operation['status']=='done' else 1
        if elapsed > 1800:
            raise SystemExit('监测超过30分钟；服务任务未被取消，结果文件保留最近状态')
        time.sleep(2)


if __name__=='__main__':
    raise SystemExit(main())
