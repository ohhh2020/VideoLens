"""只访问本机内部工作流；异步任务使用短调用轮询，不重复加载模型。"""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
from urllib.parse import urlsplit
from urllib.request import Request, ProxyHandler, build_opener
from urllib.error import HTTPError, URLError


def bilibili_url(value):
    """只接收真实B站视频/短链接，拒绝本地路径与其他站点。"""
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {'http', 'https'} or parsed.username or parsed.password:
        raise ValueError('只允许Bilibili视频链接')
    if parsed.port not in {None, 80, 443}:
        raise ValueError('不允许自定义视频端口')
    host = (parsed.hostname or '').lower()
    if host in {'www.bilibili.com', 'bilibili.com', 'm.bilibili.com'}:
        match = re.fullmatch(r'/video/(BV[0-9A-Za-z]{10}|av\d+)/?', parsed.path)
        if match:
            return 'https://www.bilibili.com/video/' + match[1]
    if host == 'b23.tv' and re.fullmatch(r'/[A-Za-z0-9]{4,32}/?', parsed.path):
        return 'https://b23.tv' + parsed.path.rstrip('/')
    raise ValueError('未识别到Bilibili视频地址')


class Bridge:
    def __init__(self, base=None, state=None):
        self.base = (base or os.getenv('VIDEOLENS_URL', 'http://127.0.0.1:8000')).rstrip('/')
        parsed = urlsplit(self.base)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'} or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            raise ValueError('VideoLens桥接地址必须是本机HTTP服务根地址')
        default = Path(__file__).resolve().parents[3] / 'VideoLens-runtime/data/mcp/qqbot.sqlite3'
        self.state = Path(state or os.getenv('VIDEOLENS_MCP_STATE', str(default)))
        self.state.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.http = build_opener(ProxyHandler({}))
        with closing(self.connect()) as conn, conn:
            conn.executescript('''CREATE TABLE IF NOT EXISTS sessions(
                id TEXT PRIMARY KEY,conversation TEXT NOT NULL,url TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,conversation TEXT NOT NULL,
                result TEXT NOT NULL DEFAULT '');''')

    def connect(self):
        return sqlite3.connect(self.state, timeout=30)

    def api(self, path, payload=None):
        request = Request(self.base + '/internal' + path,
            data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode(),
            headers={'Content-Type': 'application/json'})
        try:
            with self.http.open(request, timeout=15) as response:
                return json.load(response)
        except HTTPError as exc:
            try:
                data = json.loads(exc.read()); detail = data.get('detail', '')
            except (ValueError, OSError):
                detail = ''
            raise RuntimeError(f'VideoLens请求失败({exc.code})：{detail or "请检查服务日志"}') from None
        except (URLError, TimeoutError) as exc:
            raise RuntimeError('VideoLens本机服务不可用，请在可见Git Bash中启动网页服务') from exc

    @staticmethod
    def identity(value):
        if not value or len(value) > 512:
            raise ValueError('会话标识必须为1～512字符')
        return hashlib.sha256(value.encode()).hexdigest()

    def _submit(self, cid, text, request_id):
        rid = 'mcp-' + self.identity(request_id)
        accepted = self.api(f'/conversations/{cid}/messages',
            {'text': text, 'request_id': rid, 'options': {'force': False}})
        oid = accepted['operation_id']
        with closing(self.connect()) as conn, conn:
            conn.execute('INSERT OR IGNORE INTO jobs(id,conversation) VALUES(?,?)', (oid, cid))
        return {'job_id': oid, 'status': 'accepted'}

    def analyze(self, url, session_key, request_id, question=''):
        url, sid = bilibili_url(url), self.identity(session_key)
        question = question.strip()
        if len(question) > 2000:
            raise ValueError('随附问题最多2000字符')
        with self.lock:
            with closing(self.connect()) as conn:
                row = conn.execute('SELECT conversation,url FROM sessions WHERE id=?', (sid,)).fetchone()
            if row and row[1] != url:
                raise ValueError('同一视频消息会话不能替换为其他视频')
            if row:
                cid = row[0]
                snapshot = self.api(f'/conversations/{cid}')
                if (snapshot.get('operation') or {}).get('status') in {'running', 'queued'}:
                    oid = snapshot['operation']['id']
                    with closing(self.connect()) as conn, conn:
                        conn.execute('INSERT OR IGNORE INTO jobs(id,conversation) VALUES(?,?)', (oid, cid))
                    return {'job_id': oid, 'status': 'running'}
                if snapshot['conversation'].get('video_id'):
                    reports = [m for m in snapshot['messages'] if m['kind'] == 'report']
                    if reports and not question:
                        return {'status': 'done', 'markdown': reports[-1]['payload'].get('summary', reports[-1]['text']),
                                'kind': 'report', 'reused': True}
            else:
                cid = self.api('/conversations', {})['id']
                with closing(self.connect()) as conn, conn:
                    conn.execute('INSERT INTO sessions VALUES(?,?,?)', (sid, cid, url))
            # 已绑定视频时直接问答，避免将来源链接当作更换视频。
            text = question if row and snapshot['conversation'].get('video_id') else url + ('\n' + question if question else '')
            return self._submit(cid, text, request_id)

    def ask(self, session_key, question, request_id):
        question = question.strip()
        if not question or len(question) > 2000:
            raise ValueError('追问必须为1～2000字符')
        with self.lock:
            with closing(self.connect()) as conn:
                row = conn.execute('SELECT conversation FROM sessions WHERE id=?', (self.identity(session_key),)).fetchone()
            if not row:
                raise ValueError('请先回复这条视频发送“视频分析”')
            snapshot = self.api(f'/conversations/{row[0]}')
            if not snapshot['conversation'].get('video_id'):
                raise ValueError('视频尚未分析完成，请等待报告')
            return self._submit(row[0], question, request_id)

    def result(self, job_id):
        with self.lock:
            with closing(self.connect()) as conn:
                row = conn.execute('SELECT conversation,result FROM jobs WHERE id=?', (job_id,)).fetchone()
            if not row:
                raise ValueError('MCP任务不存在')
            snapshot = self.api(f'/conversations/{row[0]}')
            messages = [m for m in snapshot['messages'] if m.get('operation_id') == job_id and m['kind'] in {'report', 'answer'}]
            outputs = [{'kind': m['kind'], 'markdown': m['text'],
                        'refs': m.get('payload', {}).get('refs', [])} for m in messages]
            if row[1]:
                data = json.loads(row[1])
                data.pop('outputs', None)
                if data['status'] == 'done':
                    data['outputs'] = outputs
                return data
            op = snapshot.get('operation') or {}
            if op.get('id') != job_id:
                raise ValueError('任务状态已变化，请重新触发')
            data = {'job_id': job_id, 'status': op.get('status', 'unknown'), 'phase': op.get('phase'),
                    'progress': op.get('progress', 0), 'message': op.get('message', ''), 'error': op.get('error', '')}
            if op.get('status') == 'done':
                data['outputs'] = outputs
                if not messages:
                    data.update(status='failed', error='没有返回报告或回答，请检查云端配置')
            if data['status'] in {'done', 'failed', 'interrupted'}:
                with closing(self.connect()) as conn, conn:
                    # 正文只保留在业务会话，不在桥接库另存一份；视频删除后不可从缓存取回。
                    metadata = {key: value for key, value in data.items() if key != 'outputs'}
                    conn.execute('UPDATE jobs SET result=? WHERE id=?', (json.dumps(metadata, ensure_ascii=False), job_id))
            return data

    def status(self):
        result = self.api('/health')
        return {key: result.get(key) for key in ('status', 'model_devices', 'jobs_running')}
