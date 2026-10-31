"""智能工作台的持久化会话；与视频元数据共用 SQLite 和短事务。"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid

logger = logging.getLogger('zhiying.storage.conversations')


DDL_V6 = (
    """CREATE TABLE conversations (
        id TEXT PRIMARY KEY, title TEXT NOT NULL, video_id TEXT NOT NULL DEFAULT '',
        session_id TEXT NOT NULL, created_ts REAL NOT NULL, updated_ts REAL NOT NULL,
        import_key TEXT UNIQUE)""",
    "CREATE INDEX idx_conversations_updated ON conversations(updated_ts DESC, id)",
    """CREATE TABLE conversation_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
        role TEXT NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL DEFAULT '',
        payload TEXT NOT NULL DEFAULT '{}', operation_id TEXT NOT NULL DEFAULT '',
        created_ts REAL NOT NULL, import_key TEXT UNIQUE,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE)""",
    "CREATE INDEX idx_conversation_messages ON conversation_messages(conversation_id,id)",
    """CREATE TABLE conversation_operations (
        id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, request_id TEXT NOT NULL,
        status TEXT NOT NULL, phase TEXT NOT NULL DEFAULT 'intent',
        payload TEXT NOT NULL, error TEXT NOT NULL DEFAULT '', progress REAL NOT NULL DEFAULT 0,
        message TEXT NOT NULL DEFAULT '', created_ts REAL NOT NULL, updated_ts REAL NOT NULL,
        UNIQUE(conversation_id,request_id),
        FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE)""",
    """CREATE UNIQUE INDEX idx_conversation_active ON conversation_operations(conversation_id)
        WHERE status IN ('queued','running')""",
    """CREATE TABLE uploaded_videos (
        id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, filename TEXT NOT NULL,
        path TEXT NOT NULL, size_bytes INTEGER NOT NULL, created_ts REAL NOT NULL,
        claimed INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE)""",
    # 墓碑阻止用户删除导入会话后，下次启动再次导入。
    "CREATE TABLE conversation_imports (import_key TEXT PRIMARY KEY)",
)


DDL_V7 = (
    "ALTER TABLE conversations ADD COLUMN title_source TEXT NOT NULL DEFAULT 'legacy'",
)


def encode(value):
    return json.dumps(value, ensure_ascii=False)


class ConversationStore:
    def __init__(self, repository):
        self.repo = repository

    def create(self, *, video_id='', title='新会话', session_id=None, import_key=None):
        from backend.storage.lifecycle import video_guard
        cid = 'c' + uuid.uuid4().hex
        now = time.time()
        with video_guard(video_id), self.repo._tx() as conn:
            if video_id and not conn.execute('SELECT 1 FROM videos WHERE video_id=?', (video_id,)).fetchone():
                raise KeyError('视频不存在或已删除')
            conn.execute(
                'INSERT INTO conversations '
                '(id,title,video_id,session_id,created_ts,updated_ts,import_key,title_source) VALUES (?,?,?,?,?,?,?,?)',
                (cid, title, video_id, session_id or cid, now, now, import_key,
                 'pending' if title == '新会话' else 'manual'),
            )
        return self.get(cid)

    def get(self, cid):
        row = self.repo._query_one('SELECT * FROM conversations WHERE id=?', (cid,))
        if row is None:
            raise KeyError('会话不存在')
        return dict(row)

    def list(self, limit=30, offset=0):
        return [dict(row) for row in self.repo._query(
            'SELECT * FROM conversations ORDER BY updated_ts DESC,id LIMIT ? OFFSET ?',
            (limit, offset),
        )]

    def update(self, cid, **values):
        from backend.storage.lifecycle import video_guard
        allowed = {'title', 'video_id'}
        if not values or not set(values) <= allowed:
            raise ValueError('会话字段无效')
        if 'title' in values:
            values['title_source'] = 'manual'
        with video_guard(values.get('video_id', self.get(cid)['video_id'])), self.repo._tx() as conn:
            if values.get('video_id') and not conn.execute('SELECT 1 FROM videos WHERE video_id=?', (values['video_id'],)).fetchone():
                raise KeyError('视频不存在或已删除')
            conn.execute(
                'UPDATE conversations SET ' + ','.join(f'{key}=?' for key in values)
                + ',updated_ts=? WHERE id=?', (*values.values(), time.time(), cid),
            )

    def set_generated_title(self, cid, title):
        """条件更新保护生成期间的手动重命名，自动命名成功后不再重复。"""
        if not title.strip() or len(title) > 15:
            raise ValueError('会话标题必须为1至15个字符')
        with self.repo._tx() as conn:
            cursor = conn.execute(
                "UPDATE conversations SET title=?,title_source='generated',updated_ts=? "
                "WHERE id=? AND title_source='pending'",
                (title, time.time(), cid),
            )
            return bool(cursor.rowcount)

    def messages(self, cid):
        self.get(cid)
        result = []
        for row in self.repo._query(
            'SELECT * FROM conversation_messages WHERE conversation_id=? ORDER BY id', (cid,),
        ):
            item = dict(row)
            item['payload'] = json.loads(item['payload'])
            result.append(item)
        return result

    def add_message(self, cid, kind, text='', *, role='assistant', payload=None, operation_id='', import_key=None):
        now = time.time()
        with self.repo._tx() as conn:
            conn.execute(
                'INSERT OR IGNORE INTO conversation_messages '
                '(conversation_id,role,kind,text,payload,operation_id,created_ts,import_key) VALUES (?,?,?,?,?,?,?,?)',
                (cid, role, kind, text, encode(payload or {}), operation_id, now, import_key),
            )
            conn.execute('UPDATE conversations SET updated_ts=? WHERE id=?', (now, cid))

    def operation(self, oid):
        row = self.repo._query_one('SELECT * FROM conversation_operations WHERE id=?', (oid,))
        if row is None:
            raise KeyError('操作不存在')
        value = dict(row)
        value['payload'] = json.loads(value['payload'])
        return value

    def active(self, cid):
        row = self.repo._query_one(
            "SELECT id FROM conversation_operations WHERE conversation_id=? AND status IN ('queued','running')",
            (cid,),
        )
        return self.operation(row['id']) if row else None

    def latest(self, cid):
        row = self.repo._query_one(
            'SELECT id FROM conversation_operations WHERE conversation_id=? ORDER BY created_ts DESC LIMIT 1', (cid,),
        )
        return self.operation(row['id']) if row else None

    def submit(self, cid, request_id, text, attachment_id='', options=None):
        from backend.storage.lifecycle import video_guard
        with video_guard(self.get(cid)['video_id']):
            return self._submit(cid, request_id, text, attachment_id, options)

    def _submit(self, cid, request_id, text, attachment_id='', options=None):
        now, oid = time.time(), 'o' + uuid.uuid4().hex
        with self.repo._tx() as conn:
            if not conn.execute('SELECT 1 FROM conversations WHERE id=?', (cid,)).fetchone():
                raise KeyError('会话不存在或已删除')
            existing = conn.execute(
                'SELECT id FROM conversation_operations WHERE conversation_id=? AND request_id=?', (cid, request_id),
            ).fetchone()
            if existing:
                return self.operation(existing['id']), False
            if conn.execute(
                "SELECT 1 FROM conversation_operations WHERE conversation_id=? AND status IN ('queued','running')", (cid,),
            ).fetchone():
                raise ValueError('当前会话正在处理，请稍后发送')
            if attachment_id:
                attachment = conn.execute(
                    'SELECT * FROM uploaded_videos WHERE id=? AND conversation_id=?', (attachment_id, cid),
                ).fetchone()
                if not attachment:
                    raise KeyError('视频附件不存在或不属于当前会话')
            payload = {'text': text, 'attachment_id': attachment_id, 'options': options or {}}
            conn.execute(
                'INSERT INTO conversation_operations '
                '(id,conversation_id,request_id,status,payload,created_ts,updated_ts) VALUES (?,?,?,?,?,?,?)',
                (oid, cid, request_id, 'queued', encode(payload), now, now),
            )
            user_payload = {'attachment': {'id': attachment_id, 'filename': attachment['filename'],
                            'size_bytes': attachment['size_bytes']}} if attachment_id else {}
            conn.execute(
                'INSERT INTO conversation_messages (conversation_id,role,kind,text,payload,operation_id,created_ts) '
                'VALUES (?,?,?,?,?,?,?)', (cid, 'user', 'text', text, encode(user_payload), oid, now),
            )
            conn.execute(
                'UPDATE conversations SET updated_ts=? WHERE id=?',
                (now, cid),
            )
        return self.operation(oid), True

    def update_operation(self, oid, **values):
        allowed = {'status', 'phase', 'payload', 'error', 'progress', 'message'}
        if not values or not set(values) <= allowed:
            raise ValueError('操作字段无效')
        values = {key: encode(value) if key == 'payload' else value for key, value in values.items()}
        with self.repo._tx() as conn:
            conn.execute(
                'UPDATE conversation_operations SET ' + ','.join(f'{key}=?' for key in values)
                + ',updated_ts=? WHERE id=?', (*values.values(), time.time(), oid),
            )

    def retry(self, cid, oid):
        from backend.storage.lifecycle import video_guard
        with video_guard(self.get(cid)['video_id']):
            return self._retry(cid, oid)

    def _retry(self, cid, oid):
        with self.repo._tx() as conn:
            row = conn.execute('SELECT * FROM conversation_operations WHERE id=? AND conversation_id=?', (oid, cid)).fetchone()
            if not row:
                raise KeyError('操作不存在')
            if row['status'] not in {'failed', 'interrupted'}:
                raise ValueError('只有失败或中断的操作可以重试')
            if self.active(cid):
                raise ValueError('当前会话正在处理')
            conn.execute(
                "UPDATE conversation_operations SET status='queued',error='',message='等待重试',updated_ts=? WHERE id=?",
                (time.time(), oid),
            )
        return self.operation(oid)

    def interrupt_pending(self):
        with self.repo._tx() as conn:
            conn.execute(
                "UPDATE conversation_operations SET status='interrupted',error='服务重启，操作已中断，请手动重试',"
                "message='已中断',updated_ts=? WHERE status IN ('queued','running')", (time.time(),),
            )

    def delete(self, cid):
        self.get(cid)
        with self.repo._tx() as conn:
            if self.active(cid):
                raise ValueError('请等待当前会话处理完成后再删除')
            row = conn.execute('SELECT * FROM conversations WHERE id=?', (cid,)).fetchone()
            conn.execute('INSERT OR IGNORE INTO conversation_imports VALUES (?)', (f"legacy:{row['video_id']}:",))
            conn.execute('INSERT OR IGNORE INTO conversation_imports VALUES (?)', (f"legacy:{row['video_id']}:{row['session_id']}",))
            conn.execute('DELETE FROM qa_history WHERE video_id=? AND session_id=?', (row['video_id'], row['session_id']))
            conn.execute('DELETE FROM conversations WHERE id=?', (cid,))

    def attachment(self, aid, cid):
        row = self.repo._query_one('SELECT * FROM uploaded_videos WHERE id=? AND conversation_id=?', (aid, cid))
        if not row:
            raise KeyError('视频附件不存在或已过期')
        return dict(row)

    def add_attachment(self, cid, aid, filename, path, size):
        with self.repo._tx() as conn:
            conn.execute('INSERT INTO uploaded_videos VALUES (?,?,?,?,?,?,0)',
                         (aid, cid, filename, str(path), size, time.time()))

    def import_history(self):
        """一次性导入旧问答/报告；稳定键与墓碑使重复启动、删除均不复生。"""
        sessions = self.repo._query('SELECT DISTINCT video_id,session_id FROM qa_history')
        videos = self.repo._query('SELECT video_id,title,display_name,report_json FROM videos')
        groups = {(row['video_id'], row['session_id']) for row in sessions}
        for video in videos:
            if not any(vid == video['video_id'] for vid, _ in groups) and not self.repo._query_one(
                'SELECT 1 FROM conversations WHERE video_id=?', (video['video_id'],)
            ):
                groups.add((video['video_id'], ''))
        by_video = {row['video_id']: row for row in videos}
        for vid, sid in sorted(groups):
            if vid not in by_video:
                continue
            key = f'legacy:{vid}:{sid}'
            if self.repo._query_one('SELECT 1 FROM conversation_imports WHERE import_key=?', (key,)):
                continue
            if self.repo._query_one('SELECT 1 FROM conversations WHERE video_id=? AND session_id=?', (vid, sid)):
                with self.repo._tx() as conn:
                    conn.execute('INSERT OR IGNORE INTO conversation_imports VALUES (?)', (key,))
                continue
            video = by_video[vid]
            cid = 'c' + hashlib.sha256(key.encode()).hexdigest()[:32]
            now = time.time()
            with self.repo._tx() as conn:
                if conn.execute('SELECT 1 FROM conversation_imports WHERE import_key=?', (key,)).fetchone():
                    continue
                conn.execute('INSERT INTO conversation_imports VALUES (?)', (key,))
                conn.execute('INSERT INTO conversations (id,title,video_id,session_id,created_ts,updated_ts,import_key) VALUES (?,?,?,?,?,?,?)',
                             (cid, video['title'] or video['display_name'], vid, sid, now, now, key))
                if video['report_json']:
                    try:
                        report = json.loads(video['report_json'])
                        if not isinstance(report, dict):
                            raise ValueError('报告不是对象')
                    except ValueError:
                        logger.warning('旧视频报告无法读取 video=%s；保留原记录并继续导入会话', vid)
                        report = {'summary': '旧报告无法读取，请从视频库检查该视频；原始记录仍然保留。'}
                    conn.execute(
                        'INSERT INTO conversation_messages (conversation_id,role,kind,text,payload,created_ts,import_key) '
                        'VALUES (?,?,?,?,?,?,?)',
                        (cid, 'assistant', 'report', report.get('summary', ''), encode(report), now, key + ':report'),
                    )
                rows = conn.execute('SELECT * FROM qa_history WHERE video_id=? AND session_id=? ORDER BY id', (vid, sid)).fetchall()
                for row in rows:
                    try:
                        refs = json.loads(row['refs'])
                    except ValueError:
                        refs = []
                    for role, kind, text, data in (
                        ('user', 'text', row['question'], {}),
                        ('assistant', 'answer', row['answer'], {'refs': refs if isinstance(refs, list) else [], 'elapsed_ms': row['elapsed_ms']}),
                    ):
                        conn.execute(
                            'INSERT INTO conversation_messages (conversation_id,role,kind,text,payload,created_ts,import_key) '
                            'VALUES (?,?,?,?,?,?,?)',
                            (cid, role, kind, text, encode(data), row['created_ts'], f"qa:{row['id']}:{role}"),
                        )
                if rows:
                    conn.execute('UPDATE conversations SET created_ts=?,updated_ts=? WHERE id=?',
                                 (rows[0]['created_ts'], rows[-1]['created_ts'], cid))
