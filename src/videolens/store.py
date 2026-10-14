"""SQLite 是任务、证据和会话的恢复依据，向量库只保存可重新生成的索引。

每次操作使用独立连接，WAL 支持网页读取和后台任务写入并行。
业务实体保存为 JSON，检索和约束依赖固定的 SQL 列；SQL 不拼接用户输入。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


class Database:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS videos(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, video_id TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS clips(id TEXT NOT NULL, video_id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(video_id,id));
                CREATE TABLE IF NOT EXISTS frames(id TEXT NOT NULL, video_id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(video_id,id));
                CREATE TABLE IF NOT EXISTS checkpoints(job_id TEXT NOT NULL, stage TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(job_id,stage));
                CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, event TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS events_job ON events(job_id,id);
                CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, video_id TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, payload TEXT NOT NULL);
                PRAGMA user_version=1;
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def encode(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def _get(self, table, identity):
        with self.connect() as db:
            row = db.execute(f"SELECT payload FROM {table} WHERE id=?", (identity,)).fetchone()
            return json.loads(row["payload"]) if row else None

    def _all(self, table):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute(f"SELECT payload FROM {table} ORDER BY rowid DESC")]

    def save_video(self, value):
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO videos VALUES(?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload", (value["id"], self.encode(value)))

    def get_video(self, identity):
        return self._get("videos", identity)

    def list_videos(self):
        return self._all("videos")

    def save_job(self, value):
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO jobs VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload", (value["id"], value["video_id"], self.encode(value)))

    def get_job(self, identity):
        return self._get("jobs", identity)

    def list_jobs(self):
        return self._all("jobs")

    def update_job(self, identity, **changes):
        with self.lock:
            value = self.get_job(identity)
            if value is None:
                return None
            value.update(changes, updated_at=time.time())
            self.save_job(value)
            return value

    def event(self, job_id, kind, payload):
        with self.lock, self.connect() as db:
            cur = db.execute("INSERT INTO events(job_id,event,payload) VALUES(?,?,?)", (job_id, kind, self.encode(payload)))
            return cur.lastrowid

    def events(self, job_id, after=0):
        with self.connect() as db:
            rows = db.execute("SELECT id,event,payload FROM events WHERE job_id=? AND id>? ORDER BY id LIMIT 500", (job_id, after))
            return [{"id": r[0], "event": r[1], "data": json.loads(r[2])} for r in rows]

    def checkpoint(self, job_id, stage, data):
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO checkpoints VALUES(?,?,?) ON CONFLICT(job_id,stage) DO UPDATE SET payload=excluded.payload", (job_id, stage, self.encode(data)))

    def get_checkpoint(self, job_id, stage):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM checkpoints WHERE job_id=? AND stage=?", (job_id, stage)).fetchone()
            return json.loads(row[0]) if row else None

    def clear_checkpoints(self, job_id, stages):
        with self.lock, self.connect() as db:
            db.executemany("DELETE FROM checkpoints WHERE job_id=? AND stage=?", [(job_id, stage) for stage in stages])

    def _replace_items(self, table, video_id, items):
        with self.lock, self.connect() as db:
            db.execute(f"DELETE FROM {table} WHERE video_id=?", (video_id,))
            db.executemany(f"INSERT INTO {table} VALUES(?,?,?)", [(x["id"], video_id, self.encode(x)) for x in items])

    def replace_clips(self, video_id, items):
        self._replace_items("clips", video_id, items)

    def replace_frames(self, video_id, items):
        self._replace_items("frames", video_id, items)

    def _items(self, table, video_id):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute(f"SELECT payload FROM {table} WHERE video_id=? ORDER BY id", (video_id,))]

    def clips(self, video_id):
        return self._items("clips", video_id)

    def frames(self, video_id):
        return self._items("frames", video_id)

    def create_session(self, video_id):
        value = {"id": uuid4().hex, "video_id": video_id, "created_at": time.time(), "summary": ""}
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO sessions VALUES(?,?,?)", (value["id"], video_id, self.encode(value)))
        return value

    def get_session(self, identity):
        return self._get("sessions", identity)

    def list_sessions(self, video_id):
        return self._items("sessions", video_id)

    def update_session(self, value):
        with self.lock, self.connect() as db:
            db.execute("UPDATE sessions SET payload=? WHERE id=?", (self.encode(value), value["id"]))

    def messages(self, session_id):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT payload FROM messages WHERE session_id=? ORDER BY id", (session_id,))]

    def add_message(self, session_id, value):
        value = {**value, "created_at": time.time()}
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO messages(session_id,payload) VALUES(?,?)", (session_id, self.encode(value)))

    def get_meta(self, key, default=None):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM metadata WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else default

    def set_meta(self, key, value):
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO metadata VALUES(?,?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload", (key, self.encode(value)))

    def delete_video(self, video_id):
        with self.lock, self.connect() as db:
            db.execute("DELETE FROM messages WHERE session_id IN (SELECT id FROM sessions WHERE video_id=?)", (video_id,))
            db.execute("DELETE FROM checkpoints WHERE job_id IN (SELECT id FROM jobs WHERE video_id=?)", (video_id,))
            db.execute("DELETE FROM events WHERE job_id IN (SELECT id FROM jobs WHERE video_id=?)", (video_id,))
            for table in ("clips", "frames", "jobs", "sessions"):
                db.execute(f"DELETE FROM {table} WHERE video_id=?", (video_id,))
            db.execute("DELETE FROM videos WHERE id=?", (video_id,))
            db.execute("DELETE FROM metadata WHERE key=?", ("index:" + video_id,))

    def commit_analysis(self, video, clips, frames, index_meta=None):
        """一次事务发布整代证据和报告，失败/取消的重分析不会破坏旧报告引用。"""
        with self.lock, self.connect() as db:
            for table, items in (("clips", clips), ("frames", frames)):
                db.execute(f"DELETE FROM {table} WHERE video_id=?", (video["id"],))
                db.executemany(f"INSERT INTO {table} VALUES(?,?,?)", [(item["id"], video["id"], self.encode(item)) for item in items])
            db.execute("UPDATE videos SET payload=? WHERE id=?", (self.encode(video), video["id"]))
            if index_meta:
                db.execute("INSERT INTO metadata VALUES(?,?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload", ("index:" + video["id"], self.encode(index_meta)))
            else:
                # 新证据不能继续搭配旧向量；没有新向量时安全降级关键词检索。
                db.execute("DELETE FROM metadata WHERE key=?", ("index:" + video["id"],))
