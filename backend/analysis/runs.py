"""版本隔离：在独立SQLite中构建，成功后事务发布，旧分析始终可读。"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, fields, is_dataclass, replace
import json
from pathlib import Path
import shutil
import sqlite3
import threading
import time
from uuid import uuid4

from backend.analysis.policy import VERSION

DDL_V8 = (
    """CREATE TABLE analysis_runs (run_id TEXT PRIMARY KEY, video_id TEXT NOT NULL,
        operation_key TEXT UNIQUE, version TEXT NOT NULL, status TEXT NOT NULL,
        policy_json TEXT NOT NULL, work_dir TEXT NOT NULL, created_ts REAL NOT NULL,
        updated_ts REAL NOT NULL, error TEXT NOT NULL DEFAULT '')""",
    "CREATE INDEX idx_runs_video ON analysis_runs(video_id,created_ts)",
    """CREATE TABLE upload_sessions (id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
        filename TEXT NOT NULL, size_bytes INTEGER NOT NULL, fingerprint TEXT NOT NULL,
        chunk_bytes INTEGER NOT NULL, path TEXT NOT NULL, chunks TEXT NOT NULL DEFAULT '{}',
        created_ts REAL NOT NULL, updated_ts REAL NOT NULL, attachment_id TEXT NOT NULL DEFAULT '',
        FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE)""",
)


def snapshot(config):
    """只保存分析策略，云端密钥不进入参数快照。"""
    result = {name: asdict(getattr(config, name)) for name in
              ('integrity', 'chunk', 'retrieval', 'report', 'limits', 'media')}
    result['vl'] = {key: value for key, value in asdict(config.vl).items() if key not in {'cloud_api_key', 'cloud_api_key_env'}}
    result['llm'] = {'cloud': {key: value for key, value in asdict(config.llm.cloud).items() if key not in {'api_key', 'api_key_env'}}}
    from backend.prompts import get_prompt_registry
    registry = get_prompt_registry()
    result['_prompts'] = {key: registry.get(key) for key in registry.keys() if key.startswith('integrity_')}
    result['_analysis_strategy'] = VERSION
    return result


def restore(config, values):
    def merge(obj, data):
        return replace(obj, **{key: merge(getattr(obj, key), value)
                              if is_dataclass(getattr(obj, key)) else value
                              for key, value in data.items() if not key.startswith('_')})
    return merge(config, values)


def disk_check(path, reserve=0, incoming=0):
    path = Path(path)
    while not path.exists():
        path = path.parent
    if shutil.disk_usage(path).free < reserve + incoming:
        raise OSError('磁盘安全余量不足，请在视频库管理缓存后重试；已有视频和报告不会被自动删除')


class RunLedger:
    def __init__(self, directory, run_id):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.path = self.directory / 'evidence.sqlite3'
        self.write_guard = lambda incoming: None
        self.lock = threading.RLock()
        with self.connect() as conn:
            conn.executescript('''
            CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS results(key TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests(key TEXT PRIMARY KEY, task TEXT NOT NULL,
              data TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY, task TEXT NOT NULL,
              status TEXT NOT NULL, data TEXT NOT NULL, created REAL NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        with self.lock:
            conn = sqlite3.connect(self.path, timeout=30)
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.close()

    def put(self, key, data):
        self.write_guard(len(json.dumps(data, ensure_ascii=False).encode()))
        with self.connect() as conn:
            conn.execute('INSERT OR REPLACE INTO results VALUES (?,?)', (key, json.dumps(data, ensure_ascii=False)))

    def get(self, key):
        with self.connect() as conn:
            row = conn.execute('SELECT data FROM results WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def evidence(self, evidence):
        with self.connect() as conn:
            conn.execute('INSERT OR REPLACE INTO evidence VALUES (?,?)',
                         (evidence.evidence_id, json.dumps(evidence.to_dict(), ensure_ascii=False)))

    def all_evidence(self):
        with self.connect() as conn:
            return [json.loads(row[0]) for row in conn.execute('SELECT data FROM evidence ORDER BY rowid')]

    def cached(self, key):
        with self.connect() as conn:
            row = conn.execute('SELECT data FROM requests WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def forget_request(self, key):
        with self.connect() as conn:
            conn.execute('DELETE FROM requests WHERE key=?', (key,))

    def cache(self, key, task, data):
        with self.connect() as conn:
            conn.execute('INSERT OR REPLACE INTO requests VALUES (?,?,?,?)',
                         (key, task, json.dumps(data, ensure_ascii=False), time.time()))

    def call(self, task, status, data=None):
        with self.connect() as conn:
            conn.execute('INSERT INTO calls(task,status,data,created) VALUES (?,?,?,?)',
                         (task, status, json.dumps(data or {}, ensure_ascii=False), time.time()))

    def usage(self, policy):
        with self.connect() as conn:
            calls = conn.execute('SELECT status,data FROM calls').fetchall()
            rows = [json.loads(row[1]) for row in calls if row[0] == 'success']
            if not calls:
                rows = [json.loads(row[0]) for row in conn.execute('SELECT data FROM requests')]
        inputs = sum(row.get('prompt_tokens', 0) for row in rows)
        outputs = sum(row.get('completion_tokens', 0) for row in rows)
        priced = policy.input_price_per_million > 0 or policy.output_price_per_million > 0
        count = len(calls) or len(rows)
        return {'requests': count, 'successful_requests': len(rows), 'failed_requests': sum(row[0] != 'success' for row in calls),
                'prompt_tokens': inputs, 'completion_tokens': outputs,
                'provider_usage': any(row.get('usage_reported') for row in rows),
                'estimated_cost': (inputs * policy.input_price_per_million + outputs * policy.output_price_per_million) / 1e6 if priced else None,
                'warning': count >= policy.warning_requests,
                'usage_note': '服务商未返回usage的请求无法统计tokens；失败请求可能仍产生费用'}


def execute(parent, source, options):
    from backend.pipeline.domain import AnalysisReport
    started = time.monotonic()
    try:
        options.validate()
        parent._require_cloud_ready()
        return _execute(parent, source, options)
    except Exception as exc:
        if options.task_id:
            with parent._repo._tx() as conn:
                conn.execute('UPDATE analysis_runs SET status=?,error=?,updated_ts=? WHERE operation_key=?',
                             ('failed', str(exc), time.time(), options.task_id))
        return AnalysisReport(errors=[str(exc)], elapsed_ms=int((time.monotonic()-started)*1000))


def _execute(parent, source, options):
    """共享已加载的三个本地模型，使用独立仓储/目录执行新版本任务。"""
    from backend.pipeline.domain import _video_lock, AnalysisReport
    from backend.pipeline.factory import _build_scene, _build_chunker, _build_context, _build_downloader, _build_vl, _build_llm
    from backend.pipeline.orchestrator import Orchestrator
    from backend.storage.repository import SqliteRepository

    task_policy = options.policy_snapshot or snapshot(parent._cfg)
    cfg = restore(parent._cfg, task_policy)
    from backend.media.parser import ParseOptions, parse_source
    media = parse_source(source, replace(ParseOptions.from_config(cfg), do_probe=False,
                                         max_file_bytes=None, max_duration_ms=None))
    vid = media.video_id
    with _video_lock(vid):
        previous = parent._repo.find_video(vid)
        if previous and previous.status == 'ready' and not options.force:
            data = parent._repo.get_video_report(vid)
            if data:
                result = AnalysisReport()
                for field in fields(result):
                    if field.name in data and field.name != 'stages':
                        setattr(result, field.name, data[field.name])
                result.profile.setdefault('analysis_version', 'legacy')
                return result
        media = parent._parse_source(source, options)
        key = options.task_id or 'direct-' + uuid4().hex
        row = parent._repo._query_one('SELECT * FROM analysis_runs WHERE operation_key=?', (key,))
        if row:
            run_id, work = row['run_id'], Path(row['work_dir'])
            task_policy = json.loads(row['policy_json'])
            cfg = restore(parent._cfg, task_policy)
            with parent._repo._tx() as conn:
                conn.execute('UPDATE analysis_runs SET status=?,updated_ts=? WHERE run_id=?', ('running', time.time(), run_id))
        else:
            run_id = 'r' + uuid4().hex
            work = cfg.path('data') / 'analysis-runs' / run_id
            disk_check(cfg.path('data'), cfg.integrity.disk_reserve_bytes)
            work.mkdir(parents=True, exist_ok=True)
            with parent._repo._tx() as conn:
                conn.execute('INSERT INTO analysis_runs VALUES (?,?,?,?,?,?,?,?,?,?)',
                             (run_id, vid, key, VERSION, 'running', json.dumps(task_policy),
                              str(work), time.time(), time.time(), ''))
        ledger = RunLedger(work, run_id)
        from backend.analysis.resources import guard
        ledger.write_guard = lambda incoming: guard(cfg, 'results', incoming)
        ledger.put('policy', task_policy)
        staged_cfg = replace(cfg, paths=replace(cfg.paths, sqlite=str(work / 'stage.sqlite3'),
                             videos=str(work / 'source'), frames=str(work / 'frames'),
                             audio=str(work / 'audio'), temp=str(work / 'temp')))
        staged = SqliteRepository.from_config(staged_cfg)
        clone = Orchestrator(repository=staged, vectors=parent._vectors,
                             embedder=parent._embedder, asr=parent._asr, llm=_build_llm(staged_cfg, None),
                             scene_detector=_build_scene(staged_cfg), chunker=_build_chunker(staged_cfg),
                             context_builder=_build_context(staged_cfg), downloader=_build_downloader(staged_cfg),
                             vl=_build_vl(staged_cfg, None), config=staged_cfg)
        clone._downloader.config = replace(clone._downloader.config, verify_quality=True, disk_reserve_bytes=cfg.integrity.disk_reserve_bytes)
        clone._analysis_run = ledger
        clone._memory = parent._memory
        clone._suppress_memory_notify = True
        # 外层已持有同视频锁，内部执行不再次争用不可重入锁。
        from types import SimpleNamespace
        detect = clone._detect_scenes
        def cached_scenes(*args):
            saved = ledger.get('scenes')
            if saved:
                return [SimpleNamespace(**row) for row in saved['rows']], saved['detector'], saved['warnings']
            scenes, detector, warnings = detect(*args)
            ledger.put('scenes', {'rows': [{'start_ms': row.start_ms, 'end_ms': row.end_ms, 'index': i} for i, row in enumerate(scenes)],
                                  'detector': detector, 'warnings': warnings})
            return scenes, detector, warnings
        clone._detect_scenes = cached_scenes
        clone._isolated_run = True
        opts = replace(options, force=False, resume=True)
        try:
            result = clone._analyze(source, opts)
            if result.ok:
                result.profile.update({'analysis_version': task_policy.get('_analysis_strategy', 'integrity-v2'), 'analysis_run_id': run_id,
                                       'usage': ledger.usage(cfg.integrity),
                                       'coverage': ledger.get('coverage') or {},
                                       'visual_coverage': ledger.get('visual_coverage') or {},
                                       'summary_tree': ledger.get('summary_tree') or [],
                                       'report_truncated': bool(ledger.get('report_truncated'))})
                if result.profile['report_truncated']:
                    result.warnings.append('报告输出尚未完成，已达到续写预算；完整章节与原始证据保留')
                if options.skip_summary:
                    result.profile['coverage'] = {'ratio': 0, 'covered_clips': 0, 'total_clips': result.clip_count, 'skipped': True}
                    result.warnings.append('用户选择跳过总结，本次不计为总结完整性验收通过')
                staged.set_video_report(vid, result.to_dict())
                guard(cfg, 'results')
                publish(parent._repo, staged, vid, run_id=run_id)
                with parent._bm25_lock:
                    parent._bm25_cache.pop(vid, None)
                parent._notify_memory()
            with parent._repo._tx() as conn:
                conn.execute('UPDATE analysis_runs SET status=?,updated_ts=?,error=? WHERE run_id=?',
                             ('ready' if result.ok else 'failed', time.time(), '；'.join(result.errors), run_id))
            return result
        finally:
            staged.close()
            if clone._vl is not None:
                clone._vl.close()
            clone._llm.close()


def publish(target, staged, video_id, *, run_id=''):
    """元数据、切片、帧、转写与索引代号在同一事务切换；不删除旧文件/旧索引。"""
    tables = ('videos', 'clips', 'keyframes', 'transcript_segments')
    values = {table: [dict(row) for row in staged._query(f'SELECT * FROM {table} WHERE video_id=?', (video_id,))]
              for table in tables}
    with target._tx() as conn:
        video = values['videos'][0]
        for table, rows in values.items():
            if table != 'videos':
                current = conn.execute(f'SELECT COUNT(*) FROM {table} WHERE video_id=?', (video_id,)).fetchone()[0]
                target._assert_capacity(conn, table, max(0, len(rows) - current))
            elif not conn.execute('SELECT 1 FROM videos WHERE video_id=?', (video_id,)).fetchone():
                target._assert_capacity(conn, 'videos', 1)
        names = list(video)
        updates = ','.join(f'{name}=excluded.{name}' for name in names if name not in {'video_id', 'created_at'})
        conn.execute(f'INSERT INTO videos ({",".join(names)}) VALUES ({",".join("?" for _ in names)}) '
                     f'ON CONFLICT(video_id) DO UPDATE SET {updates}', tuple(video.values()))
        for table in ('keyframes', 'clips', 'transcript_segments'):
            conn.execute(f'DELETE FROM {table} WHERE video_id=?', (video_id,))
        for table in ('clips', 'transcript_segments', 'keyframes'):
            for row in values[table]:
                names = list(row)
                conn.execute(f'INSERT INTO {table} ({",".join(names)}) VALUES ({",".join("?" for _ in names)})', tuple(row.values()))
        if run_id:
            conn.execute("UPDATE analysis_runs SET status='ready',updated_ts=?,error='' WHERE run_id=?", (time.time(),run_id))


def purge_versions(orch, video_id):
    """仅响应用户在视频库的明确删除；验证边界后清理该视频的版本产物。"""
    root = (orch._cfg.path('data')/'analysis-runs').resolve()
    count = 0
    for row in orch._repo._query('SELECT * FROM analysis_runs WHERE video_id=?', (video_id,)):
        directory = Path(row['work_dir']).resolve()
        if directory.parent != root:
            raise ValueError('分析运行目录不在预期范围，已停止清理')
        staged_path = directory/'stage.sqlite3'
        if staged_path.is_file() and orch._vectors:
            conn = sqlite3.connect(f'file:{staged_path.as_posix()}?mode=ro', uri=True)
            try:
                video = conn.execute('SELECT vector_generation FROM videos WHERE video_id=?', (video_id,)).fetchone()
            finally:
                conn.close()
            if video and video[0]:
                orch._vectors.drop_generation(video[0])
        if directory.is_dir():
            shutil.rmtree(directory)
            count += 1
        with orch._repo._tx() as conn:
            conn.execute('DELETE FROM analysis_runs WHERE run_id=?', (row['run_id'],))
    return count
