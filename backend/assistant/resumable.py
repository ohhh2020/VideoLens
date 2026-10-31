"""网页8MiB分块上传：幂等、断点持久化、完整校验，不整文件读入内存。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.parse import unquote
from uuid import uuid4

from fastapi import HTTPException
from backend.analysis.runs import disk_check
from backend.media.ffmpeg_utils import probe_media
from backend.media.parser import DEFAULT_ALLOWED_EXTENSIONS

_locks = {}
_guard = threading.Lock()


def lock(upload_id):
    with _guard:
        return _locks.setdefault(upload_id, threading.RLock())


def state(service, cid, uid):
    row = service.store.repo._query_one('SELECT * FROM upload_sessions WHERE id=? AND conversation_id=?', (uid, cid))
    if not row:
        raise HTTPException(404, '上传会话不存在或不属于当前对话')
    return dict(row)


def public(row):
    return {'upload_id': row['id'], 'filename': row['filename'], 'size_bytes': row['size_bytes'],
            'chunk_bytes': row['chunk_bytes'], 'received': sorted(int(key) for key in json.loads(row['chunks'])),
            'attachment_id': row['attachment_id'], 'updated_ts': row['updated_ts']}


def initialize(service, cid, filename, size, fingerprint):
    try:
        service.store.get(cid)
    except KeyError as exc:
        raise HTTPException(404, '对话不存在') from exc
    if service.store.get(cid)['video_id'] or service.store.active(cid):
        raise HTTPException(409, '当前会话已绑定视频或正在处理')
    name = Path(unquote(filename).replace('\\', '/')).name
    if not name or len(name) > 255 or any(ord(char) < 32 for char in name):
        raise HTTPException(400, '文件名无效')
    if Path(name).suffix.lower().lstrip('.') not in DEFAULT_ALLOWED_EXTENSIONS:
        raise HTTPException(415, '不支持的视频格式')
    if not 0 < size <= service.config.limits.max_file_bytes:
        raise HTTPException(413, '视频超过文件大小限制或为空')
    cutoff = time.time() - service.config.integrity.upload_ttl_hours * 3600
    old = service.store.repo._query_one('SELECT * FROM upload_sessions WHERE conversation_id=? AND filename=? '
              'AND size_bytes=? AND fingerprint=? AND updated_ts>=? ORDER BY updated_ts DESC LIMIT 1',
              (cid, name, size, fingerprint, cutoff))
    if old:
        return public(dict(old))
    directory = service.config.path('data') / 'uploads'
    directory.mkdir(parents=True, exist_ok=True)
    disk_check(directory, service.config.integrity.disk_reserve_bytes)
    uid = 'up' + uuid4().hex
    path = directory / (uid + '.part')
    path.touch(exist_ok=False)
    with service.store.repo._tx() as conn:
        conn.execute('INSERT INTO upload_sessions VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                     (uid, cid, name, size, fingerprint, service.config.integrity.upload_chunk_bytes,
                      str(path), '{}', time.time(), time.time(), ''))
    return public(state(service, cid, uid))


def write_chunk(service, cid, uid, index, data, digest):
    with lock(uid):
        row = state(service, cid, uid)
        if row['attachment_id']:
            return public(row)
        chunk = row['chunk_bytes']
        count = (row['size_bytes'] + chunk - 1) // chunk
        expected = min(chunk, row['size_bytes'] - index * chunk)
        if index < 0 or index >= count or len(data) != expected:
            raise HTTPException(400, '分块序号或长度无效')
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest.lower():
            raise HTTPException(422, '分块哈希不匹配，请重传此块')
        chunks = json.loads(row['chunks'])
        if str(index) in chunks:
            if chunks[str(index)] != actual:
                raise HTTPException(409, '相同序号的分块内容冲突')
            return public(row)
        growth = max(0, index * chunk + len(data) - Path(row['path']).stat().st_size)
        disk_check(Path(row['path']).parent, service.config.integrity.disk_reserve_bytes, growth)
        with Path(row['path']).open('r+b') as handle:
            handle.seek(index * chunk)
            handle.write(data)
            handle.flush()
        chunks[str(index)] = actual
        with service.store.repo._tx() as conn:
            conn.execute('UPDATE upload_sessions SET chunks=?,updated_ts=? WHERE id=?',
                         (json.dumps(chunks), time.time(), uid))
        return public(state(service, cid, uid))


def finalize(service, cid, uid, digest):
    with lock(uid):
        row = state(service, cid, uid)
        if row['attachment_id']:
            attachment = service.store.attachment(row['attachment_id'], cid)
            return {key: attachment[key] for key in ('id', 'filename', 'size_bytes')}
        count = (row['size_bytes'] + row['chunk_bytes'] - 1) // row['chunk_bytes']
        if len(json.loads(row['chunks'])) != count:
            raise HTTPException(409, '仍有未完成的分块')
        if service.store.get(cid)['video_id'] or service.store.active(cid):
            raise HTTPException(409, '当前对话已绑定视频或正在处理')
        if digest.lower() != row['fingerprint']:
            raise HTTPException(422, '完整文件哈希与初始化指纹不一致，请重选同一文件')
        path = Path(row['path'])
        sha = hashlib.sha256()
        published = path.parent / ('sha256-' + digest.lower() + Path(row['filename']).suffix.lower())
        source = path if path.exists() else published
        with source.open('rb') as handle:
            for block in iter(lambda: handle.read(1024**2), b''):
                sha.update(block)
        actual = sha.hexdigest()
        if actual != digest.lower():
            raise HTTPException(422, '完整文件哈希不匹配，请重新选择同一文件')
        info = probe_media(source)
        if not info.has_video or not info.width or not info.height:
            raise HTTPException(415, '文件没有有效视频画面')
        if info.duration_ms > service.config.limits.max_duration_ms:
            raise HTTPException(413, '视频时长超过2小时限制')
        destination = path.parent / ('sha256-' + actual + Path(row['filename']).suffix.lower())
        if destination.exists():
            path.unlink(missing_ok=True)
        else:
            path.replace(destination)
        aid = 'u' + uuid4().hex
        # 文件已发布后两个登记放在一个短事务中，完成接口可安全重复提交。
        with service.store.repo._tx() as conn:
            conn.execute('INSERT INTO uploaded_videos VALUES (?,?,?,?,?,?,0)',
                         (aid, cid, row['filename'], str(destination), row['size_bytes'], time.time()))
            conn.execute('UPDATE upload_sessions SET attachment_id=?,updated_ts=? WHERE id=?', (aid, time.time(), uid))
        return {'id': aid, 'filename': row['filename'], 'size_bytes': row['size_bytes']}


def clean(service):
    cutoff = time.time() - service.config.integrity.upload_ttl_hours * 3600
    for item in service.store.repo._query('SELECT * FROM upload_sessions WHERE updated_ts<? AND attachment_id=?', (cutoff, '')):
        with lock(item['id']):
            row = state(service, item['conversation_id'], item['id'])
            if row['updated_ts'] >= cutoff:
                continue
            path = Path(row['path']).resolve()
            root = (service.config.path('data') / 'uploads').resolve()
            if path.parent == root and path.suffix == '.part':
                path.unlink(missing_ok=True)
            with service.store.repo._tx() as conn:
                conn.execute('DELETE FROM upload_sessions WHERE id=?', (row['id'],))
