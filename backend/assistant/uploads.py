"""浏览器原始文件流上传；不将整个视频或 multipart 表单读入内存。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path
from urllib.parse import unquote

from fastapi import HTTPException
from backend.media.ffmpeg_utils import probe_media
from backend.media.parser import DEFAULT_ALLOWED_EXTENSIONS


def cleanup_uploads(service):
    from backend.pipeline.deletion import file_key,file_keys
    from backend.assistant.resumable import clean
    clean(service)
    directory = Path(service.config.path('data')) / 'uploads'
    directory.mkdir(parents=True, exist_ok=True)
    if service.config.storage.retention.temp_hours <= 0:
        return
    cutoff = time.time() - service.config.storage.retention.temp_hours * 3600
    # 已有视频/已提交操作使用的文件永不按临时附件删除。
    referenced = {key for row in service.store.repo._query(
        "SELECT path FROM uploaded_videos WHERE claimed=1 OR created_ts>=?", (cutoff,)
    ) for key in file_keys(row['path'])}
    referenced.update(key for row in service.store.repo._query(
        "SELECT local_path FROM videos WHERE local_path<>''"
    ) for key in file_keys(row['local_path']))
    for row in service.store.repo._query("SELECT payload,conversation_id FROM conversation_operations WHERE status IN ('queued','running')"):
        payload = json.loads(row['payload'])
        if payload.get('source') and '://' not in payload['source']:
            referenced.update(file_keys(payload['source']))
        if payload.get('attachment_id'):
            attachment = service.store.attachment(payload['attachment_id'], row['conversation_id'])
            referenced.update(file_keys(attachment['path']))
    referenced.update(key for row in service.store.repo._query(
        "SELECT path FROM upload_sessions WHERE attachment_id='' ") for key in file_keys(row['path']))
    for path in directory.iterdir():
        if (path.is_file() or path.is_symlink()) and path.lstat().st_mtime < cutoff and file_key(path) not in referenced:
            path.unlink(missing_ok=True)
    with service.store.repo._tx() as conn:
        expired = conn.execute('SELECT id,path FROM uploaded_videos WHERE claimed=0 AND created_ts<?', (cutoff,)).fetchall()
        for row in expired:
            if file_key(row['path']) not in referenced:
                conn.execute('DELETE FROM uploaded_videos WHERE id=?', (row['id'],))


async def receive_upload(request, service, cid):
    service.store.get(cid)
    if service.store.get(cid)['video_id']:
        raise HTTPException(409, '此会话已绑定视频，请新建会话上传')
    if service.store.active(cid):
        raise HTTPException(409, '当前会话正在处理')
    name = Path(unquote(request.headers.get('x-filename', '')).replace('\\', '/')).name
    if not name or len(name) > 255 or any(ord(c) < 32 for c in name):
        raise HTTPException(400, '请提供有效视频文件名')
    suffix = Path(name).suffix.lower()
    if suffix.lstrip('.') not in DEFAULT_ALLOWED_EXTENSIONS:
        raise HTTPException(415, '不支持的视频格式')
    maximum = service.config.limits.max_file_bytes
    try:
        declared_size = int(request.headers.get('content-length', '0'))
    except ValueError as exc:
        raise HTTPException(400, '文件长度无效') from exc
    if declared_size > maximum:
        raise HTTPException(413, '视频超过文件大小限制')
    directory = Path(service.config.path('data')) / 'uploads'
    directory.mkdir(parents=True, exist_ok=True)
    aid = 'u' + uuid.uuid4().hex
    partial = directory / (aid + '.part')
    count = 0
    digest = hashlib.sha256()
    try:
        from backend.analysis.runs import disk_check
        disk_check(directory, service.config.integrity.disk_reserve_bytes, declared_size)
        with partial.open('xb') as handle:
            async for chunk in request.stream():
                count += len(chunk)
                if count > maximum:
                    raise HTTPException(413, '视频超过文件大小限制')
                disk_check(directory, service.config.integrity.disk_reserve_bytes, len(chunk))
                digest.update(chunk)
                await asyncio.to_thread(handle.write, chunk)
        if not count:
            raise HTTPException(400, '视频文件为空')
        info = await asyncio.to_thread(probe_media, partial)
        if not info.has_video or info.width <= 0 or info.height <= 0:
            raise HTTPException(415, '文件没有有效视频画面')
        if info.duration_ms > service.config.limits.max_duration_ms:
            raise HTTPException(413, '视频时长超过限制')
        # 同一内容使用稳定路径，沿用原流水线的来源缓存，不因重复上传重新分析。
        final = directory / ('sha256-' + digest.hexdigest() + suffix)
        if final.exists():
            partial.unlink()
        else:
            partial.replace(final)
        service.store.add_attachment(cid, aid, name, final, count)
    except BaseException as exc:
        partial.unlink(missing_ok=True)
        # 已发布的内容路径可能正被其他会话使用，数据库失败时交给孤儿清理。
        if isinstance(exc, (HTTPException, asyncio.CancelledError)):
            raise
        if isinstance(exc, OSError):
            raise HTTPException(507, f'无法写入视频缓存：{exc}') from exc
        raise HTTPException(415, '视频无法读取，请检查文件是否损坏') from exc
    return {'id': aid, 'filename': name, 'size_bytes': count}
