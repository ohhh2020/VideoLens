"""仅供本机网页的视频助手会话、附件与事件流。"""
from __future__ import annotations

import asyncio
import threading
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic import ValidationError

from backend.assistant.service import AssistantService
from backend.assistant.uploads import cleanup_uploads, receive_upload
from backend.web.dependencies import _orchestrator, _sse, require_linked_delete_ui
from backend.web.schemas import AnalyzeRequest
from backend.pipeline.domain import PipelineError

_initialization = threading.Lock()


def assistant(request):
    with _initialization:
        service = getattr(request.app.state, 'assistant', None)
        if service is None:
            service = AssistantService(_orchestrator(request), request.app.state.config)
            request.app.state.assistant = service
            cleanup_uploads(service)
        service.config = service.orch._cfg if hasattr(service.orch, '_cfg') else service.config
        return service


class NewConversation(BaseModel):
    video_id: str = Field('', max_length=64)


class RenameConversation(BaseModel):
    title: str = Field(min_length=1, max_length=15)


class UploadInit(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)
    fingerprint: str = Field(min_length=64, max_length=64, pattern=r'^[a-f0-9]{64}$')


class UploadComplete(BaseModel):
    sha256: str = Field(min_length=64, max_length=64, pattern=r'^[a-f0-9]{64}$')


class SendMessage(BaseModel):
    text: str = Field('', max_length=8000)
    attachment_id: str = Field('', max_length=64)
    request_id: str = Field(min_length=1, max_length=80, pattern=r'^[a-zA-Z0-9_-]+$')
    options: dict[str, Any] = Field(default_factory=dict)


class LocalVideo(BaseModel):
    path: str = Field(min_length=1,max_length=32768)
    size_bytes: int | None = Field(None,gt=0)
    last_modified: int | None = Field(None,ge=0)


def create_router(cfg=None):
    router = APIRouter()

    @router.post('/local-videos/select')
    def select_local_video(request: Request):
        from backend.media.local_video import select_native_video,describe_local_file
        path=select_native_video()
        if path is None:
            return {'cancelled':True}
        return describe_local_file(path,request.app.state.config)

    @router.post('/local-videos/inspect')
    def inspect_local_video(request: Request,payload:LocalVideo):
        from backend.media.local_video import describe_local_file
        return describe_local_file(payload.path,request.app.state.config)

    @router.post('/conversations/{cid}/local-video',status_code=201)
    def link_local_video(request:Request,cid:str,payload:LocalVideo):
        from backend.media.local_video import attach_local_video
        try:
            return attach_local_video(assistant(request),cid,payload.path,
                expected_size=payload.size_bytes,expected_modified=payload.last_modified)
        except KeyError as exc:
            raise HTTPException(404,str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(415,'本地视频无法读取，请检查原文件是否损坏或被占用') from exc

    @router.get('/videos/{vid}/evidence')
    def video_evidence(request: Request, vid: str, run_id: str = '', offset: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=500)):
        from backend.analysis.runs import RunLedger
        orch = _orchestrator(request)
        if not run_id:
            report = orch.repository.get_video_report(vid) or {}
            run_id = report.get('profile', {}).get('analysis_run_id', '')
        row = orch.repository._query_one('SELECT * FROM analysis_runs WHERE run_id=? AND video_id=? AND status=?', (run_id, vid, 'ready'))
        if not row:
            return {'items': [], 'total': 0, 'version': 'legacy'}
        ledger = RunLedger(row['work_dir'], run_id)
        with ledger.connect() as conn:
            total = conn.execute('SELECT COUNT(*) FROM evidence').fetchone()[0]
            records = conn.execute('SELECT data FROM evidence ORDER BY rowid LIMIT ? OFFSET ?', (limit, offset)).fetchall()
        import json
        return {'items': [json.loads(record[0]) for record in records], 'total': total, 'version': row['version']}

    @router.get('/conversations')
    def list_conversations(request: Request, limit: int = Query(30, ge=1, le=100), offset: int = Query(0, ge=0)):
        return {'items': assistant(request).store.list(limit, offset)}

    @router.post('/conversations', status_code=201)
    def create_conversation(request: Request, payload: NewConversation):
        service = assistant(request)
        title = '新会话'
        if payload.video_id:
            record = service.orch.repository.find_video(payload.video_id)
            if not record:
                raise HTTPException(404, '视频不存在')
            if record.status != 'ready':
                raise HTTPException(409, '视频尚未分析完成，请先完成原任务或在新会话提交视频')
        result = service.store.create(video_id=payload.video_id, title=title)
        if payload.video_id:
            report = service.orch.repository.get_video_report(payload.video_id)
            if report:
                service.store.add_message(result['id'], 'report', report.get('summary', ''), payload=report)
                service.ensure_title(result['id'], report)
        return service.store.get(result['id'])

    @router.get('/conversations/{cid}')
    def get_conversation(request: Request, cid: str):
        try:
            return assistant(request).snapshot(cid)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.patch('/conversations/{cid}')
    def rename_conversation(request: Request, cid: str, payload: RenameConversation):
        service = assistant(request)
        try:
            service.store.get(cid)
            if not payload.title.strip():
                raise HTTPException(422, '标题不能为空')
            service.store.update(cid, title=payload.title.strip())
            return service.store.get(cid)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.delete('/conversations/{cid}')
    def delete_conversation(request: Request, cid: str):
        require_linked_delete_ui(request)
        service = assistant(request)
        try:
            return service.delete_conversation(cid)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (ValueError, PipelineError) as exc:
            raise HTTPException(409, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc

    @router.post('/conversations/{cid}/uploads', status_code=201)
    async def upload(request: Request, cid: str):
        service = assistant(request)
        await asyncio.to_thread(cleanup_uploads, service)
        try:
            return await receive_upload(request, service, cid)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.post('/conversations/{cid}/uploads/init', status_code=201)
    def upload_init(request: Request, cid: str, payload: UploadInit):
        from backend.assistant.resumable import initialize, clean
        service = assistant(request)
        clean(service)
        try:
            return initialize(service, cid, payload.filename, payload.size_bytes, payload.fingerprint)
        except OSError as exc:
            raise HTTPException(507, f'无法写入上传缓存：{exc}') from exc

    @router.get('/conversations/{cid}/uploads/{uid}')
    def upload_state(request: Request, cid: str, uid: str):
        from backend.assistant.resumable import state, public
        return public(state(assistant(request), cid, uid))

    @router.put('/conversations/{cid}/uploads/{uid}/chunks/{index}')
    async def upload_chunk(request: Request, cid: str, uid: str, index: int):
        from backend.assistant.resumable import state, write_chunk
        service = assistant(request)
        row = state(service, cid, uid)
        data = bytearray()
        async for part in request.stream():
            data.extend(part)
            if len(data) > row['chunk_bytes']:
                raise HTTPException(413, '单个分块过大')
        try:
            return await asyncio.to_thread(write_chunk, service, cid, uid, index, bytes(data), request.headers.get('x-chunk-sha256', ''))
        except OSError as exc:
            raise HTTPException(507, f'分块写入失败：{exc}') from exc

    @router.post('/conversations/{cid}/uploads/{uid}/complete')
    async def upload_complete(request: Request, cid: str, uid: str, payload: UploadComplete):
        from backend.assistant.resumable import finalize
        try:
            return await asyncio.to_thread(finalize, assistant(request), cid, uid, payload.sha256)
        except OSError as exc:
            raise HTTPException(507, f'上传合并或写入失败：{exc}') from exc

    @router.post('/conversations/{cid}/messages', status_code=202)
    def message(request: Request, cid: str, payload: SendMessage):
        if not payload.text.strip() and not payload.attachment_id:
            raise HTTPException(422, '请填写消息或添加视频附件')
        # 复用原分析参数的白名单与范围校验，不能由消息传入任意内部参数。
        try:
            if 'source' in payload.options:
                raise HTTPException(422, '来源只能通过附件或消息选择')
            options = AnalyzeRequest(source='placeholder', **payload.options).model_dump(exclude={'source'})
        except ValidationError as exc:
            raise HTTPException(422, '分析选项无效') from exc
        try:
            operation = assistant(request).submit(cid, payload.request_id, payload.text.strip(), payload.attachment_id, options)
            return {'operation_id': operation['id'], 'status': operation['status']}
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @router.post('/conversations/{cid}/operations/{oid}/retry', status_code=202)
    def retry(request: Request, cid: str, oid: str):
        try:
            operation = assistant(request).retry(cid, oid)
            return {'operation_id': operation['id'], 'status': operation['status']}
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get('/conversations/{cid}/events')
    async def events(request: Request, cid: str):
        service = assistant(request)
        queue = asyncio.Queue(maxsize=128)
        try:
            service.subscribe(cid, asyncio.get_running_loop(), queue)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

        async def stream():
            try:
                snapshot = service.snapshot(cid)
                yield _sse({'type': 'snapshot', **snapshot})
                if not snapshot['operation'] or snapshot['operation']['status'] not in {'queued', 'running'}:
                    return
                while not await request.is_disconnected():
                    try:
                        event = await asyncio.wait_for(queue.get(), service.config.api.sse_heartbeat_s)
                        yield _sse(event)
                        if event['type'] == 'snapshot' and (event.get('operation') or {}).get('status') not in {'queued', 'running'}:
                            return
                    except asyncio.TimeoutError:
                        yield ': heartbeat\n\n'
            finally:
                service.unsubscribe(cid, queue)
        return StreamingResponse(stream(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

    return router
