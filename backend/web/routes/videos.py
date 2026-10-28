"""网页内部 videos 路由。"""

from __future__ import annotations

import os

from pathlib import Path

from typing import Any, Literal

from urllib.parse import quote

from fastapi import APIRouter, Body, HTTPException, Query, Request, Response, status

from fastapi.responses import (
    FileResponse,
    PlainTextResponse,
)


from backend.pipeline.orchestrator import (
    AnalysisReport,
    AnalyzeOptions,
    PipelineError,
)

from backend.storage.repository import NotFoundError, StorageError

from backend.web.jobs import get_job_store

from backend.web.schemas import (
    AnalyzeRequest,
    AnalyzeOut,
    VideoOut,
    ClipOut,
    KeyframeOut,
    VideoDetailOut,
    JobOut,
    DeleteOut,
)
from backend.web.dependencies import (
    _orchestrator,
    _video_out,
    _clip_out,
    _keyframe_out,
    _build_srt,
    _build_vtt,
    _subtitle_filename,
    _analyze_out,
)


def create_router(cfg=None):
    router = APIRouter()

    def _cfg(request):
        return cfg or request.app.state.config

    @router.post(
        "/videos/analyze",
        response_model=JobOut,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["videos"],
        summary="提交一个视频分析任务（异步）",
    )
    def analyze(request: Request, payload: AnalyzeRequest = Body(...)) -> JobOut:
        orch = _orchestrator(request)
        conf = _cfg(request)
        orch._require_cloud_ready()

        # 入参前置校验：URL 长度上限、文件大小上限由 core 层再校验一次，
        # 这里先做"明显不合理"的拦截，避免无意义的排队。
        source = payload.source
        if len(source) > conf.limits.max_url_length and "://" in source:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"URL 超过 {conf.limits.max_url_length} 字符",
            )

        try:
            job = get_job_store().submit(
                orch,
                source,
                AnalyzeOptions(
                    language=payload.language,
                    skip_vision=payload.skip_vision,
                    skip_summary=payload.skip_summary,
                    skip_vectors=payload.skip_vectors,
                    force=payload.force,
                    resume=payload.resume,
                    max_keyframes=payload.max_keyframes,
                    title=payload.title,
                ),
            )
        except RuntimeError as exc:  # 线程池已关停
            raise HTTPException(status_code=503, detail="服务正在关闭") from exc
        return JobOut.model_validate(job.snapshot())

    @router.get(
        "/videos",
        response_model=list[VideoOut],
        tags=["videos"],
        summary="视频列表",
    )
    def list_videos(
        request: Request,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        status_filter: str | None = Query(None, alias="status"),
    ) -> list[VideoOut]:
        orch = _orchestrator(request)
        try:
            records = orch.repository.list_videos(
                limit=limit, offset=offset, status=status_filter
            )
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=f"读取列表失败: {exc}") from exc
        return [_video_out(item) for item in records]

    @router.get(
        "/videos/{video_id}",
        response_model=VideoDetailOut,
        tags=["videos"],
        summary="视频详情（含切片与关键帧）",
    )
    def video_detail(request: Request, video_id: str) -> VideoDetailOut:
        orch = _orchestrator(request)
        try:
            record = orch.repository.find_video(video_id)
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if record is None:
            raise HTTPException(status_code=404, detail=f"视频不存在: {video_id}")
        clips = [_clip_out(c) for c in orch.repository.get_clips(video_id)]
        frames = [_keyframe_out(k) for k in orch.repository.get_keyframes(video_id)]
        return VideoDetailOut(
            video=_video_out(record),
            clips=clips,
            keyframes=frames,
            clip_count=len(clips),
            keyframe_count=len(frames),
        )

    @router.get(
        "/videos/{video_id}/clips",
        response_model=list[ClipOut],
        tags=["videos"],
        summary="切片列表",
    )
    def video_clips(request: Request, video_id: str) -> list[ClipOut]:
        orch = _orchestrator(request)
        return [_clip_out(c) for c in orch.repository.get_clips(video_id)]

    @router.get(
        "/videos/{video_id}/subtitles",
        tags=["videos"],
        summary="导出字幕文件（format=srt|vtt，默认 srt）",
    )
    def video_subtitles(
        request: Request,
        video_id: str,
        fmt: Literal["srt", "vtt"] = Query("srt", alias="format"),
    ) -> Response:
        orch = _orchestrator(request)
        try:
            record = orch.repository.find_video(video_id)
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if record is None:
            raise HTTPException(status_code=404, detail=f"视频不存在: {video_id}")
        segments = orch.repository.get_segments(video_id)
        if not segments:
            raise HTTPException(status_code=404, detail="该视频没有可导出的转写片段")
        text = _build_srt(segments) if fmt == "srt" else _build_vtt(segments)
        filename = _subtitle_filename(
            str(getattr(record, "title", "") or ""), video_id, fmt
        )
        quoted = quote(filename)
        return Response(
            content=text,
            media_type="application/x-subrip"
            if fmt == "srt"
            else "text/vtt; charset=utf-8",
            headers={"Content-Disposition": (f"attachment; filename*=UTF-8''{quoted}")},
        )

    @router.post("/videos/{video_id}/open-cache", summary="在本机打开该云端视频的缓存目录")
    def open_cache(request: Request, video_id: str):
        record = _orchestrator(request).repository.find_video(video_id)
        if record is None:
            raise HTTPException(404,"视频不存在")
        if record.source_kind != "url":
            raise HTTPException(409,"本地视频没有云端下载缓存")
        value = str(record.local_path or "").strip()
        if not value:
            raise HTTPException(409,"该视频尚未下载到本地")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise HTTPException(404,"本地缓存文件已被清理或移动")
        if os.name != "nt":
            raise HTTPException(501,"当前系统不支持从网页打开本地文件夹")
        try:
            # 路径只取自视频记录；始终打开目录，不执行视频或客户端传入的路径。
            os.startfile(str(path.parent))
        except OSError as exc:
            raise HTTPException(500,"无法打开本地缓存文件夹") from exc
        return {"opened":True,"directory":str(path.parent)}

    @router.get(
        "/videos/{video_id}/transcript",
        tags=["videos"],
        summary="转写片段列表（JSON，供界面展示转写结果）",
    )
    def video_transcript(request: Request, video_id: str) -> dict[str, Any]:
        orch = _orchestrator(request)
        try:
            record = orch.repository.find_video(video_id)
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if record is None:
            raise HTTPException(status_code=404, detail=f"视频不存在: {video_id}")
        segments = orch.repository.get_segments(video_id)
        return {
            "video_id": video_id,
            "language": str(getattr(record, "language", "") or ""),
            "count": len(segments),
            "segments": [
                {
                    "index": s.index_no,
                    "start_ms": s.start_ms,
                    "end_ms": s.end_ms,
                    "text": s.text,
                }
                for s in segments
            ],
        }

    @router.get(
        "/videos/{video_id}/report",
        tags=["videos"],
        summary="上次分析的持久化报告（重启后仍可查）",
    )
    def video_report(request: Request, video_id: str) -> dict[str, Any]:
        orch = _orchestrator(request)
        try:
            report = orch.repository.get_video_report(video_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if report is None:
            raise HTTPException(status_code=404, detail="该视频还没有已保存的分析报告")
        return report

    @router.get(
        "/videos/{video_id}/keyframes",
        response_model=list[KeyframeOut],
        tags=["videos"],
        summary="关键帧列表",
    )
    def video_keyframes(request: Request, video_id: str) -> list[KeyframeOut]:
        orch = _orchestrator(request)
        return [_keyframe_out(k) for k in orch.repository.get_keyframes(video_id)]

    @router.get(
        "/videos/{video_id}/keyframes/{keyframe_id}/image",
        tags=["videos"],
        summary="取关键帧图片",
    )
    def keyframe_image(
        request: Request, video_id: str, keyframe_id: str
    ) -> FileResponse:
        """按 ID 取图，**不接受客户端路径**，因此不存在路径穿越面。"""
        orch = _orchestrator(request)
        try:
            record = orch.repository.get_keyframe(keyframe_id)
        except NotFoundError as exc:
            raise HTTPException(
                status_code=404, detail=f"关键帧不存在: {keyframe_id}"
            ) from exc
        if record is None or record.video_id != video_id:
            raise HTTPException(status_code=404, detail=f"关键帧不存在: {keyframe_id}")
        path = Path(record.path)
        if not path.is_file():
            raise HTTPException(status_code=410, detail="关键帧文件已被清理")
        return FileResponse(
            path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @router.delete(
        "/videos/{video_id}",
        response_model=DeleteOut,
        tags=["videos"],
        summary="删除视频及其切片、关键帧、向量",
    )
    def delete_video(request: Request, video_id: str) -> DeleteOut:
        orch = _orchestrator(request)
        try:
            removed = orch.delete(video_id)
        except PipelineError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if removed.get("db_error") and "不存在" in str(removed["db_error"]):
            raise HTTPException(status_code=404, detail=f"视频不存在: {video_id}")
        return DeleteOut(
            video_id=video_id,
            db=list(removed.get("db") or []),
            vectors=dict(removed.get("vectors") or {}),
        )

    @router.post(
        "/videos/{video_id}/summary",
        response_model=AnalyzeOut,
        tags=["videos"],
        summary="重新生成整片总结",
    )
    def resummarize(request: Request, video_id: str) -> AnalyzeOut:
        orch = _orchestrator(request)
        try:
            text = orch.summarize(video_id)
        except PipelineError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        record = orch.repository.find_video(video_id)
        report = AnalysisReport(
            video_id=video_id,
            title=(record.title or record.display_name) if record else "",
            status="ready",
            summary=text,
            clip_count=len(orch.repository.get_clips(video_id)),
            keyframe_count=len(orch.repository.get_keyframes(video_id)),
        )
        return _analyze_out(report)

    @router.get(
        "/videos/{video_id}/chapters",
        tags=["videos"],
        summary="章节化摘要（长视频按时间切片逐章总结）",
    )
    def list_chapters(
        request: Request,
        video_id: str,
        chapter_seconds: int = 300,
        max_chapters: int = 40,
        min_seconds: int = 300,
        summarize: bool = True,
    ) -> dict[str, Any]:
        orch = _orchestrator(request)
        chapter_seconds = max(30, min(int(chapter_seconds), 3600))
        max_chapters = max(1, min(int(max_chapters), 200))
        try:
            chapters = orch.chapters(
                video_id,
                chapter_seconds=chapter_seconds,
                max_chapters=max_chapters,
                min_seconds=max(0, int(min_seconds)),
                summarize=bool(summarize),
            )
        except PipelineError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "video_id": video_id,
            "chapter_seconds": chapter_seconds,
            # 空列表意味着视频太短，没必要拆章 —— 客户端据此退回单段总结。
            "needs_chapters": bool(chapters),
            "count": len(chapters),
            "chapters": [c.to_dict() for c in chapters],
        }

    @router.get(
        "/videos/{video_id}/chapters/report",
        response_class=PlainTextResponse,
        tags=["videos"],
        summary="章节化 markdown 报告",
    )
    def chapter_report(
        request: Request,
        video_id: str,
        chapter_seconds: int = 300,
        min_seconds: int = 300,
    ) -> PlainTextResponse:
        orch = _orchestrator(request)
        try:
            text = orch.chapter_report(
                video_id,
                chapter_seconds=max(30, min(int(chapter_seconds), 3600)),
                min_seconds=max(0, int(min_seconds)),
            )
        except PipelineError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")

    return router
