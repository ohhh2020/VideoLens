"""网页内部请求和响应结构。"""

from __future__ import annotations


from typing import Literal


from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    """``POST /videos/analyze`` 的入参。"""

    model_config = {"extra": "forbid"}

    source: str = Field(
        ...,
        min_length=1,
        max_length=32_768,
        description="本地路径或视频 URL；URL ≤ limits.max_url_length（默认 2048）字符，本地路径 ≤32768",
        examples=["D:/videos/demo.mp4", "https://www.bilibili.com/video/BV1xx"],
    )
    title: str = Field("", max_length=200, description="自定义标题，留空则自动")
    language: str | None = Field(
        None, max_length=16, description="强制语言，如 zh / en"
    )
    skip_vision: bool = Field(False, description="跳过视觉理解（只看语音）")
    skip_summary: bool = Field(False, description="跳过整片总结")
    skip_vectors: bool = Field(False, description="跳过向量写入（只留 SQLite）")
    force: bool = Field(False, description="强制重跑，忽略已有结果")
    resume: bool = Field(True, description="断点续跑（跳过已完成阶段）")
    max_keyframes: int | None = Field(None, ge=1, le=5000)

    @field_validator("source")
    @classmethod
    def _strip(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("source 不能为空")
        return text


class AskRequest(BaseModel):
    """``POST /videos/{video_id}/ask`` 的入参。"""

    model_config = {"extra": "forbid"}

    question: str = Field(
        ..., min_length=1, max_length=500, description="问题（≤500 字符）"
    )
    top_k: int | None = Field(None, ge=1, le=50, description="检索条数，默认取配置")
    max_context_tokens: int | None = Field(None, ge=512, le=32_000)
    session_id: str = Field(
        "", max_length=64, description="会话分组（/chat 页每段对话一个；留空归 legacy）"
    )

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("question 不能为空")
        return text


class CitationOut(BaseModel):
    index: int
    clip_id: str
    video_id: str
    start_ms: int
    end_ms: int
    time_range: str
    keyframe_paths: list[str] = Field(default_factory=list)
    text: str = ""


class AnswerOut(BaseModel):
    question: str
    answer: str
    video_id: str
    refs: list[CitationOut] = Field(default_factory=list)
    backend: str = ""
    degraded: bool = False
    elapsed_ms: int = 0
    context_chars: int = 0
    candidates: int = 0
    notes: list[str] = Field(default_factory=list)


class StageOut(BaseModel):
    stage: str
    ok: bool
    skipped: bool = False
    degraded: bool = False
    message: str = ""
    error: str = ""
    elapsed_ms: int = 0


class AnalyzeOut(BaseModel):
    video_id: str
    title: str = ""
    status: str
    ok: bool = False
    duration_ms: int = 0
    clip_count: int = 0
    keyframe_count: int = 0
    segment_count: int = 0
    vector_count: int = 0
    transcript_source: str = ""
    language: str = ""
    summary: str = ""
    summary_backend: str = ""
    degraded: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    elapsed_ms: int = 0
    stages: list[StageOut] = Field(default_factory=list)


class VideoOut(BaseModel):
    video_id: str
    display_name: str = ""
    title: str = ""
    source_kind: str = ""
    source_url: str = ""
    duration_ms: int = 0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    language: str = ""
    status: str = ""
    stage: str = ""
    progress: float = 0.0
    error: str = ""
    created_at: str = ""
    updated_at: str = ""


class ClipOut(BaseModel):
    clip_id: str
    video_id: str
    index: int
    start_ms: int
    end_ms: int
    core_start_ms: int
    core_end_ms: int
    time_range: str
    text: str
    core_text: str
    keyframe_ids: list[str] = Field(default_factory=list)


class KeyframeOut(BaseModel):
    keyframe_id: str
    video_id: str
    clip_id: str | None = None
    scene_index: int = 0
    timestamp_ms: int
    timecode: str
    path: str
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    sharpness: float = 0.0
    description: str = ""
    ocr_text: str = ""


class VideoDetailOut(BaseModel):
    video: VideoOut
    clips: list[ClipOut] = Field(default_factory=list)
    keyframes: list[KeyframeOut] = Field(default_factory=list)
    clip_count: int = 0
    keyframe_count: int = 0


class JobOut(BaseModel):
    task_id: str
    video_id: str = ""
    status: Literal["queued", "running", "done", "failed", "cancelled"] = "queued"
    stage: str = ""
    progress: float = 0.0
    message: str = ""
    error: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    elapsed_ms: int = 0
    report: AnalyzeOut | None = None


class HealthOut(BaseModel):
    pid: int = 0
    model_devices: dict[str, str] = Field(default_factory=dict)
    status: str = "ok"
    version: str = ""
    project_root: str = ""
    orchestrator: bool = False
    device: str = ""
    auth: str = "loopback-only"
    jobs_running: int = 0
    jobs_total: int = 0


class StatsOut(BaseModel):
    videos: int = 0
    clips: int = 0
    transcript_segments: int = 0
    keyframes: int = 0
    tasks_pending: int = 0
    tasks_running: int = 0
    keyframe_bytes: int = 0
    db_bytes: int = 0
    usage: dict[str, float] = Field(default_factory=dict)


class DeleteOut(BaseModel):
    video_id: str
    db: list[str] = Field(default_factory=list)
    vectors: dict[str, int] = Field(default_factory=dict)
    deleted: bool = True
