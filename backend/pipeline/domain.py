"""分析选项、报告、引用、阶段状态与进程共享锁。"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Final, Sequence

from backend.config import AppConfig

#: 普通 8G 机器的余量；两路任务并行 2 份权重只会换来换页假死，不会更快。
#: 用可重入锁，同一次分析里嵌套获取不会自己锁死自己。
_INFERENCE_LOCK: Final[threading.RLock] = threading.RLock()
_INFERENCE_POLL_S: Final[float] = 0.2

#: 同一 video_id 的分析互斥（类级注册表：设置页重建编排器后依然生效）。
#: 没有它，用户在界面上把同一个视频点两次，两条完整流水线会并行互踩 ——
#: 全局推理锁让它们互相排队（实测一个 5 分钟视频被拖成 20+ 分钟），
#: 转写/抽帧/视觉理解全部算两遍。串行化之后，后到者等先到者完成，
#: 再走"已分析完成/断点恢复"的快路径，代价接近零。
_VIDEO_LOCKS: Final[dict[str, threading.Lock]] = {}
_VIDEO_LOCKS_GUARD = threading.Lock()


def _video_lock(video_id: str) -> threading.Lock:
    with _VIDEO_LOCKS_GUARD:
        lock = _VIDEO_LOCKS.get(video_id)
        if lock is None:
            lock = threading.Lock()
            _VIDEO_LOCKS[video_id] = lock
        return lock


def compute_fingerprint(config: AppConfig) -> str:
    """一次分析"用了什么模型、按什么规格切"的指纹。

    只要它变了，旧切片 / 旧向量 / 旧摘要就不能和新结果混用 —— 混用不会抛异常，
    只会让检索出现"同一条证据两个来源、分数不可比"，答案看着通顺、引用是错的。
    """
    media = config.media
    parts = [
        "vision_output=description-ocr-v2",
        f"asr={config.asr.model}",
        f"text={config.embedding.text_model}",
        f"image={config.embedding.image_model}",
        f"vl={config.vl.cloud_model}",
        f"chunk={config.chunk.min_ms}-{config.chunk.max_ms}-{config.chunk.overlap_ms}",
        f"scene={media.scene_detect.detector}-{media.scene_detect.threshold}",
        f"frames={media.scene_detect.frames_per_scene}",
        f"kf={media.scene_detect.keyframe.min_short_side}-{media.scene_detect.keyframe.max_jpeg_bytes}",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


# =============================================================================
# 异常与阶段
# =============================================================================


class PipelineError(RuntimeError):
    """不可降级的管道故障（拿不到视频本体、数据库不可用）。"""


class Stage(str, Enum):
    """管道阶段（字符串值与 ``VideoStatus`` / 日志字段对齐）。"""

    PARSE = "parse"
    PROBE = "probe"
    AUDIO = "audio"
    TRANSCRIBE = "transcribe"
    SCENE = "scene"
    CHUNK = "chunk"
    VISION = "vision"
    INDEX = "index"
    SUMMARY = "summary"
    DONE = "done"

    @property
    def order(self) -> int:
        return _STAGE_ORDER[self]


_STAGE_ORDER: Final[dict[Stage, int]] = {
    Stage.PARSE: 0,
    Stage.PROBE: 1,
    Stage.AUDIO: 2,
    Stage.TRANSCRIBE: 3,
    Stage.SCENE: 4,
    Stage.CHUNK: 5,
    Stage.VISION: 6,
    Stage.INDEX: 7,
    Stage.SUMMARY: 8,
    Stage.DONE: 9,
}

#: 阶段 → 对外的 video.status（供 Web 端显示进度条配色）
_STAGE_STATUS: Final[dict[Stage, str]] = {
    Stage.PARSE: "downloading",
    Stage.PROBE: "probing",
    Stage.AUDIO: "transcribing",
    Stage.TRANSCRIBE: "transcribing",
    Stage.SCENE: "detecting",
    Stage.CHUNK: "detecting",
    Stage.VISION: "detecting",
    Stage.INDEX: "indexing",
    Stage.SUMMARY: "indexing",
    Stage.DONE: "ready",
}


# =============================================================================
# 选项与结果
# =============================================================================


@dataclass(slots=True)
class AnalyzeOptions:
    """一次分析的开关。"""

    language: str | None = None
    skip_vision: bool = False
    skip_summary: bool = False
    skip_vectors: bool = False
    force: bool = False
    resume: bool = True
    max_keyframes: int | None = None
    title: str = ""
    progress: Callable[[str, float, str], None] | None = None
    #: 外部任务号（backend.web.jobs.JobStore 发的 ``t<uuid>``）。传入时 ``_start_task``
    #: 用它写 SQLite tasks 表，内存进度流与持久化恢复记录共用同一个 ID。
    task_id: str = ""

    def validate(self) -> None:
        if self.max_keyframes is not None and self.max_keyframes < 1:
            raise ValueError(f"max_keyframes 必须 >= 1: {self.max_keyframes}")


@dataclass(slots=True)
class StageReport:
    stage: Stage
    ok: bool = True
    elapsed_ms: int = 0
    skipped: bool = False
    degraded: bool = False
    message: str = ""
    error: str = ""
    # 阶段起点（monotonic），由 _begin 写入、_finish 用来算 elapsed_ms。
    started_at: float = field(default=0.0, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "ok": self.ok,
            "skipped": self.skipped,
            "degraded": self.degraded,
            "elapsed_ms": self.elapsed_ms,
            "message": self.message,
            "error": self.error,
        }


@dataclass(slots=True)
class Citation:
    """一条引用（写进回答的 ``[n]`` 与播放器跳转用的毫秒区间）。"""

    index: int
    clip_id: str
    video_id: str
    start_ms: int
    end_ms: int
    time_range: str
    keyframe_paths: tuple[str, ...] = ()
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "clip_id": self.clip_id,
            "video_id": self.video_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "time_range": self.time_range,
            "keyframe_paths": list(self.keyframe_paths),
            "text": self.text,
        }


@dataclass(slots=True)
class AnalysisReport:
    """一次分析的完整结果（可直接 ``to_dict()`` 给 API 返回）。"""

    video_id: str = ""
    title: str = ""
    status: str = "failed"
    stages: list[StageReport] = field(default_factory=list)
    clip_count: int = 0
    keyframe_count: int = 0
    segment_count: int = 0
    vector_count: int = 0
    transcript_source: str = ""
    language: str = ""
    duration_ms: int = 0
    summary: str = ""
    summary_backend: str = ""
    #: 内容画像（方案 A）：speech_ratio / scene_density / high_motion /
    #: vision_mode_used —— 前端可见"这段视频按什么策略分析"。
    profile: dict[str, Any] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "ready"

    def stage(self, name: Stage) -> StageReport | None:
        for item in self.stages:
            if item.stage is name:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "status": self.status,
            "ok": self.ok,
            "duration_ms": self.duration_ms,
            "clip_count": self.clip_count,
            "keyframe_count": self.keyframe_count,
            "segment_count": self.segment_count,
            "vector_count": self.vector_count,
            "transcript_source": self.transcript_source,
            "language": self.language,
            "profile": dict(self.profile),
            "summary": self.summary,
            "summary_backend": self.summary_backend,
            "degraded": list(self.degraded),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "elapsed_ms": self.elapsed_ms,
            "stages": [s.to_dict() for s in self.stages],
        }


@dataclass(slots=True)
class AskResult:
    """一次问答结果。``refs`` 是按 ``[n]`` 排好序的引用列表。"""

    question: str = ""
    answer: str = ""
    refs: list[Citation] = field(default_factory=list)
    video_id: str = ""
    elapsed_ms: int = 0
    backend: str = ""
    degraded: bool = False
    notes: list[str] = field(default_factory=list)
    context_chars: int = 0
    candidates: int = 0
    #: 本轮归属的会话（/chat 页每段对话一个 id；空=未分组的历史）
    session_id: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.answer.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "video_id": self.video_id,
            "refs": [r.to_dict() for r in self.refs],
            "elapsed_ms": self.elapsed_ms,
            "backend": self.backend,
            "degraded": self.degraded,
            "notes": list(self.notes),
            "context_chars": self.context_chars,
            "candidates": self.candidates,
            "session_id": self.session_id,
        }


# =============================================================================
# 引用解析
# =============================================================================

_CITE_RE: Final[re.Pattern[str]] = re.compile(r"\[(\d{1,3})\]")


def extract_citations(answer: str, refs: Sequence[dict[str, Any]]) -> list[Citation]:
    """从回答里找出 ``[n]`` 并按出现顺序返回对应引用。

    模型有时会写 ``[1][3]``、有时 ``[1, 3]``。这里只认紧邻的单个编号；
    异常编号（超出范围）忽略，不编造引用。
    """
    by_index = {int(r.get("index", 0)): r for r in refs}
    ordered: list[int] = []
    seen: set[int] = set()

    for match in _CITE_RE.finditer(answer or ""):
        try:
            index = int(match.group(1))
        except ValueError:
            continue
        if index in by_index and index not in seen:
            seen.add(index)
            ordered.append(index)
    # 回答里没标引用时，把 Top 证据全列出来（至少保证"每条结论有证据"）
    if not ordered:
        ordered = sorted(by_index)[:3]

    out: list[Citation] = []
    for index in ordered:
        raw = by_index[index]
        out.append(
            Citation(
                index=index,
                clip_id=str(raw.get("clip_id", "")),
                video_id=str(raw.get("video_id", "")),
                start_ms=int(raw.get("start_ms", 0) or 0),
                end_ms=int(raw.get("end_ms", 0) or 0),
                time_range=str(raw.get("time_range", "")),
                keyframe_paths=tuple(raw.get("keyframe_paths", ()) or ()),
                text=str(raw.get("text", "") or ""),
            )
        )
    return out


# =============================================================================
# 编排器
# =============================================================================


_STAGE_PROGRESS: Final[dict[Stage, float]] = {
    Stage.PARSE: 0.02,
    Stage.PROBE: 0.06,
    Stage.AUDIO: 0.10,
    Stage.TRANSCRIBE: 0.20,
    Stage.SCENE: 0.50,
    Stage.CHUNK: 0.60,
    Stage.VISION: 0.68,
    Stage.INDEX: 0.88,
    Stage.SUMMARY: 0.95,
    Stage.DONE: 1.0,
}

_STAGE_LABEL: Final[dict[Stage, str]] = {
    Stage.PARSE: "解析视频来源",
    Stage.PROBE: "读取媒体信息",
    Stage.AUDIO: "抽取音频",
    Stage.TRANSCRIBE: "语音转写",
    Stage.SCENE: "镜头检测",
    Stage.CHUNK: "切片合并",
    Stage.VISION: "画面理解",
    Stage.INDEX: "建立向量索引",
    Stage.SUMMARY: "生成总结",
    Stage.DONE: "完成",
}
