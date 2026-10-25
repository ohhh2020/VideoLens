"""知影 ZhiYing —— 分析管道编排（把七个模块串成一条可恢复的流水线）。

一条视频从"给个链接"到"能问答"，要过十个阶段
--------------------------------------------

::

    parse → probe → audio → transcribe → scene → chunk → vision → index → summary → ready
      │        │       │         │          │       │        │        │        │
      │        │       │         │          │       │        │        │        └ 整片总结
      │        │       │         │          │       │        │        └ 文本/图像向量入 Chroma
      │        │       │         │          │       │        └ 关键帧 VL 描述 + OCR
      │        │       │         │          │       └ 10~30s 切片 + 关键帧绑定
      │        │       │         │          └ 镜头切分 + 关键帧抽取（含清晰度过滤）
      │        │       │         └ 字幕优先，无字幕走 faster-whisper
      │        │       └ 抽 16kHz 单声道 WAV
      │        └ ffprobe 元数据校验（时长/体积/分辨率上限）
      └ URL 走 yt-dlp，本地文件直接校验

每个阶段都做三件事：写 ``video.status/stage/progress``、写 ``task`` 心跳、
失败时把原因塞进 ``report.degraded`` 而**不是**直接崩。

断点恢复
--------
``analyze(source, resume=True)`` 会读 ``video.stage``，从**失败的那个阶段之前**
重入。已完成的阶段不重跑 —— 1 小时视频的 ASR 要 40 分钟，不能因为最后写
向量失败就全丢。为此每个阶段结束都 ``repo.checkpoint()``。

降级不阻断
----------
需求文档表 4 列了七种故障。这里的处理原则：**能降级就降级，只有拿不到视频本体
才算失败**。

===================  ============================================
阶段失败             降级动作
===================  ============================================
下载失败            重试 3 次（yt-dlp 内部）→ 整体失败
不含音轨            跳过转写，只用视觉描述，``degraded`` 记一笔
转写失败            切 CPU 重试 → 再用字幕 → 都无则空转录
镜头检测失败        ffmpeg 场景检测 → 等分兜底
视觉模型不可用      跳过描述，纯文本链路照跑
向量写入失败        丢弃该批向量，字面 BM25 仍可检索
云端 LLM 失败       本地 GGUF → jieba 关键词摘要
===================  ============================================

用法
----
>>> orch = Orchestrator.from_config()
>>> report = orch.analyze("https://www.bilibili.com/video/BV1xx411c7mD")
>>> report.status, report.clip_count, report.keyframe_count
('ready', 42, 71)
>>> ans = orch.ask(report.video_id, "切片长度是怎么定的")
>>> ans.answer[:30]
'切片长度默认在 10 到 30 秒之间…'
>>> ans.refs[0].time_range
'00:00:20.100–00:00:42.300'
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace as _dc_replace
from enum import Enum
from uuid import uuid4
from pathlib import Path
from typing import Any, Callable, Final, Sequence

from core.config import AppConfig, get_app_config

logger = logging.getLogger("zhiying.pipeline.orchestrator")

#: 全局推理互斥：同一进程里同一时刻只让一个"重模型"占内存。
#: faster-whisper(1.5G) + BGE-M3(2.2G) + Qwen2-VL(3.4G) + 本地 7B(4.5G) 已经超过
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

__all__ = [
    "AnalysisReport",
    "AnalyzeOptions",
    "AskResult",
    "Citation",
    "Orchestrator",
    "PipelineError",
    "Stage",
    "StageReport",
    "compute_fingerprint",
]


def compute_fingerprint(config: AppConfig) -> str:
    """一次分析"用了什么模型、按什么规格切"的指纹。

    只要它变了，旧切片 / 旧向量 / 旧摘要就不能和新结果混用 —— 混用不会抛异常，
    只会让检索出现"同一条证据两个来源、分数不可比"，答案看着通顺、引用是错的。
    """
    media = config.media
    parts = [
        f"asr={config.asr.model}",
        f"text={config.embedding.text_model}",
        f"image={config.embedding.image_model}",
        f"vl={config.vl.model}",
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
    #: 外部任务号（api.jobs.JobStore 发的 ``t<uuid>``）。传入时 ``_start_task``
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


class Orchestrator:
    """分析管道门面。所有依赖可注入，缺哪个降级哪个。"""

    def __init__(
        self,
        *,
        repository: Any,
        vectors: Any = None,
        embedder: Any = None,
        asr: Any = None,
        scene_detector: Any = None,
        chunker: Any = None,
        vl: Any = None,
        llm: Any = None,
        downloader: Any = None,
        context_builder: Any = None,
        searcher: Any = None,
        config: AppConfig | None = None,
        on_progress: Callable[[str, float, str], None] | None = None,
    ) -> None:
        self._repo = repository
        self._vectors = vectors
        self._embedder = embedder
        self._asr = asr
        self._scene = scene_detector
        self._chunker = chunker
        self._vl = vl
        self._llm = llm
        self._downloader = downloader
        self._ctx = context_builder
        self._searcher = searcher
        self._cfg = config or get_app_config()
        self._on_progress = on_progress
        self._lock = threading.RLock()
        # BM25 通道缓存：按 (video_id, 切片数, 末片ID) 记账，切片一变自然失效。
        # 不缓存的话每次提问都要对整库切片重新分词（1 小时视频约 180 片）。
        self._bm25_cache: OrderedDict[str, tuple[tuple[str, int, str], Any]] = OrderedDict()
        self._bm25_lock = threading.Lock()
        #: 最近一次视觉描述实际走的模式（segments/frames[-downgraded]），
        #: 写进 report.profile.vision_mode_used 供前端与验收查看。
        self._last_vision_mode: str = ""

    # -- 只读组件访问（客户端/接口层列数据时用，避免到处摸下划线属性）------

    @property
    def repository(self) -> Any:
        """元数据仓储（列出视频、看详情、统计容量）。"""
        return self._repo

    @property
    def vectors(self) -> Any:
        """向量库（可能为 ``None``：装配失败时已降级）。"""
        return self._vectors

    @property
    def config(self) -> AppConfig:
        return self._cfg

    # -- 装配 -------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: AppConfig | None = None,
        *,
        repository: Any | None = None,
        on_progress: Callable[[str, float, str], None] | None = None,
        load_models: bool = False,
    ) -> Orchestrator:
        """按配置装配全部组件。

        装配失败一律降级（记 WARNING），保证 ``analyze`` 总能跑 —— 只是结果里
        会带 ``degraded`` 条目。``load_models=True`` 时才真正下载权重（首次启动
        会拉几个 GB，测试与 CI 里务必关掉）。
        """
        cfg = config or get_app_config()
        degraded: list[str] = []

        repo = repository
        if repo is None:
            try:
                from core.storage.repository import SqliteRepository  # noqa: PLC0415

                repo = SqliteRepository.from_config(cfg)
            except Exception as exc:  # noqa: BLE001
                raise PipelineError(f"数据库不可用，无法建立分析管道: {exc}") from exc

        # ModelManager 总是装配：它只负责"定位已下载的权重"，不加载也不下载。
        # 少了它，embed/vl/asr 会一律降级成哈希编码/纯文本 —— 权重明明在硬盘上
        # 却用不上（Web 端曾因此比 CLI 差一截）。真正的下载入口仍只有
        # scripts/download_models.py；这里是否允许联网拉取由 models.auto_download 决定。
        manager = None
        try:
            from core.models.manager import get_model_manager  # noqa: PLC0415

            manager = get_model_manager(
                cfg.path("models"),
                allow_download=bool(cfg.models.auto_download),
            )
        except Exception as exc:  # noqa: BLE001
            degraded.append(f"ModelManager 不可用: {exc}")
        # ``load_models`` 保留给调用方表达"启动即加载"的意图；实际加载由各组件
        # 在自己的第一次 encode/transcribe 时懒做 —— 服务启动不该卡几十秒。
        _ = load_models

        def build(label: str, fn: Callable[[], Any]) -> Any:
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                degraded.append(f"{label} 装配失败: {exc}")
                logger.warning("%s 装配失败，将降级: %s", label, exc)
                return None

        vectors = build("向量库", lambda: _build_vectors(cfg))
        embedder = build("向量编码器", lambda: _build_embedder(cfg, manager))
        asr = build("ASR", lambda: _build_asr(cfg, manager))
        scene = build("场景检测", lambda: _build_scene(cfg))
        chunker = build("切片器", lambda: _build_chunker(cfg))
        vl = build("视觉理解", lambda: _build_vl(cfg, manager))
        llm = build("LLM 网关", lambda: _build_llm(cfg, manager))
        downloader = build("下载器", lambda: _build_downloader(cfg))
        ctx = build("上下文组装", lambda: _build_context(cfg))

        orch = cls(
            repository=repo,
            vectors=vectors,
            embedder=embedder,
            asr=asr,
            scene_detector=scene,
            chunker=chunker,
            vl=vl,
            llm=llm,
            downloader=downloader,
            context_builder=ctx,
            config=cfg,
            on_progress=on_progress,
        )
        for reason in degraded:
            logger.warning("管道降级: %s", reason)
        orch._assembly_notes = degraded  # noqa: SLF001 - 供首份报告带出装配告警
        return orch

    # -- 公开入口 ---------------------------------------------------------

    def analyze(self, source: str, options: AnalyzeOptions | None = None) -> AnalysisReport:
        """分析一个视频（本地路径或 URL）。**不抛异常**，失败写进 report。"""
        opts = options or AnalyzeOptions()
        opts.validate()
        started = time.monotonic()
        report = AnalysisReport(duration_ms=0)
        report.degraded.extend(getattr(self, "_assembly_notes", []) or [])
        progress = opts.progress or self._on_progress

        video_id = ""
        ctx: dict[str, Any] = {}
        video_guard: threading.Lock | None = None

        try:
            # ---------------- parse ----------------
            stage = self._begin(report, Stage.PARSE, progress)
            media_source = self._parse_source(source, opts)
            video_id = media_source.video_id
            report.video_id = video_id
            report.title = opts.title or media_source.display_name
            ctx["source"] = media_source

            # 同视频串行：后来者等先到者完成后走快路径（复用结果/断点续跑），
            # 绝不并行跑两条全量流水线。见 _VIDEO_LOCKS 的注释。
            video_guard = _video_lock(video_id)
            if not video_guard.acquire(timeout=0.5):
                logger.info("video_id=%s 已有分析在运行，排队等待其完成", video_id)
                report.warnings.append("同视频已有任务在运行，已排队等待（未重复计算）")
                video_guard.acquire()

            existing = self._repo.find_video(video_id)
            if existing and existing.status == "ready" and not opts.force:
                self._finish(stage, message="视频已分析完成，跳过（force=True 可重跑）")
                report.status = "ready"
                report.clip_count = len(self._repo.get_clips(video_id))
                report.keyframe_count = len(self._repo.get_keyframes(video_id))
                report.elapsed_ms = int((time.monotonic() - started) * 1000)
                report.stages.append(
                    StageReport(stage=Stage.DONE, skipped=True, message="已完成，未重跑")
                )
                self._end(report, Stage.DONE, progress)
                return report
            # 断点必须在 _register_video **之前**读：注册会 upsert 视频行，
            # 把上一轮失败的 stage 覆写成空，恢复点就永远读不到了。
            resume_from = self._resume_point(video_id, opts)
            resume_ctx = self._resume_context(video_id) if resume_from is not None else {}
            self._register_video(media_source, opts)
            # 指纹不一致 → 旧切片/向量必须先清，否则新旧产物混在库里
            stale = self._apply_fingerprint(report, video_id, compute_fingerprint(self._cfg))
            if stale:
                # 指纹变了说明产物已全部作废，刚体检通过的缓存也不能再用
                resume_from, resume_ctx = None, {}
            self._finish(stage, message=f"{media_source.kind}: {media_source.display_name}")

            task_id = self._start_task(video_id, opts)

            # ---------------- probe ----------------
            stage = self._begin(report, Stage.PROBE, progress)
            local_path, probe_info = self._resolve_local(media_source, progress)
            ctx["path"] = local_path
            ctx["probe"] = probe_info
            report.duration_ms = int(getattr(probe_info, "duration_ms", 0) or 0)
            self._repo.update_video_meta(
                video_id,
                duration_ms=report.duration_ms,
                width=int(getattr(probe_info, "display_width", 0) or 0),
                height=int(getattr(probe_info, "display_height", 0) or 0),
                fps=float(getattr(probe_info, "fps", 0.0) or 0.0),
                has_audio=bool(getattr(probe_info, "has_audio", False)),
                local_path=str(local_path),
            )
            self._finish(stage, message=f"时长 {report.duration_ms} ms")

            # ---------------- audio + transcribe ----------------
            stage = self._begin(report, Stage.AUDIO, progress)
            audio_path, has_audio = self._extract_audio(local_path, video_id, probe_info)
            self._finish(
                stage,
                message=str(audio_path) if audio_path else "无音轨或抽取失败",
                degraded=audio_path is None,
            )
            if audio_path is None:
                report.degraded.append("视频不含可用音轨，仅走视觉链路")

            stage = self._begin(report, Stage.TRANSCRIBE, progress)
            transcript = resume_ctx.get("transcript")
            reused = self._resumable(Stage.TRANSCRIBE, resume_from, transcript)
            if not reused:
                transcript = self._transcribe(video_id, audio_path, local_path, probe_info, opts)
            ctx["transcript"] = transcript
            report.transcript_source = str(getattr(transcript, "source", "") or "")
            report.language = str(getattr(transcript, "language", "") or "")
            segments = list(getattr(transcript, "segments", []) or [])
            report.segment_count = len(segments)
            if segments and not reused:
                self._repo.add_segments(video_id, [_segment_payload(s) for s in segments])
            if reused:
                report.warnings.append("转写复用库中已有结果，未重新识别")
            self._finish(
                stage,
                message=(
                    f"{len(segments)} 段，来源 {report.transcript_source or 'asr'}"
                    + ("（断点恢复）" if reused else "")
                ),
                degraded=not segments,
            )
            if not segments:
                report.degraded.append("转写无结果，切片仅依赖视觉与时长")

            # ---------------- scene ----------------
            stage = self._begin(report, Stage.SCENE, progress)
            scenes, detector_used, scene_warnings = self._detect_scenes(local_path, probe_info)
            ctx["scenes"] = scenes
            report.warnings.extend(scene_warnings)
            self._finish(stage, message=f"{len(scenes)} 个场景（{detector_used}）")

            # 内容画像（方案 A 最小实现）：高运动判定驱动视觉分派
            duration_s = max(1.0, (report.duration_ms or 0) / 1000.0)
            scene_density = len(scenes) / (duration_s / 60.0)
            speech_ms = sum(
                max(0, int(getattr(s, "end_ms", 0) or 0) - int(getattr(s, "start_ms", 0) or 0))
                for s in segments
            )
            high_motion = scene_density >= float(self._cfg.vl.high_motion_density_per_min)
            profile = {
                "scene_count": len(scenes),
                "scene_density_per_min": round(scene_density, 1),
                "speech_ratio": round(min(1.0, speech_ms / max(1, report.duration_ms or 1)), 3),
                "transcript_source": report.transcript_source,
                "high_motion": high_motion,
            }
            ctx["profile"] = profile

            # ---------------- chunk ----------------
            stage = self._begin(report, Stage.CHUNK, progress)
            clips = resume_ctx.get("clips")
            reused = self._resumable(Stage.CHUNK, resume_from, clips)
            if not reused:
                clips = self._chunk(video_id, segments, report.duration_ms)
                if clips:
                    self._repo.add_clips(video_id, [_clip_payload(c) for c in clips])
            ctx["clips"] = clips
            report.clip_count = len(clips)
            if reused:
                report.warnings.append("切片复用库中已有结果，未重新切分")
            self._finish(
                stage,
                message=f"{len(clips)} 个切片" + ("（断点恢复）" if reused else ""),
            )
            if not clips:
                raise PipelineError("切片结果为空，无法继续（转录与时长都为空？）")

            # ---------------- vision ----------------
            stage = self._begin(report, Stage.VISION, progress)
            keyframes = resume_ctx.get("keyframes")
            reused = self._resumable(Stage.VISION, resume_from, keyframes)
            if not reused:
                keyframes = self._extract_keyframes(
                    local_path, video_id, scenes, clips, opts, report,
                    high_motion=high_motion,
                )
            profile["vision_mode_used"] = getattr(self, "_last_vision_mode", "")
            report.profile = profile
            ctx["keyframes"] = keyframes
            report.keyframe_count = len(keyframes)
            if reused:
                report.warnings.append("关键帧复用库中已有结果与描述，未重新抽帧")
            self._finish(
                stage,
                message=f"{len(keyframes)} 张关键帧" + ("（断点恢复）" if reused else ""),
                degraded=not keyframes or opts.skip_vision,
            )

            # ---------------- index ----------------
            stage = self._begin(report, Stage.INDEX, progress)
            vectors_written = self._index(video_id, clips, keyframes, report, opts)
            report.vector_count = vectors_written
            self._finish(
                stage,
                message=f"{vectors_written} 条向量",
                degraded=vectors_written == 0 and not opts.skip_vectors,
            )

            # ---------------- summary ----------------
            stage = self._begin(report, Stage.SUMMARY, progress)
            summary, backend = self._summarize(video_id, report.title, report.duration_ms, opts)
            report.summary = summary
            report.summary_backend = backend
            self._finish(stage, message=f"{len(summary)} 字符（{backend}）", degraded=backend == "keywords")

            # ---------------- done ----------------
            self._repo.update_video_status(
                video_id, "ready", stage="ready", progress=1.0, finished=True
            )
            if task_id:
                self._repo.update_task(task_id, status="done", stage="done", progress=1.0,
                                       message="分析完成")
            report.status = "ready"
            self._end(report, Stage.DONE, progress)

        except Exception as exc:  # noqa: BLE001 - 顶层兜底
            message = f"{type(exc).__name__}: {exc}"
            logger.exception("分析失败 video_id=%s", video_id or "?")
            report.status = "failed"
            report.errors.append(message)
            report.stages.append(StageReport(stage=Stage.DONE, ok=False, error=message))
            if video_id:
                try:
                    self._repo.update_video_status(
                        video_id, "failed", stage="failed", error=message, finished=True
                    )
                except Exception:  # noqa: BLE001
                    logger.debug("记录失败状态时再次出错", exc_info=True)
        finally:
            if video_guard is not None:
                video_guard.release()

        report.elapsed_ms = int((time.monotonic() - started) * 1000)
        return report

    def ask(
        self,
        video_id: str,
        question: str,
        *,
        top_k: int | None = None,
        max_context_tokens: int | None = None,
        session_id: str = "",
    ) -> AskResult:
        """基于已索引视频回答一个问题。**不抛异常**。

        ``session_id``：会话分组（/chat 页每段对话一个）；空则历史归入 legacy 组。
        """
        for kind, payload in self._ask_impl(
            video_id, question, top_k=top_k,
            max_context_tokens=max_context_tokens, stream=False,
            session_id=session_id,
        ):
            if kind == "final":
                return payload
        return AskResult(question=question, video_id=video_id)

    def ask_stream(
        self,
        video_id: str,
        question: str,
        *,
        top_k: int | None = None,
        max_context_tokens: int | None = None,
        session_id: str = "",
    ) -> Iterator[tuple[str, Any]]:
        """流式问答：逐 token 产出，最后 yield 定稿结果。

        事件序列：``("meta", {...})`` → 若干 ``("delta", str)`` →
        ``("final", AskResult)``。证据校验（claims）在收尾统一做 —— 流式
        期间的增量可能含被事后撕掉的编号，客户端应以 final 为准替换全文。
        LLM 不支持流式时退化为单个 delta（一次性全文），接口形状不变。
        """
        yield from self._ask_impl(
            video_id, question, top_k=top_k,
            max_context_tokens=max_context_tokens, stream=True,
            session_id=session_id,
        )

    def _ask_impl(
        self,
        video_id: str,
        question: str,
        *,
        top_k: int | None,
        max_context_tokens: int | None,
        stream: bool,
        session_id: str = "",
    ) -> Iterator[tuple[str, Any]]:
        """ask / ask_stream 的共享实现（检索→组装→生成→校验）。

        恒为生成器且以 ``("final", AskResult)`` 收尾；``stream=False`` 时
        不产出 meta/delta，只给 final（ask() 的消费形态）。
        """
        started = time.monotonic()
        result = AskResult(question=question, video_id=video_id)
        text = (question or "").strip()
        if not text:
            result.notes.append("问题为空")
            yield ("final", result)
            return
        video = self._repo.find_video(video_id)
        if video is None:
            result.notes.append(f"视频不存在: {video_id}")
            yield ("final", result)
            return

        # ---- 多轮（C 端对话）----
        # 1) 查询改写：追问常带指代/省略（"那第二个进球呢"），坏查询直接
        #    召回坏证据 —— 改写发生在检索之前，用上一轮问答补全。
        # 2) 历史前缀：最近 N 轮作为 messages 前缀传给 LLM（证据只挂本轮）。
        from core.pipeline.chat_sessions import get_chat_store  # noqa: PLC0415

        store = get_chat_store()
        history = store.history(video_id)
        retrieval_query = text
        if history and self._cfg.qa.query_rewrite:
            rewritten = self._rewrite_query(text, history)
            if rewritten:
                retrieval_query = rewritten
                result.notes.append(f"追问已改写检索: {rewritten[:60]}")

        try:
            items = self._retrieve(video_id, retrieval_query, top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("检索失败，退化为时间线摘要: %s", exc)
            result.notes.append(f"检索失败: {exc}")
            items = []
        result.candidates = len(items)
        if not items:
            items = self._fallback_items(video_id)
            result.notes.append("检索无命中，改用片段摘录作为证据")
            result.degraded = True
        items = self._attach_visuals(video_id, items)

        builder = self._ctx or _build_context(self._cfg)
        if max_context_tokens:
            from core.pipeline.context_builder import (  # noqa: PLC0415
                ContextBudget,
                ContextBuilder,
            )
            from core.prompts import get_prompt_registry  # noqa: PLC0415

            builder = ContextBuilder(
                ContextBudget(max_tokens=max_context_tokens),
                prompts=get_prompt_registry(),
            )
        built = builder.build(
            items, text, video_title=video.title or video.display_name or "该视频"
        )
        result.context_chars = built.used_chars
        result.notes.extend(built.notes)
        if not built.ok:
            result.notes.append("上下文组装为空，无法生成回答")
            yield ("final", result)
            return

        if stream:
            yield ("meta", {
                "video_id": video_id,
                "candidates": result.candidates,
                "context_chars": result.context_chars,
                "backend_hint": getattr(self._llm, "primary", "") if self._llm is not None else "",
            })

        llm = self._llm or _build_llm(self._cfg, None)
        answer_parts: list[str] = []
        generated = False
        resp: Any = None
        stream_fn = getattr(llm, "stream", None)
        # 多轮生成：历史轮作为前缀（只保留问答文本），证据 prompt 永远是
        # 最后一条 user 消息 —— 引用编号只对应本轮检索结果，跨轮不串。
        history_window = int(self._cfg.qa.history_window)
        prefix = history[-history_window * 2:] if history else []
        gen_messages = [*prefix, {"role": "user", "content": built.text}]
        try:
            if stream and callable(stream_fn):
                with self._inference_slot("回答问题", None):
                    for piece in stream_fn(gen_messages):
                        answer_parts.append(str(piece))
                        yield ("delta", str(piece))
                raw_answer = "".join(answer_parts)
                generated = bool(raw_answer.strip())
                if not generated:
                    raise RuntimeError("流式输出为空")
            else:
                with self._inference_slot("回答问题", None):
                    resp = llm.complete(gen_messages)
                raw_answer = resp.text
                generated = bool(raw_answer.strip())
                if stream and generated:
                    # 后端不支持流式（如注入的 Fake LLM / keywords 兜底）：
                    # 一次性吐全文，接口形状不变，首 token 延迟=总延迟。
                    answer_parts = [raw_answer]
                    yield ("delta", raw_answer)
        except Exception as exc:  # noqa: BLE001 - 生成失败退化为证据罗列
            logger.warning("LLM 生成失败: %s", exc)
            result.notes.append(f"LLM 不可用: {exc}")
            raw_answer = _evidence_only_answer(built)
            result.backend = "evidence"
            result.degraded = True
            if stream and not answer_parts:
                yield ("delta", raw_answer)
        else:
            if resp is not None:
                result.backend = resp.backend
                result.degraded = result.degraded or resp.degraded
            elif generated:
                result.backend = getattr(llm, "primary", "stream")

        result.answer = raw_answer
        result.refs = extract_citations(result.answer, built.refs)
        result.answer = _enforce_answer_claims(result, built)
        result.refs = extract_citations(result.answer, built.refs)
        result.elapsed_ms = int((time.monotonic() - started) * 1000)
        # 记忆本轮：内存 LRU（供多轮前缀）+ SQLite 持久化（重启不丢）。
        # session_id 由调用方经 AnalyzeOptions 传入（/chat 页每段对话一个），
        # 空则归入 legacy 组。持久化失败不影响返回。
        persisted = False
        if history or self._cfg.qa.multi_turn:
            try:
                from core.pipeline.chat_sessions import Turn  # noqa: PLC0415

                store.remember(video_id, Turn(
                    question=text, answer=result.answer,
                    rewritten=retrieval_query if retrieval_query != text else "",
                    elapsed_ms=result.elapsed_ms,
                ))
            except Exception:  # noqa: BLE001
                logger.debug("会话记忆失败", exc_info=True)
        try:
            self._repo.add_qa(
                video_id,
                question=text,
                answer=result.answer,
                refs=[r.to_dict() for r in result.refs],
                rewritten=retrieval_query if retrieval_query != text else "",
                elapsed_ms=result.elapsed_ms,
                session_id=session_id,
            )
            persisted = True
        except Exception:  # noqa: BLE001 - 持久化是尽力而为
            logger.debug("问答历史落库失败", exc_info=True)
        result.notes.append("已存历史" if persisted else "历史未落库")
        result.session_id = session_id
        yield ("final", result)

    def _rewrite_query(self, question: str, history: list[dict[str, str]]) -> str:
        """多轮查询改写：用上一轮问答把追问补全成独立查询。

        改写失败/输出异常时返回空串（调用方退回原问题），绝不让这步
        可选优化阻断问答主链路。
        """
        try:
            last_q = ""
            last_a = ""
            for msg in reversed(history):
                if msg.get("role") == "user" and not last_q:
                    last_q = str(msg.get("content", ""))
                elif msg.get("role") == "assistant" and not last_a:
                    last_a = str(msg.get("content", ""))
                if last_q and last_a:
                    break
            if not last_q:
                return ""
            from core.prompts import get_prompt_registry  # noqa: PLC0415

            llm = self._llm or _build_llm(self._cfg, None)
            prompt = get_prompt_registry().render(
                "chat_query_rewrite",
                last_question=last_q[:300],
                last_answer=last_a[:600],
                question=question[:300],
            )
            with self._inference_slot("查询改写", None):
                resp = llm.complete(
                    [{"role": "user", "content": prompt}],
                    max_tokens=120, temperature=0.0,
                )
            out = str(resp.text or "").strip().strip('"“”‘’').strip()
            # 违约防护：改写结果异常长（模型开始解释）或与原问一样时弃用
            if not out or len(out) > 200 or out == question:
                return ""
            return out[:200]
        except Exception as exc:  # noqa: BLE001 - 改写是可选优化
            logger.debug("查询改写失败，使用原问题: %s", exc)
            return ""

    def summarize(self, video_id: str, *, max_items: int | None = None) -> str:
        """重新生成整片总结（不重跑分析）。

        ``max_items=None``（默认）跟随"覆盖全片"预算；显式传值用于裁剪。
        """
        video = self._repo.find_video(video_id)
        if video is None:
            raise PipelineError(f"视频不存在: {video_id}")
        payloads = list(self._repo.iter_clip_payloads(video_id))
        payloads = self._attach_visuals(video_id, payloads)
        from core.pipeline.context_builder import ContextBuilder  # noqa: PLC0415
        from core.prompts import get_prompt_registry  # noqa: PLC0415

        # 与分析期的 _summarize 同一套"覆盖全片"预算（历史 bug：这里沿用
        # 问答的 8 条预算 + 兜底提示词，重新生成的总结与分析期一样被压扁）。
        built = ContextBuilder(
            _summary_budget(len(payloads)), prompts=get_prompt_registry()
        ).build_summary(
            payloads,
            video_title=video.title or video.display_name,
            duration_ms=video.duration_ms,
            max_items=max_items,
        )
        if not built.ok:
            return "（没有足够的切片可供总结）"
        llm = self._llm or _build_llm(self._cfg, None)
        response = llm.complete([{"role": "user", "content": built.text}])
        return response.text

    def chapters(
        self,
        video_id: str,
        *,
        chapter_seconds: int = 300,
        max_chapters: int = 40,
        max_items: int = 8,
        min_seconds: int = 300,
        summarize: bool = True,
    ) -> list[Any]:
        """把长视频按章节切开并逐章摘要，返回 :class:`Chapter` 列表。

        短视频（短于 ``min_seconds``）返回**空列表** —— 半分钟就能读完的内容拆成
        三章只是让读者多点两次。调用方拿到空列表就继续走 :meth:`summarize`。

        每章独立调一次模型，所以长视频的耗时与章节数成正比；``summarize=False``
        只做分组与提取式摘要，用于"先看骨架再决定要不要逐章细化"的场景。
        """
        from core.pipeline.chapters import Chapter, plan_chapters  # noqa: PLC0415

        video = self._repo.find_video(video_id)
        if video is None:
            raise PipelineError(f"视频不存在: {video_id}")
        if video.duration_ms and video.duration_ms < int(min_seconds) * 1000:
            return []
        payloads = list(self._repo.iter_clip_payloads(video_id))
        payloads = self._attach_visuals(video_id, payloads)
        if not payloads:
            return []
        by_id = {str(p.get("clip_id") or ""): p for p in payloads}
        plans = plan_chapters(
            payloads,
            chapter_seconds=chapter_seconds,
            duration_ms=video.duration_ms,
            max_chapters=max_chapters,
        )
        builder = self._ctx or _build_context(self._cfg)
        title = video.title or video.display_name or video_id
        llm = (self._llm or _build_llm(self._cfg, None)) if summarize else None

        out: list[Any] = []
        for plan in plans:
            items = [by_id[cid] for cid in plan.clip_ids if cid in by_id]
            if not items:
                continue
            built = builder.build_summary(
                items,
                video_title=title,
                duration_ms=plan.duration_ms,
                prompt_key="chapter_summarize",
                max_items=max_items,
            )
            summary, backend, degraded = "", "", False
            notes: list[str] = []
            if built.ok and llm is not None:
                try:
                    with self._inference_slot("章节摘要", None):
                        response = llm.complete(
                            [{"role": "user", "content": built.text}],
                            temperature=0.2,
                        )
                    summary = (response.text or "").strip()
                    backend = response.backend
                except Exception as exc:  # noqa: BLE001
                    logger.warning("第 %d 章摘要失败: %s", plan.index, exc)
                    notes.append(f"摘要失败: {exc}")
                    degraded = True
            if not summary:
                summary = _chapter_fallback(items)
                backend = backend or "evidence"
                if not built.ok:
                    degraded = True
                    notes.append("片段太少，未能组成本章上下文")
            else:
                summary = _flatten_qa_summary(_strip_chapter_refs(summary))
            out.append(
                Chapter(
                    index=plan.index,
                    start_ms=plan.start_ms,
                    end_ms=plan.end_ms,
                    summary=summary,
                    clip_ids=plan.clip_ids,
                    backend=backend,
                    degraded=degraded,
                    notes=tuple(notes),
                )
            )
        return out

    def chapter_report(
        self,
        video_id: str,
        *,
        chapter_seconds: int = 300,
        max_chapters: int = 40,
        min_seconds: int = 300,
        total_summary: str = "",
    ) -> str:
        """对外报告入口（CLI export / ``/chapters/report``）。

        自 2026-10 起产出**三层报告**（综合总结 / 内容驱动脉络 / 修字解说词，
        见 :meth:`full_report`）；旧的"逐章摘要拼接"格式仍可经
        ``chapters_markdown`` 取到，但不再是默认导出。
        """
        return self.full_report(
            video_id,
            chapter_seconds=chapter_seconds,
            max_chapters=max_chapters,
            min_seconds=min_seconds,
            total_summary=total_summary,
        )

    # ------------------------------------------------------------------ #
    # 三层报告（改进 A/B）
    # ------------------------------------------------------------------ #

    def full_report(
        self,
        video_id: str,
        *,
        chapter_seconds: int = 300,
        max_chapters: int = 40,
        min_seconds: int = 300,
        total_summary: str = "",
    ) -> str:
        """三层报告：综合总结 / 内容驱动的脉络 / 修字后的详细解说词。

        改进 A：解说词是转写原文经 LLM **只修错字**（不增删不总结）的版本，
        单独成节 —— 长视频的"几千字详细内容"由它承担，而不是靠总结注水。
        改进 B：脉络不再按 30s 网格让模型逐格填空，而是逐章从素材里挑
        真实节点（时间取自证据时间戳，禁止均匀切分），再合并成全片脉络。

        每层都有降级：LLM 不可用 → 脉络取章摘要首句、解说词用原文、
        综合用章节摘要拼接 —— 报告永远出得来，只是层数变薄。
        """
        from core.pipeline.chapters import Chapter, _first_clause  # noqa: PLC0415
        from core.pipeline.context_builder import format_hms  # noqa: PLC0415
        from core.prompts import get_prompt_registry  # noqa: PLC0415

        video = self._repo.find_video(video_id)
        if video is None:
            raise PipelineError(f"视频不存在: {video_id}")
        title = video.title or video.display_name or video_id
        duration_ms = video.duration_ms or 0
        cfg_report = self._cfg.report
        registry = get_prompt_registry()
        llm = self._llm or _build_llm(self._cfg, None)

        try:
            segment_rows = self._repo.get_segments(video_id)
        except Exception:  # noqa: BLE001
            segment_rows = []

        # 1) 章节骨架：长视频走既有逐章摘要；短视频合并为单章
        chapters = self.chapters(
            video_id, chapter_seconds=chapter_seconds,
            max_chapters=max_chapters, min_seconds=min_seconds,
        )
        if not chapters:
            summary = total_summary or self.summarize(video_id)
            chapters = [Chapter(index=1, start_ms=0, end_ms=duration_ms,
                                summary=summary.strip())]

        # 2) 逐章：内容驱动的里程碑 + 修字解说词
        milestone_lines: list[str] = []
        transcript_parts: list[str] = []
        for chapter in chapters:
            lo, hi = chapter.start_ms, max(chapter.end_ms, chapter.start_ms + 1)
            seg_texts = [
                (int(getattr(r, "start_ms", 0) or 0),
                 str(getattr(r, "text", "") or "").strip())
                for r in segment_rows
                if lo <= int(getattr(r, "start_ms", 0) or 0) < hi
                and str(getattr(r, "text", "") or "").strip()
            ]
            chapter_notes: list[str] = []

            # 2a) 里程碑（改进 B）
            if seg_texts:
                context = "\n".join(
                    f"{_ms_label(ms)} {text}" for ms, text in seg_texts[:400]
                )[:6_000]
            else:
                context = f"（本章无语音，画面摘要）{chapter.summary}"
            try:
                prompt = registry.render(
                    "chapter_milestones",
                    title=title, start_time=_ms_label(lo), end_time=_ms_label(hi),
                    max_items=int(cfg_report.milestones_max_per_chapter),
                    context=context,
                )
                with self._inference_slot("章节脉络", None):
                    response = llm.complete(
                        [{"role": "user", "content": prompt}], temperature=0.2
                    )
                lines = _parse_milestones(
                    str(response.text or ""), lo, hi,
                )
            except Exception as exc:  # noqa: BLE001 - 脉络失败降级为章摘要首句
                logger.debug("第 %d 章里程碑失败: %s", chapter.index, exc)
                lines = []
            if not lines:
                first_sentence = _first_clause(chapter.summary) or "本章内容"
                lines = [(lo, first_sentence)]
                chapter_notes.append("脉络降级：章摘要首句")
            milestone_lines.extend(
                # 脉络是秒级信息，毫秒尾数只是噪声
                f"- **{_ms_label(ms)[:-4]}** {text}" for ms, text in lines
            )

            # 2b) 修字解说词（改进 A）
            header = f"### {_ms_label(lo)} – {_ms_label(hi)}"
            if not seg_texts:
                transcript_parts.append(f"{header}\n\n（本章无语音内容）")
                continue
            raw_text = "\n".join(text for _, text in seg_texts)
            if not cfg_report.polish_transcript:
                transcript_parts.append(f"{header}\n\n{raw_text}")
                continue
            polished = self._polish_transcript(llm, registry, raw_text, lo, hi,
                                               notes=chapter_notes)
            transcript_parts.append(f"{header}\n\n{polished or raw_text}")

        # 3) 综合总结：各章摘要再综合（改进 A 第一层）
        if total_summary.strip():
            synthesis = total_summary.strip()
        else:
            digest = "\n\n".join(
                f"[第 {c.index} 章 {_ms_label(c.start_ms)}–{_ms_label(c.end_ms)}] "
                f"{(c.summary or '').strip()}"
                for c in chapters
            )
            try:
                prompt = registry.render(
                    "report_synthesis", title=title,
                    duration=format_hms(duration_ms) if duration_ms else "未知",
                    context=digest[:12_000],
                )
                with self._inference_slot("综合总结", None):
                    response = llm.complete(
                        [{"role": "user", "content": prompt}], temperature=0.3
                    )
                synthesis = str(response.text or "").strip()
            except Exception as exc:  # noqa: BLE001
                logger.debug("综合总结生成失败，退回章节摘要拼接: %s", exc)
                synthesis = digest
            if not synthesis:
                synthesis = digest
        # 综合层自带 "## 内容概览/## 关键结论" 小节，降为三级避免与
        # 报告的二级节标题（综合总结/主要脉络/详细解说词）平级。
        synthesis = re.sub(r"^##\s+", "### ", synthesis, flags=re.M)

        transcript_chars = sum(len(p) for p in transcript_parts)
        lines = [
            f"# {title}",
            "",
            f"时长 {format_hms(duration_ms) if duration_ms else '未知'} ｜ "
            f"{len(chapters)} 章 ｜ 脉络 {len(milestone_lines)} 条 ｜ "
            f"解说词 {transcript_chars} 字",
            "",
            "## 综合总结",
            "",
            synthesis,
            "",
            "## 主要脉络",
            "",
            *milestone_lines,
            "",
            "## 详细解说词",
            "",
            *transcript_parts,
            "",
        ]
        return "\n".join(lines)

    def _polish_transcript(
        self,
        llm: Any,
        registry: Any,
        raw_text: str,
        lo: int,
        hi: int,
        *,
        notes: list[str] | None = None,
    ) -> str:
        """修字（只改错字/标点，不改写）。超长按块切分，逐块调用后拼接。"""
        cap = max(500, int(self._cfg.report.polish_block_chars))
        blocks: list[str] = []
        current: list[str] = []
        used = 0
        for line in raw_text.splitlines():
            if used + len(line) > cap and current:
                blocks.append("\n".join(current))
                current, used = [], 0
            current.append(line)
            used += len(line) + 1
        if current:
            blocks.append("\n".join(current))

        out: list[str] = []
        total = len(blocks)
        for i, block in enumerate(blocks, start=1):
            span_lo = _ms_label(lo) if i == 1 else f"（续 {i}/{total}）"
            span_hi = _ms_label(hi) if i == total else f"（待续 {i}/{total}）"
            try:
                prompt = registry.render(
                    "transcript_polish", start_time=span_lo, end_time=span_hi,
                    transcript=block,
                )
                with self._inference_slot("解说词修字", None):
                    response = llm.complete(
                        [{"role": "user", "content": prompt}], temperature=0.0
                    )
                text = str(response.text or "").strip()
                # 修字层不许 LLM 偷懒总结：输出明显短于输入一半时视为违约，
                # 该块退回原文（宁可保留错字，不能丢内容）。
                if text and len(text) >= len(block) * 0.5:
                    out.append(text)
                else:
                    out.append(block)
                    if notes is not None:
                        notes.append(f"解说词第 {i} 块修字输出过短，保留原文")
            except Exception as exc:  # noqa: BLE001 - 修字失败退原文
                logger.debug("解说词修字失败（块 %d/%d）: %s", i, total, exc)
                out.append(block)
                if notes is not None:
                    notes.append(f"解说词第 {i} 块修字失败，保留原文")
        return "\n\n".join(out)

    def delete(self, video_id: str) -> dict[str, Any]:
        """删视频及其向量与磁盘产物（``DELETE /videos/{id}`` 的后端）。"""
        with self._bm25_lock:
            self._bm25_cache.pop(video_id, None)
        removed: dict[str, Any] = {"video_id": video_id}
        generation = ""
        local_path = ""
        try:
            video = self._repo.find_video(video_id)
            if video is not None:
                generation = video.vector_generation or ""
                local_path = str(getattr(video, "local_path", "") or "")
        except Exception:  # noqa: BLE001
            generation = ""
        try:
            if self._vectors is not None:
                # 先删活动代的两张集合，再按 video_id 扫基础集合（旧数据没有代）。
                if generation:
                    removed["generation"] = self._vectors.drop_generation(generation)
                removed["vectors"] = self._vectors.delete_video(video_id)
        except Exception as exc:  # noqa: BLE001
            removed["vectors_error"] = str(exc)
        try:
            doomed = self._repo.delete_video(video_id)
            removed["db"] = doomed
            # 库删了文件不能留：关键帧 JPG 与音频 WAV 都在按 video_id 命名的
            # 目录里，漏删一次就是几 MB 的孤儿数据，且再也无法从库里定位。
            removed["files"] = self.purge_files(video_id, doomed, local_path=local_path)
        except Exception as exc:  # noqa: BLE001
            removed["db_error"] = str(exc)
        return removed

    def purge_files(
        self, video_id: str, paths: Sequence[str] = (), *, local_path: str = ""
    ) -> dict[str, int]:
        """删掉某视频在磁盘上的产物，返回 ``{类别: 删除文件数}``。

        数据库层只返回关键帧路径（它不碰文件系统），其余目录按约定推断：
        ``{paths.frames}/{vid}`` 与 ``{paths.audio}/{vid}`` 整目录删。
        视频副本只在确实落在本项目数据目录里时才删 —— 用户自己拖进来的
        原始文件不属于系统，删了就是事故。
        """
        removed: dict[str, int] = {}
        for name in ("frames", "audio"):
            root = Path(self._cfg.path(name)) / video_id
            try:
                if root.is_dir():
                    count = sum(1 for item in root.rglob("*") if item.is_file())
                    shutil.rmtree(root, ignore_errors=True)
                    removed[name] = count
            except OSError:
                logger.debug("删除目录失败: %s", root)
        stray = _unlink_all(paths)
        if stray:
            removed["stray"] = stray
        if local_path:
            _unlink_owned(Path(local_path), [Path(self._cfg.path(tag)) for tag in ("videos", "temp")])
            if not Path(local_path).exists():
                removed["source"] = 1
        return removed

    # -- 阶段实现 ---------------------------------------------------------

    def _parse_source(self, source: str, opts: AnalyzeOptions) -> Any:
        from core.media.parser import ParseOptions, parse_source  # noqa: PLC0415

        options = ParseOptions.from_config(self._cfg)
        return parse_source(source, options)

    def _register_video(self, media_source: Any, opts: AnalyzeOptions) -> str:
        """登记视频。**重复登记不能抛** —— 断点续跑时行已存在。"""
        existing = self._repo.find_video(media_source.video_id)
        if existing is not None:
            if opts.force:
                self._repo.add_video(
                    video_id=media_source.video_id,
                    display_name=media_source.display_name,
                    source_kind="local" if media_source.is_local else "url",
                    source_url=str(media_source.url or ""),
                    local_path=str(media_source.path or ""),
                    title=opts.title or media_source.display_name,
                    size_bytes=int(media_source.size_bytes or 0),
                    status="downloading",
                    replace=True,
                )
            else:
                self._repo.update_video_status(
                    media_source.video_id, "downloading", stage=Stage.PARSE.value,
                    progress=0.0, error=None, finished=False,
                )
            return media_source.video_id
        return self._repo.add_video(
            video_id=media_source.video_id,
            display_name=media_source.display_name,
            source_kind="local" if media_source.is_local else "url",
            source_url=str(media_source.url or ""),
            local_path=str(media_source.path or ""),
            title=opts.title or media_source.display_name,
            size_bytes=int(media_source.size_bytes or 0),
            status="downloading",
        )

    def _apply_fingerprint(
        self, report: AnalysisReport, video_id: str, fingerprint: str
    ) -> bool:
        """写入本次分析指纹；与上次不同就清空派生结果，返回是否作废了旧产物。"""
        previous = self._repo.find_video(video_id)
        old = (previous.fingerprint if previous else "") or ""
        stale = bool(old) and old != fingerprint
        if stale:
            try:
                paths = self._repo.reset_outputs(video_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("清空旧产物失败: %s", exc)
                paths = []
            if self._vectors is not None:
                try:
                    self._vectors.delete_video(video_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("删除旧向量失败: %s", exc)
            for path in paths:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    logger.debug("删除关键帧文件失败: %s", path)
            report.warnings.append(
                f"分析参数已变化（指纹 {old} → {fingerprint}），旧切片与向量作废，从头重跑"
            )
            logger.info("指纹变化，清空 %s 的旧产物（%d 个文件）", video_id, len(paths))
        try:
            self._repo.update_video_meta(video_id, fingerprint=fingerprint)
        except Exception:  # noqa: BLE001
            logger.debug("写入分析指纹失败", exc_info=True)
        return stale

    def _resolve_local(
        self, media_source: Any, progress: Callable[[str, float, str], None] | None
    ) -> tuple[Path, Any]:
        """拿到本地文件 + ffprobe 信息（URL 走下载器）。"""
        from core.media.ffmpeg_utils import probe_media  # noqa: PLC0415

        if not media_source.requires_download:
            path = Path(media_source.path)
            if not path.exists():
                raise PipelineError(f"本地文件不存在: {path}")
            info = media_source.info or probe_media(path)
            return path, info

        if self._downloader is None:
            raise PipelineError("URL 输入需要下载器（yt-dlp 未安装或装配失败）")
        self._emit(progress, Stage.PROBE, 0.05, "开始下载")
        result = self._downloader.download(
            media_source, progress=_download_progress(progress)
        )
        path = Path(result.path)
        if not path.exists():
            raise PipelineError(f"下载完成但文件不存在: {path}")
        self._repo.update_video_meta(
            media_source.video_id,
            local_path=str(path),
            size_bytes=int(result.size_bytes or 0),
            duration_ms=int(getattr(result, "duration_ms", 0) or 0),
        )
        return path, probe_media(path)

    def _extract_audio(self, local_path: Path, video_id: str, probe_info: Any) -> tuple[Path | None, bool]:
        from core.media.ffmpeg_utils import extract_audio  # noqa: PLC0415

        if not getattr(probe_info, "has_audio", False):
            return None, False
        audio_dir = Path(self._cfg.path("audio")) / video_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        dest = audio_dir / "audio.wav"
        if dest.exists() and dest.stat().st_size > 1024:
            return dest, True
        try:
            extract_audio(local_path, dest, sample_rate=self._cfg.media.audio_sample_rate)
        except Exception as exc:  # noqa: BLE001
            logger.warning("音频抽取失败: %s", exc)
            return None, False
        return dest, True

    def _transcribe(
        self,
        video_id: str,
        audio_path: Path | None,
        local_path: Path,
        probe_info: Any,
        opts: AnalyzeOptions,
    ) -> Any:
        from core.models.asr import Transcript  # noqa: PLC0415

        if self._asr is None:
            return Transcript(video_id=video_id, segments=[], source="asr", language="")
        subtitles = self._find_subtitles(local_path)
        source_path = audio_path or local_path
        try:
            with self._inference_slot("语音转写", None):
                return self._asr.transcribe(
                    source_path,
                    video_id=video_id,
                    subtitles=subtitles,
                    duration_ms=int(getattr(probe_info, "duration_ms", 0) or 0),
                    allow_subtitle=self._cfg.asr.fallback_to_subtitle,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("转写失败: %s", exc)
            return Transcript(video_id=video_id, segments=[], source="asr", language="")

    @contextmanager
    def _inference_slot(self, label: str, report: AnalysisReport | None) -> Iterator[None]:
        """重模型临界区：跨实例串行，等待期间继续轮询而不是死等。

        为什么不在构造时排队：一次分析要走 ASR → 视觉 → 向量 → LLM 四段，
        中间夹着纯 CPU 的抽帧和切片。整段串行会让第二个任务连"能先干的活"
        都干不了；分段持锁才是真正省内存又不浪费时间的做法。
        """
        if not self._cfg.limits.inference_serial:
            yield
            return
        started = time.monotonic()
        announced = False
        while not _INFERENCE_LOCK.acquire(timeout=_INFERENCE_POLL_S):
            if not announced:
                announced = True
                logger.info("%s 等待其它任务的模型释放", label)
                if report is not None:
                    report.warnings.append(
                        f"{label}排队中：CPU 上同时跑两份权重只会互相拖慢"
                    )
        waited = time.monotonic() - started
        try:
            yield
        finally:
            _INFERENCE_LOCK.release()
        if waited >= 1.0:
            logger.info("%s 排队等待 %.1fs", label, waited)

    def _find_subtitles(self, local_path: Path) -> list[Path]:
        """在同目录下找字幕文件，按优先级排序（外部字幕优先于内嵌字幕）。

        yt-dlp 会把字幕下到 ``<video_id>/`` 里（``source.zh-Hans.vtt``）；
        用户手工放同名 ``.srt`` 也在这里被捡到。
        """
        from core.media.subtitles import SUPPORTED_SUBTITLE_SUFFIXES  # noqa: PLC0415

        order = {suffix: i for i, suffix in enumerate(self._cfg.asr.subtitle_priority)}
        try:
            candidates = [
                p
                for p in local_path.parent.iterdir()
                if p.is_file() and p.suffix.lstrip(".").lower() in SUPPORTED_SUBTITLE_SUFFIXES
            ]
        except OSError:
            return []
        candidates.sort(key=lambda p: (order.get(p.suffix.lstrip(".").lower(), 99), p.name))
        return candidates

    def _detect_scenes(self, local_path: Path, probe_info: Any) -> tuple[list[Any], str, list[str]]:
        if self._scene is None:
            return [], "none", ["场景检测不可用，切片将只依赖转录"]
        detector = self._scene
        warnings: list[str] = []
        # D4/Tier2：长视频检测加速。优先 ffmpeg+NVDEC 硬解+降采样（快一个
        # 量级）；不可用时退 pyscedetect 跳帧档。PySceneDetect 在 Python 里
        # 逐帧算 HSV 差分，高运动长视频实测 6 分半要 10 分钟 —— 检测不该
        # 比视频本身长。
        try:
            duration_ms = int(getattr(probe_info, "duration_ms", 0) or 0)
        except Exception:  # noqa: BLE001
            duration_ms = 0
        long_video = 0 < self._cfg.media.scene_detect.long_video_min_ms <= duration_ms
        if long_video and detector is not None:
            boosted = self._long_video_detector()
            if boosted is not None:
                detector = boosted
                warnings.append(
                    "长视频场景检测走 ffmpeg/NVDEC 硬解 + 480px 打分"
                    "（切点精度略降，耗时大幅下降）"
                )
            else:
                boost = self._scene_boost(duration_ms)
                if boost is not None:
                    try:
                        detector = self._boosted_scene_detector(*boost)
                        warnings.append(
                            f"长视频场景检测加速（frame_skip={boost[0]}, downscale={boost[1]}）："
                            f"切点精度略降，检测耗时大幅下降"
                        )
                    except Exception as exc:  # noqa: BLE001 - 加速失败退回原配置
                        logger.debug("构造加速检测器失败，退回默认: %s", exc)
                        detector = self._scene
        try:
            scenes, det, warns = detector.detect_scenes(local_path, probe_info)
            return list(scenes), str(det), list(warns) + warnings
        except Exception as exc:  # noqa: BLE001
            logger.warning("场景检测失败: %s", exc)
            return [], "none", [f"场景检测失败: {exc}"]

    def _scene_boost(self, duration_ms: int) -> tuple[int, int] | None:
        """长视频的 pyscedetect 加速档（frame_skip/downscale）。

        注意 downscale 不是 SceneSection 的配置项（检测器默认 2），
        这里以"不小于默认值"为下限取 long_video_downscale。
        """
        sec = self._cfg.media.scene_detect
        if sec.long_video_min_ms <= 0 or duration_ms < sec.long_video_min_ms:
            return None
        if sec.detector not in {"content", "adaptive", "hash"}:
            return None  # ffmpeg/fixed 档本来就不慢
        return (
            max(int(sec.frame_skip or 0), int(sec.long_video_frame_skip)),
            max(2, int(sec.long_video_downscale)),
        )

    def _boosted_scene_detector(self, frame_skip: int, downscale: int) -> Any:
        return self._scene_detector_with(frame_skip=frame_skip, downscale=downscale)

    def _long_video_detector(self) -> Any | None:
        """长视频专用检测器（Tier 2）：ffmpeg + NVDEC 硬解 + 480px 打分。

        PySceneDetect 是 Python 逐帧软解，高运动长视频实测比视频本身还长；
        ffmpeg 档走解码器硬解 + 缩小画面后再打分，快一个量级。构造失败
        （配置/参数问题）返回 None，调用方退回 pyscedetect 加速档。
        """
        sec = self._cfg.media.scene_detect
        if not sec.long_video_use_ffmpeg:
            return None
        try:
            return self._scene_detector_with(
                detector="ffmpeg",
                ffmpeg_hwaccel=bool(sec.long_video_hwaccel),
                ffmpeg_detect_width=int(sec.long_video_detect_width),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("构造 ffmpeg 硬解检测器失败: %s", exc)
            return None

    def _chunk(self, video_id: str, segments: list[Any], duration_ms: int) -> list[Any]:
        if self._chunker is None:
            return []
        if not segments:
            # 无转录时按固定间隔造切片，至少让视觉链路有落点
            from core.rag.chunker import Clip  # noqa: PLC0415

            return _synthetic_clips(video_id, duration_ms, self._cfg.chunk.max_ms)
        return list(
            self._chunker.chunk(
                video_id, segments, total_duration_ms=duration_ms or None
            )
        )

    def _extract_keyframes(
        self,
        local_path: Path,
        video_id: str,
        scenes: list[Any],
        clips: list[Any],
        opts: AnalyzeOptions,
        report: AnalysisReport,
        *,
        high_motion: bool = False,
    ) -> list[Any]:
        if self._scene is None or opts.skip_vision:
            return []
        out_dir = Path(self._cfg.path("frames")) / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        clip_bounds = [(c.clip_id, c.core_start_ms, c.core_end_ms) for c in clips]
        resolver = _clip_resolver(clip_bounds)

        # D3 高运动降档：场景密度高的视频（体育/比赛），把相邻场景合并到
        # >= high_motion_scene_span_ms 再抽帧、每场景只留 1 帧 —— 静帧本来就
        # 区分不出两次进攻，多抽只是烧钱烧时间。
        detector = self._scene
        eff_scenes = scenes
        if high_motion and scenes:
            span_ms = int(self._cfg.vl.high_motion_scene_span_ms)
            eff_scenes = _merge_scenes(scenes, span_ms)
            try:
                detector = self._scene_detector_with(frames_per_scene=1, min_frames_per_scene=1)
            except Exception as exc:  # noqa: BLE001
                logger.debug("构造降档检测器失败，沿用原配置: %s", exc)
            report.warnings.append(
                f"高运动视频视觉降档：{len(scenes)} 场景合并为 {len(eff_scenes)} 窗"
                f"（≥{span_ms // 1000}s/窗，每窗 1 帧）"
            )

        try:
            extraction = detector.extract_keyframes(
                local_path,
                video_id,
                out_dir,
                scenes=eff_scenes or None,
                clip_resolver=resolver,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("关键帧抽取失败: %s", exc)
            report.degraded.append(f"关键帧抽取失败: {exc}")
            return []

        keyframes = list(getattr(extraction, "keyframes", []) or [])
        if opts.max_keyframes:
            keyframes = keyframes[: opts.max_keyframes]
        report.warnings.extend(getattr(extraction, "warnings", []) or [])
        if getattr(extraction, "degraded", False):
            report.degraded.append("关键帧抽取降级（部分帧质量不达标）")

        # 视觉理解（描述）：segments / frames 按 vl.video_mode 与高运动画像分派
        payloads: list[dict[str, Any]] = []
        if self._vl is not None and not opts.skip_vision:
            payloads = self._describe_visual(
                local_path, keyframes, report, eff_scenes, high_motion
            )
        else:
            payloads = [_keyframe_payload(kf, "", "") for kf in keyframes]
        if payloads:
            self._repo.add_keyframes(video_id, payloads)
            try:
                self._repo.set_keyframe_binding(video_id)
            except Exception:  # noqa: BLE001
                logger.debug("回填 clip.keyframe_ids 失败", exc_info=True)
        else:
            # 整轮抽帧失败：清掉上一轮的记录，别让指向已删文件的孤儿帧继续被检索到
            try:
                self._repo.clear_keyframes(video_id)
            except Exception:  # noqa: BLE001
                logger.debug("清理旧关键帧记录失败", exc_info=True)
        return keyframes

    def _scene_detector_with(self, **overrides: Any) -> Any:
        """按当前配置构造场景检测器，允许临时覆盖若干参数（D3 降档用）。"""
        from core.media.scene_detect import SceneDetectConfig, SceneDetector  # noqa: PLC0415

        payload = self._cfg.media.scene_detect.flattened()
        payload.update({k: v for k, v in overrides.items()})
        known = set(SceneDetectConfig.__dataclass_fields__)
        return SceneDetector(
            SceneDetectConfig(**{k: v for k, v in payload.items() if k in known})
        )

    def _describe_visual(
        self,
        local_path: Path,
        keyframes: list[Any],
        report: AnalysisReport,
        scenes: list[Any],
        high_motion: bool,
    ) -> list[dict[str, Any]]:
        """视觉描述总分派：视频段（云端）/ 抽帧（本地或云端）按配置与画像选路。"""
        vl = self._vl
        if vl is None or not keyframes:
            return [_keyframe_payload(kf, "", "") for kf in keyframes]
        mode = str(getattr(self._cfg.vl, "video_mode", "auto") or "auto")
        supports_seg = bool(getattr(vl, "supports_video_segments", False))
        want_segments = mode == "segments" or (mode == "auto" and high_motion)
        if want_segments and supports_seg and local_path and Path(local_path).is_file():
            self._last_vision_mode = "segments"
            payloads = self._describe_by_segments(vl, Path(local_path), keyframes, report, scenes)
            if payloads:
                return payloads
            # 整条段通道没产出（全部失败）→ 落回抽帧，不能让视觉整体缺失
            self._last_vision_mode = "segments→frames"
        self._last_vision_mode = "frames" + ("-downgraded" if high_motion else "")
        cloud = bool(getattr(vl, "is_cloud", False))
        if cloud:
            # D5：云端是网络型调用，不吃本地显存，无需推理锁，且可并发。
            return self._describe_keyframes(keyframes, report, parallel=True)
        with self._inference_slot("画面理解", report):
            return self._describe_keyframes(keyframes, report)

    def _describe_by_segments(
        self,
        vl: Any,
        video_path: Path,
        keyframes: list[Any],
        report: AnalysisReport,
        scenes: list[Any],
    ) -> list[dict[str, Any]]:
        """P1 视频段理解：场景聚成 ≤ video_segment_max_s 的时间窗，整段发云端。

        每窗产出一段含运动语义的描述（顺带读出画面烧录字幕），写进该窗全部
        关键帧；单窗失败退回该窗抽帧描述；段数受 ``video_segment_budget``
        约束（费用闸门），超出部分直接走抽帧。
        """
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        cfg_vl = self._cfg.vl
        chunks = _segment_chunks(scenes or [], int(cfg_vl.video_segment_max_s))
        if not chunks:
            return []
        budget = max(1, int(cfg_vl.video_segment_budget))
        seg_chunks, rest_chunks = chunks[:budget], chunks[budget:]
        if rest_chunks:
            report.warnings.append(
                f"视频段预算 {budget} 用满，其余 {len(rest_chunks)} 窗走抽帧描述"
            )
        by_scene: dict[int, list[Any]] = {}
        for kf in keyframes:
            by_scene.setdefault(int(getattr(kf, "scene_index", 0)), []).append(kf)

        payloads_by_id: dict[str, dict[str, Any]] = {}
        ok = degraded = 0
        errors: dict[str, int] = {}

        def run_segment(chunk: list[int]) -> tuple[list[int], Any]:
            start = int(scenes[chunk[0]].start_ms)
            end = int(scenes[chunk[-1]].end_ms)
            return chunk, vl.describe_video_segment(video_path, start, end)

        workers = max(1, min(4, len(seg_chunks)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vl-seg") as pool:
            for chunk, result in pool.map(run_segment, seg_chunks):
                frames = [kf for idx in chunk for kf in by_scene.get(idx, [])]
                if result is None or getattr(result, "degraded", False):
                    degraded += 1
                    reason = str(getattr(result, "error", "") or "未知原因")
                    errors[reason] = errors.get(reason, 0) + 1
                    for payload in self._describe_frames_subset(vl, frames):
                        payloads_by_id[payload["keyframe_id"]] = payload
                else:
                    ok += 1
                    for kf in frames:
                        payloads_by_id[kf.keyframe_id] = _keyframe_payload(
                            kf, result.description, result.ocr_text
                        )

        # 预算外的窗：走抽帧描述（云端并发 / 本地串行）
        rest_scene_ids = {idx for chunk in rest_chunks for idx in chunk}
        rest_frames = [kf for kf in keyframes
                       if int(getattr(kf, "scene_index", 0)) in rest_scene_ids]
        if rest_frames:
            for payload in self._describe_keyframes(rest_frames, report,
                                                    parallel=bool(getattr(vl, "is_cloud", False))):
                payloads_by_id[payload["keyframe_id"]] = payload

        if ok or degraded:
            report.warnings.append(
                f"视频段理解: {ok} 窗成功、{degraded} 窗回退抽帧（≤{cfg_vl.video_segment_max_s}s/窗）"
            )
        for reason, count in errors.items():
            report.degraded.append(
                f"视频段理解降级 {count} 窗: {reason[:120]}"
            )
        return [payloads_by_id.get(kf.keyframe_id, _keyframe_payload(kf, "", ""))
                for kf in keyframes]

    def _describe_frames_subset(
        self, vl: Any, frames: list[Any]
    ) -> list[dict[str, Any]]:
        """一小段关键帧的抽帧描述（段通道单窗失败时的回退）。"""
        if not frames:
            return []
        describe_sequence = getattr(vl, "describe_sequence", None)
        if bool(self._cfg.vl.sequence_mode) and callable(describe_sequence) and len(frames) > 1:
            span = (_kf_ms(frames[0]), _kf_ms(frames[-1]))
            result = describe_sequence(
                [(kf.path, _kf_ms(kf)) for kf in frames], span_ms=span
            )
            return [_keyframe_payload(kf, result.description, result.ocr_text) for kf in frames]
        out = []
        for kf in frames:
            result = vl.describe(kf.path)
            out.append(_keyframe_payload(kf, result.description, result.ocr_text))
        return out

    def _describe_keyframes(
        self, keyframes: list[Any], report: AnalysisReport, *, parallel: bool = False
    ) -> list[dict[str, Any]]:
        """关键帧 → 画面描述。连续模式下按场景分组，一次推理出一段叙述。

        ``parallel=True`` 仅用于云端后端（网络型，不吃本地显存）；
        本地 GGUF 始终串行并受推理锁保护。
        """
        vl = self._vl
        if vl is None:
            return [_keyframe_payload(kf, "", "") for kf in keyframes]
        payloads: list[dict[str, Any]] = []
        # 逐帧 append 会把"模型权重缺失"这一件事写成 N 条同样的告警（7 张帧 7 条），
        # 用户读到的是一屏废话。这里按原因归并，最后每个原因只报一次并带上帧数。
        vision_errors: dict[str, int] = {}
        describe_sequence = getattr(vl, "describe_sequence", None)
        sequence_ok = bool(self._cfg.vl.sequence_mode) and callable(describe_sequence)

        if sequence_ok:
            groups = _group_by_scene(keyframes, self._cfg.vl.sequence_frames)

            def work(group: list[Any]) -> tuple[Any, list[Any]]:
                span = (_kf_ms(group[0]), _kf_ms(group[-1]))
                return describe_sequence(
                    [(kf.path, _kf_ms(kf)) for kf in group], span_ms=span
                ), group

            if parallel and len(groups) > 1:
                from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

                with ThreadPoolExecutor(
                    max_workers=min(4, len(groups)), thread_name_prefix="vl-cloud"
                ) as pool:
                    results = list(pool.map(work, groups))
            else:
                results = [work(g) for g in groups]
            for result, group in results:
                span = (_kf_ms(group[0]), _kf_ms(group[-1]))
                if result.degraded:
                    reason = (str(result.error or "未知原因")).strip()
                    vision_errors[reason] = vision_errors.get(reason, 0) + len(group)
                else:
                    report.warnings.append(
                        f"连续画面理解: {_ms_label(span[0])}–{_ms_label(span[1])}"
                        f"（{len(group)} 帧）"
                    )
                for kf in group:
                    payloads.append(
                        _keyframe_payload(kf, result.description, result.ocr_text)
                    )
        else:
            for kf in keyframes:
                result = vl.describe(kf.path)
                if result.degraded:
                    reason = (str(result.error or "未知原因")).strip()
                    vision_errors[reason] = vision_errors.get(reason, 0) + 1
                payloads.append(_keyframe_payload(kf, result.description, result.ocr_text))
        for reason, count in vision_errors.items():
            report.degraded.append(
                f"关键帧视觉理解降级 {count}/{len(keyframes)} 张: {reason[:120]}"
            )
        return payloads

    def _index(
        self,
        video_id: str,
        clips: list[Any],
        keyframes: list[Any],
        report: AnalysisReport,
        opts: AnalyzeOptions,
    ) -> int:
        if opts.skip_vectors or self._vectors is None or self._embedder is None:
            if self._vectors is None:
                report.degraded.append("向量库不可用，检索将退化为 BM25 字面匹配")
            return 0

        written = 0
        # 本轮写进新的一代，全部写完才切指针；中途失败只丢半成品，
        # 上一代索引保持可用（否则重跑一次失败的代价是检索彻底空手）。
        store = self._vectors
        generation = uuid4().hex[:12]
        try:
            store = self._vectors.with_generation(generation)
        except Exception as exc:  # noqa: BLE001
            logger.warning("无法创建索引代，退回原地覆盖写入: %s", exc)
            generation = ""
            store = self._vectors

        # 文本向量
        try:
            texts = [c.text or c.core_text for c in clips]
            with self._inference_slot("文本向量编码", report):
                embeddings = self._embedder.embed_texts(texts)
            written += store.upsert_clips(
                ids=[c.clip_id for c in clips],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "clip_id": c.clip_id,
                        "video_id": video_id,
                        "index_no": _clip_index(c),
                        "start_ms": int(c.start_ms),
                        "end_ms": int(c.end_ms),
                        "core_start_ms": int(c.core_start_ms),
                        "core_end_ms": int(c.core_end_ms),
                    }
                    for c in clips
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("文本向量写入失败: %s", exc)
            report.degraded.append(f"文本向量写入失败: {exc}")

        # 图像向量
        if keyframes:
            try:
                paths = [Path(k.path) for k in keyframes]
                with self._inference_slot("图像向量编码", report):
                    image_vectors = self._embedder.embed_images(paths)
                descriptions = {k.keyframe_id: "" for k in keyframes}
                for record in self._repo.get_keyframes(video_id):
                    descriptions[record.keyframe_id] = record.description
                written += store.upsert_frames(
                    ids=[k.keyframe_id for k in keyframes],
                    embeddings=image_vectors,
                    documents=[descriptions.get(k.keyframe_id, "") or k.keyframe_id for k in keyframes],
                    metadatas=[
                        {
                            "keyframe_id": k.keyframe_id,
                            "clip_id": k.clip_id or "",
                            "video_id": video_id,
                            "timestamp_ms": int(k.timestamp_ms),
                            "keyframe_path": str(k.path),
                        }
                        for k in keyframes
                    ],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("图像向量写入失败: %s", exc)
                report.degraded.append(f"图像向量写入失败: {exc}")

        try:
            self._repo.checkpoint()
        except Exception:  # noqa: BLE001
            pass

        if generation and written > 0:
            self._commit_generation(video_id, generation, report)
        elif generation:
            try:
                self._vectors.drop_generation(generation)
            except Exception:  # noqa: BLE001
                pass
        return written

    def _commit_generation(
        self, video_id: str, generation: str, report: AnalysisReport
    ) -> None:
        """把活动指针切到新代，再清掉上一代。

        顺序不能反：先删旧的再写指针，中间崩溃就没有任何一代可用。
        """
        previous = ""
        try:
            video = self._repo.find_video(video_id)
            previous = video.vector_generation if video else ""
            self._repo.update_video_meta(video_id, vector_generation=generation)
        except Exception as exc:  # noqa: BLE001
            logger.warning("索引指针切换失败: %s", exc)
            report.degraded.append(f"索引指针切换失败: {exc}")
            # 指针没切过去，检索仍走上一代；本轮写满的新代已成孤儿，必须清掉，
            # 否则它会永远留在向量库里（stats 双计数、磁盘白占）。
            try:
                self._vectors.drop_generation(generation)
            except Exception:  # noqa: BLE001
                logger.debug("清理未提交的索引代失败: %s", generation, exc_info=True)
            return
        if previous and previous != generation:
            try:
                self._vectors.drop_generation(previous)
            except Exception as exc:  # noqa: BLE001
                logger.debug("清理上一代索引失败（不影响检索）: %s", exc)
        elif not previous:
            # 旧数据没有代号：基础集合里可能还留着上一轮按 video_id 写的向量。
            # 新代已经接管检索，这些行留着只占盘。
            try:
                self._vectors.delete_video(video_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("清理旧基础集合失败（不影响检索）: %s", exc)

    def _summarize(
        self, video_id: str, title: str, duration_ms: int, opts: AnalyzeOptions
    ) -> tuple[str, str]:
        if opts.skip_summary:
            return "", "skipped"
        payloads = list(self._repo.iter_clip_payloads(video_id))
        payloads = self._attach_visuals(video_id, payloads)
        from core.pipeline.context_builder import ContextBuilder  # noqa: PLC0415
        from core.prompts import get_prompt_registry  # noqa: PLC0415

        built = ContextBuilder(
            _summary_budget(len(payloads)), prompts=get_prompt_registry()
        ).build_summary(payloads, video_title=title, duration_ms=duration_ms)
        if not built.ok:
            return "", "none"
        llm = self._llm or _build_llm(self._cfg, None)
        try:
            with self._inference_slot("总结生成", None):
                response = llm.complete([{"role": "user", "content": built.text}])
            return response.text, response.backend
        except Exception as exc:  # noqa: BLE001
            logger.warning("总结失败: %s", exc)
            return _evidence_only_answer(built), "evidence"

    # -- 检索 -------------------------------------------------------------

    def _attach_visuals(self, video_id: str, items: list[Any]) -> list[Any]:
        """把命中关键帧的画面描述拼进证据文本。

        图像通道能把「问的是什么画面」映射到相关帧，但帧是二进制、
        LLM 看不见；只有 ``VlService`` 写下的 description / ocr_text 是文字。
        不拼进上下文，问「画面里有什么颜色、屏幕上写着什么」时模型只能
        看到转写文本，会答成「没有提到」。这是跨模态检索能不能落地
        的关键一步，不是可选的加分项。
        """
        if not items:
            return items
        try:
            frames = {kf.keyframe_id: kf for kf in self._repo.get_keyframes(video_id)}
        except Exception as exc:  # noqa: BLE001 - 视觉注入失败不应影响问答
            logger.debug("读取关键帧失败，跳过视觉注入: %s", exc)
            return items
        if not frames:
            return items

        out: list[Any] = []
        # D2 跨片段折叠：运动类视频每段画面的描述几乎全同（"球员奔跑传球"），
        # 14 段同质证据进总结就是 8 遍复读 + 逼模型在结尾幻觉差异。
        # 首次出现的描述保留全文，相似（bigram-Dice ≥ 0.75）的后续片段折叠成
        # "画面与前段类似"，时间跨度保留 —— 覆盖率不丢，重复信息只算一次。
        from core.pipeline.claims import dice  # noqa: PLC0415

        seen_descs: list[str] = []
        for item in items:
            notes: list[str] = []
            # 一段连续画面会被写进同一组内的每一帧（同一段叙述 + 不同时间戳），
            # 这里按"叙述内容"归并，并把跨度显示成时间区间 —— 这才能让
            # 模型看出「这是 0:01 到 0:12 之间的一段」，而不是两个孤立帧。
            grouped: dict[str, list[int]] = {}
            for fid in _item_keyframe_ids(item):
                kf = frames.get(fid)
                if kf is None:
                    continue
                desc = str(getattr(kf, "description", "") or "").strip()
                ocr = str(getattr(kf, "ocr_text", "") or "").strip()
                if ocr and ocr not in desc:
                    desc = f"{desc}（画面文字：{ocr}）" if desc else f"（画面文字：{ocr}）"
                if not desc:
                    continue
                ms = int(getattr(kf, "timestamp_ms", 0) or 0)
                grouped.setdefault(desc, []).append(ms)
            if not grouped:
                out.append(item)
                continue
            joined = " ".join(
                d for d, _ in sorted(grouped.items(), key=lambda kv: min(kv[1]))
            )
            similar = next((s for s in seen_descs if dice(s, joined) >= 0.75), None)
            if similar is None:
                seen_descs.append(joined)
                for desc, stamps in sorted(grouped.items(), key=lambda kv: min(kv[1])):
                    lo, hi = min(stamps), max(stamps)
                    span = (
                        f"{_ms_label(lo)}–{_ms_label(hi)}" if hi - lo > 500 else _ms_label(lo)
                    )
                    notes.append(f"【画面 {span}】{desc[:240]}")
            else:
                stamps_all = [ms for sts in grouped.values() for ms in sts]
                lo, hi = min(stamps_all), max(stamps_all)
                notes.append(
                    f"【画面 {_ms_label(lo)}–{_ms_label(hi)}】"
                    f"（画面与前段类似，此处不重复展开）"
                )
            extra = "\n".join(notes)
            base = _item_text(item).strip()
            out.append(_item_with_text(item, f"{base}\n{extra}" if base else extra))
        return out

    def _retrieve(self, video_id: str, query: str, top_k: int | None) -> list[Any]:
        from core.rag.hybrid_search import (  # noqa: PLC0415
            BM25Channel,
            CallableEmbedder,
            HybridConfig,
            HybridSearcher,
            VectorChannel,
        )

        channels: list[Any] = []
        # 检索走这个视频当前活动的索引代。代为空说明是旧数据或还没建过索引，
        # 用基础集合名 —— 不强行猜，查不到就由 BM25 兜底。
        store = self._vectors
        if store is not None:
            try:
                video = self._repo.find_video(video_id)
                generation = (video.vector_generation if video else "") or ""
                if generation:
                    store = store.with_generation(generation)
            except Exception as exc:  # noqa: BLE001
                logger.debug("读取索引代号失败，退回基础集合: %s", exc)
        if store is not None and self._embedder is not None:
            # 查询向量必须在这里同步算好。通道有 channel_timeout_ms（默认 3s）
            # 的硬超时，把首次模型调用放进通道线程，会让每个新进程的第一次
            # 提问必然超时、整段检索退化成 fallback —— 用户看到的就是
            # 「第一问永远答没有提到，第二问开始才正常」。
            text_vec = self._query_vector(query, cross_modal=False)
            image_vec = self._query_vector(query, cross_modal=True)
            if text_vec is not None:
                try:
                    # create=False：检索是读路径，集合不存在就跳过该通道，
                    # 不能因为一次查询凭空建出空集合（stats 的集合数会虚涨）。
                    clip_coll = store.collection("clip", create=False)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("文本向量通道不可用: %s", exc)
                    clip_coll = None
                if clip_coll is not None:
                    channels.append(
                        VectorChannel(
                            "text",
                            clip_coll,
                            CallableEmbedder(lambda _text, _v=text_vec: _v),
                        )
                    )
            if image_vec is not None:
                try:
                    frame_coll = store.collection("frame", create=False)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("图像向量通道不可用: %s", exc)
                    frame_coll = None
                if frame_coll is not None:
                    channels.append(
                        VectorChannel(
                            "image",
                            frame_coll,
                            CallableEmbedder(lambda _text, _v=image_vec: _v),
                            id_field="keyframe_id",
                        )
                    )

        payloads = list(self._repo.iter_clip_payloads(video_id))
        clip_index = {str(p["clip_id"]): p for p in payloads}
        bm25 = self._cached_bm25(video_id, payloads)
        if bm25 is not None:
            channels.append(bm25)

        if not channels:
            return []

        cfg = HybridConfig(
            top_k=top_k or self._cfg.retrieval.top_k,
            weights=dict(self._cfg.retrieval.weights),
            k=self._cfg.retrieval.rrf_k,
            min_score=self._cfg.retrieval.min_score,
            timeout_ms=self._cfg.retrieval.channel_timeout_ms,
        )
        keyframe_paths = self._repo.keyframe_paths(video_id)
        searcher = HybridSearcher(
            channels,
            cfg,
            clip_index=clip_index,
            keyframe_resolver=lambda fid: keyframe_paths.get(fid),
        )
        try:
            response = searcher.search(query, filters={"video_id": video_id})
        finally:
            # searcher 自带线程池（每实例最多 3 线程），不关就随提问泄漏。
            searcher.close()
        return list(response.items)

    def _cached_bm25(self, video_id: str, payloads: list[Any]) -> Any | None:
        """BM25 通道按视频缓存（切片数/末片 ID 变化即失效），最多留 4 个视频。"""
        if not payloads:
            return None
        from core.rag.hybrid_search import BM25Channel  # noqa: PLC0415

        stamp = (video_id, len(payloads), str(payloads[-1].get("clip_id") or ""))
        with self._bm25_lock:
            hit = self._bm25_cache.get(video_id)
            if hit is not None and hit[0] == stamp:
                self._bm25_cache.move_to_end(video_id)
                return hit[1]
        try:
            channel = BM25Channel.from_clips(payloads)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 通道不可用: %s", exc)
            return None
        with self._bm25_lock:
            self._bm25_cache[video_id] = (stamp, channel)
            self._bm25_cache.move_to_end(video_id)
            while len(self._bm25_cache) > 4:
                self._bm25_cache.popitem(last=False)
        return channel

    def _query_vector(self, query: str, *, cross_modal: bool) -> Any:
        """同步算查询向量；失败返回 ``None`` 让对应通道缺席而不是拖垮检索。"""
        if self._embedder is None:
            return None
        try:
            if cross_modal:
                return self._embedder.embed_cross_modal(query)
            return self._embedder.embed_query(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "%s 查询向量生成失败: %s", "跨模态" if cross_modal else "文本", exc
            )
            return None

    def _fallback_items(self, video_id: str, limit: int = 6) -> list[dict[str, Any]]:
        payloads = sorted(
            self._repo.iter_clip_payloads(video_id), key=lambda p: int(p.get("start_ms", 0))
        )
        return payloads[:limit]

    # -- 任务与进度 -------------------------------------------------------

    def _resumable(self, stage: Stage, resume_from: Stage | None, cached: Any) -> bool:
        """该阶段能否直接用库里已有的结果顶替：断点之后、且确实有产物。"""
        if resume_from is None or not cached:
            return False
        return _STAGE_ORDER[stage] < _STAGE_ORDER[resume_from]

    def _resume_context(self, video_id: str) -> dict[str, Any]:
        """把库里已产出的中间结果读回来，供断点续跑跳过昂贵阶段。

        关键差别在**体检**：不能只问"库里有几行"，还要问"产出的东西还在不在"。
        关键帧被清理过、切片是上一次半途写的，硬接上去的后果是拿着空集合去建向量 ——
        跑完不报错，检索一条也命中不了，比直接重跑更糟。
        """
        ready: dict[str, Any] = {}
        try:
            segments = self._repo.get_segments(video_id)
        except Exception:  # noqa: BLE001
            segments = []
        if segments:
            ready["transcript"] = _restore_transcript(video_id, segments)
            ready["segments"] = list(getattr(ready["transcript"], "segments", ()))

        try:
            clip_records = self._repo.get_clips(video_id)
        except Exception:  # noqa: BLE001
            clip_records = []
        if clip_records:
            ready["clips"] = _restore_clips(clip_records)

        try:
            kf_records = self._repo.get_keyframes(video_id)
        except Exception:  # noqa: BLE001
            kf_records = []
        if kf_records:
            keyframes = _restore_keyframes(kf_records)
            alive = len(keyframes) / len(kf_records)
            if keyframes and alive >= 0.8:
                ready["keyframes"] = keyframes
            else:
                logger.info(
                    "视频 %s 的关键帧只剩 %d/%d 个文件，放弃恢复视觉结果",
                    video_id, len(keyframes), len(kf_records),
                )
                for key in ("clips", "keyframes"):
                    ready.pop(key, None)
        return ready

    def _resume_point(self, video_id: str, opts: AnalyzeOptions) -> Stage | None:
        if not opts.resume or opts.force:
            return None
        video = self._repo.find_video(video_id)
        if video is None or video.status == "ready":
            return None
        stage_name = (video.stage or "").strip()
        for stage in Stage:
            if stage.value == stage_name:
                logger.info("从阶段 %s 恢复 video_id=%s", stage_name, video_id)
                return stage
        return None

    def _start_task(self, video_id: str, opts: AnalyzeOptions) -> str:
        try:
            return self._repo.upsert_task(
                video_id,
                task_id=opts.task_id or None,
                kind="ingest",
                status="running",
                stage=Stage.PARSE.value,
                progress=0.0,
                message="开始分析",
                payload={"force": opts.force, "resume": opts.resume},
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("创建任务记录失败: %s", exc)
            return ""

    def _begin(
        self, report: AnalysisReport, stage: Stage, progress: Callable[[str, float, str], None] | None
    ) -> StageReport:
        item = StageReport(stage=stage, started_at=time.monotonic())
        report.stages.append(item)
        self._emit(progress, stage, _STAGE_PROGRESS[stage], _STAGE_LABEL[stage])
        if report.video_id:
            try:
                self._repo.update_video_status(
                    report.video_id, _STAGE_STATUS[stage], stage=stage.value,
                    progress=_STAGE_PROGRESS[stage],
                )
            except Exception:  # noqa: BLE001
                logger.debug("更新 video 阶段失败", exc_info=True)
        return item

    def _finish(
        self, item: StageReport, *, message: str = "", degraded: bool = False, elapsed_ms: int = 0
    ) -> None:
        item.message = message
        item.degraded = degraded
        # 显式传了就用传的；没传就按 _begin 记的起点现算。以前这里恒等于 0，
        # 于是所有阶段耗时都显示 0ms，"1h 视频 ≤30min" 的性能指标根本无从核对。
        if elapsed_ms:
            item.elapsed_ms = elapsed_ms
        elif item.started_at:
            item.elapsed_ms = max(0, int((time.monotonic() - item.started_at) * 1000))

    def _end(
        self, report: AnalysisReport, stage: Stage, progress: Callable[[str, float, str], None] | None
    ) -> None:
        self._emit(progress, stage, 1.0, "完成")

    def _emit(
        self,
        progress: Callable[[str, float, str], None] | None,
        stage: Stage,
        ratio: float,
        message: str,
    ) -> None:
        if progress is None:
            return
        try:
            progress(stage.value, max(0.0, min(1.0, ratio)), message)
        except Exception:  # noqa: BLE001 - 回调异常不该影响分析
            logger.debug("进度回调异常", exc_info=True)

    # -- 其它 -------------------------------------------------------------

    def status(self, video_id: str) -> dict[str, Any]:
        video = self._repo.find_video(video_id)
        if video is None:
            return {"video_id": video_id, "exists": False}
        return {"exists": True, **video.to_dict()}

    def close(self) -> None:
        for component in (self._vectors, self._embedder, self._vl, self._llm):
            close = getattr(component, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    logger.debug("关闭组件时忽略异常", exc_info=True)
        close = getattr(self._repo, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> Orchestrator:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        parts = []
        for label, comp in (
            ("asr", self._asr), ("vl", self._vl), ("embed", self._embedder),
            ("vectors", self._vectors), ("llm", self._llm),
        ):
            parts.append(f"{label}={'on' if comp is not None else 'off'}")
        return f"Orchestrator({', '.join(parts)})"


# =============================================================================
# 内部工具
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
    Stage.SCENE: "镜头检测与抽帧",
    Stage.CHUNK: "切片合并",
    Stage.VISION: "画面理解",
    Stage.INDEX: "建立向量索引",
    Stage.SUMMARY: "生成总结",
    Stage.DONE: "完成",
}


def _summary_budget(clip_count: int) -> Any:
    """总结用的专用证据预算（与问答的 top_k+2 上限解耦）。

    问答要"挑最相关的"，总结要"覆盖全片"。早期共用 8 条预算，16 片的视频
    有一半内容模型根本没看到 —— 5 分钟的讲话被压成三句话不是模型的错。
    全量片段上限 32：再长的视频靠"按位置铺开"保证首尾，超出预算的部分
    仍由 build_summary 的 spread_order 挑代表。
    """
    from core.pipeline.context_builder import ContextBudget  # noqa: PLC0415

    count = max(0, int(clip_count or 0))
    return ContextBudget(
        max_tokens=8_000,
        max_chars=max(12_000, 700 * max(1, count)),
        max_items=max(12, min(32, count or 12)),
        max_chars_per_item=1_200,
    )


def _download_progress(
    progress: Callable[[str, float, str], None] | None,
) -> Callable[[Any], None] | None:
    if progress is None:
        return None

    def handler(event: Any) -> None:
        percent = float(getattr(event, "percent", 0.0) or 0.0)
        # 下载占总进度 0~6%
        progress(Stage.PROBE.value, 0.02 + 0.04 * (percent / 100.0), getattr(event, "human", "") or "下载中")

    return handler


def _clip_resolver(bounds: Sequence[tuple[str, int, int]]) -> Callable[[int], str | None]:
    """毫秒时间戳 → 命中的 clip_id（用于把关键帧绑到切片）。"""
    ordered = sorted(bounds, key=lambda x: x[1])

    def resolve(timestamp_ms: int) -> str | None:
        for clip_id, start, end in ordered:
            if start <= timestamp_ms <= end:
                return clip_id
        if ordered:
            return ordered[0][0] if timestamp_ms < ordered[0][1] else ordered[-1][0]
        return None

    return resolve


def _segment_payload(segment: Any) -> dict[str, Any]:
    return {
        "video_id": getattr(segment, "video_id", ""),
        "index_no": int(getattr(segment, "index_no", None) or getattr(segment, "index", 0) or 0),
        "start_ms": int(getattr(segment, "start_ms", 0) or 0),
        "end_ms": int(getattr(segment, "end_ms", 0) or 0),
        "text": str(getattr(segment, "text", "") or ""),
        "language": str(getattr(segment, "language", "") or ""),
        "confidence": float(
            getattr(segment, "confidence", None)
            or getattr(segment, "avg_logprob", 0.0)
            or 0.0
        ),
        "source": str(getattr(segment, "source", "asr") or "asr"),
    }


def _clip_index(clip: Any) -> int:
    """``Clip`` 的序号字段历史上叫过 ``index_no``，现在是 ``index``。"""
    return int(getattr(clip, "index", None) or getattr(clip, "index_no", 0) or 0)


def _clip_payload(clip: Any) -> dict[str, Any]:
    return {
        "clip_id": str(getattr(clip, "clip_id", "") or ""),
        "video_id": getattr(clip, "video_id", ""),
        "index_no": _clip_index(clip),
        "start_ms": int(getattr(clip, "start_ms", 0) or 0),
        "end_ms": int(getattr(clip, "end_ms", 0) or 0),
        "core_start_ms": int(getattr(clip, "core_start_ms", 0) or 0),
        "core_end_ms": int(getattr(clip, "core_end_ms", 0) or 0),
        "text": str(getattr(clip, "text", "") or ""),
        "core_text": str(getattr(clip, "core_text", "") or ""),
        "truncated": bool(getattr(clip, "truncated", False)),
    }


# =============================================================================
# 断点恢复：把库里的记录还原成内存对象
# =============================================================================


def _restore_transcript(video_id: str, records: Sequence[Any]) -> Any:
    """库里的 ``SegmentRecord`` → ``Transcript``，供断点续跑复用转写结果。"""
    from core.models.asr import Transcript, TranscriptSegment  # noqa: PLC0415

    segments = tuple(
        TranscriptSegment(
            index=int(getattr(r, "index_no", 0) or 0),
            start_ms=int(getattr(r, "start_ms", 0) or 0),
            end_ms=int(getattr(r, "end_ms", 0) or 0),
            text=str(getattr(r, "text", "") or ""),
        )
        for r in records
    )
    first = records[0] if records else None
    source = str(getattr(first, "source", "asr") or "asr")
    return Transcript(
        video_id=video_id,
        segments=segments,
        language=str(getattr(first, "language", "") or ""),
        duration_ms=segments[-1].end_ms if segments else 0,
        source="subtitle" if source == "subtitle" else "whisper",
        model_name="(断点恢复)",
        warnings=("转写来自库中已有片段，未重新识别",),
    )


def _restore_clips(records: Sequence[Any]) -> list[Any]:
    """库里的 ``ClipRecord`` → ``Clip``。"""
    from core.rag.chunker import Clip  # noqa: PLC0415

    return [
        Clip(
            clip_id=str(getattr(r, "clip_id", "") or ""),
            video_id=str(getattr(r, "video_id", "") or ""),
            index=int(getattr(r, "index_no", 0) or 0),
            start_ms=int(getattr(r, "start_ms", 0) or 0),
            end_ms=int(getattr(r, "end_ms", 0) or 0),
            core_start_ms=int(getattr(r, "core_start_ms", 0) or 0),
            core_end_ms=int(getattr(r, "core_end_ms", 0) or 0),
            text=str(getattr(r, "text", "") or ""),
            core_text=str(getattr(r, "core_text", "") or ""),
            keyframe_ids=tuple(getattr(r, "keyframe_ids", ()) or ()),
            truncated=bool(getattr(r, "truncated", False)),
        )
        for r in records
    ]


def _restore_keyframes(records: Sequence[Any]) -> list[Any]:
    """库里的 ``KeyframeRecord`` → ``Keyframe``。"""
    from pathlib import Path as _Path  # noqa: PLC0415

    from core.media.scene_detect import Keyframe  # noqa: PLC0415

    restored: list[Any] = []
    for r in records:
        path = str(getattr(r, "path", "") or "")
        if not path or not _Path(path).is_file():
            continue
        restored.append(
            Keyframe(
                keyframe_id=str(getattr(r, "keyframe_id", "") or ""),
                video_id=str(getattr(r, "video_id", "") or ""),
                timestamp_ms=int(getattr(r, "timestamp_ms", 0) or 0),
                path=_Path(path),
                width=int(getattr(r, "width", 0) or 0),
                height=int(getattr(r, "height", 0) or 0),
                size_bytes=int(getattr(r, "size_bytes", 0) or 0),
                scene_index=int(getattr(r, "scene_index", 0) or 0),
                clip_id=str(getattr(r, "clip_id", "") or ""),
                sharpness=float(getattr(r, "sharpness", 0.0) or 0.0),
                sharpness_ok=bool(getattr(r, "sharpness_ok", True)),
                quality_ok=bool(getattr(r, "quality_ok", True)),
                requested_ms=int(getattr(r, "timestamp_ms", 0) or 0),
            )
        )
    return restored


def _item_keyframe_ids(item: Any) -> tuple[str, ...]:
    """从 :class:`FusedItem` 或切片 payload dict 里取关键帧 ID。"""
    raw = (
        item.get("keyframe_ids")
        if isinstance(item, Mapping)
        else getattr(item, "keyframe_ids", ())
    )
    if not raw:
        return ()
    if isinstance(raw, str):
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    try:
        return tuple(str(part) for part in raw)
    except TypeError:
        return ()


def _item_text(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item.get("core_text") or item.get("text") or "")
    return str(getattr(item, "text", "") or "")


def _item_with_text(item: Any, text: str) -> Any:
    """返回替换了 text 的副本；对象不可变替换时原样返回。"""
    if isinstance(item, dict):
        clone = dict(item)
        clone["text"] = text
        return clone
    try:
        return _dc_replace(item, text=text)
    except TypeError:
        return item


def _ms_label(value: Any) -> str:
    """毫秒 → ``HH:MM:SS.mmm``（溯源铁律要求毫秒级）。"""
    try:
        ms = max(0, int(value))
    except (TypeError, ValueError):
        ms = 0
    seconds, millis = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _kf_ms(keyframe: Any) -> int:
    try:
        return max(0, int(getattr(keyframe, "timestamp_ms", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _kf_scene(keyframe: Any) -> int:
    try:
        return int(getattr(keyframe, "scene_index", -1) or 0)
    except (TypeError, ValueError):
        return 0


_MILESTONE_RE = re.compile(
    r"^\s*[-*•·]*\s*(?:\d+[.、)）]\s*)?"
    r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*[,，、)）]?\s*[—\-：:]?\s*(\S.*)$"
)


def _parse_milestones(
    text: str, window_start_ms: int, window_end_ms: int
) -> list[tuple[int, str]]:
    """解析 LLM 的里程碑行 ``HH:MM:SS 说明``，只收窗口内的时间。

    时间落在章节窗口 ±60s 之外的行直接丢弃 —— 模型偶尔会抄示例里的
    "00:00:00" 或编一个超界时间，宁缺毋滥。两段式时间（MM:SS）也接受。
    """
    out: list[tuple[int, str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("格式"):
            continue
        match = _MILESTONE_RE.match(line)
        if not match:
            continue
        h, m, s, rest = match.groups()
        try:
            hh = int(h)
            mm = int(m)
            ss = int(s or 0)
        except ValueError:
            continue
        # 两种格式：HH:MM:SS（三段）或 M:SS（两段，"时"位其实是分）
        if s is None:
            total_ms = hh * 60_000 + mm * 1_000
        else:
            total_ms = ((hh * 60 + mm) * 60 + ss) * 1000
        if not (window_start_ms - 60_000 <= total_ms <= window_end_ms + 60_000):
            continue
        note = rest.strip().strip("。；;")[:60]
        if note:
            out.append((total_ms, note))
    out.sort(key=lambda item: item[0])
    return out


def _merge_scenes(scenes: Sequence[Any], span_ms: int) -> list[Any]:
    """D3：把相邻场景合并到 ``>= span_ms`` 的窗口（高运动降档用）。

    返回带 index/start_ms/end_ms/duration_ms/midpoint_ms 的轻量对象，
    满足 :meth:`SceneDetector.extract_keyframes` 对 scenes 的全部取用。
    """
    from types import SimpleNamespace  # noqa: PLC0415

    merged: list[list[int]] = []
    cur: list[int] | None = None
    for scene in scenes:
        start = int(getattr(scene, "start_ms", 0) or 0)
        end = int(getattr(scene, "end_ms", 0) or 0)
        if cur is None:
            cur = [start, end]
        else:
            cur[1] = max(cur[1], end)
        if cur[1] - cur[0] >= span_ms:
            merged.append(cur)
            cur = None
    if cur:
        merged.append(cur)

    out: list[Any] = []
    total_end = max((int(getattr(s, "end_ms", 0) or 0) for s in scenes), default=0)
    for i, (start, end) in enumerate(merged):
        out.append(SimpleNamespace(
            index=i, start_ms=start, end_ms=end,
            duration_ms=end - start, midpoint_ms=(start + end) // 2,
            detector="merged",
        ))
    if out and total_end and out[-1].end_ms < total_end:
        out[-1].end_ms = total_end
        out[-1].duration_ms = total_end - out[-1].start_ms
    return out


def _segment_chunks(scenes: Sequence[Any], max_s: int) -> list[list[int]]:
    """P1：把连续场景聚成 ``<= max_s`` 秒的时间窗，返回场景下标分组。

    供视频段理解使用 —— 段长决定上传体积与模型可看的运动时长，
    超过 API 限制的段会被服务端拒绝，这里在源头剪好。
    """
    limit_ms = max(1, int(max_s)) * 1000
    chunks: list[list[int]] = []
    cur: list[int] = []
    cur_start = 0
    for i, scene in enumerate(scenes):
        start = int(getattr(scene, "start_ms", 0) or 0)
        end = int(getattr(scene, "end_ms", 0) or 0)
        if cur and end - cur_start > limit_ms:
            chunks.append(cur)
            cur = []
            cur_start = start
        if not cur:
            cur_start = start
        cur.append(i)
    if cur:
        chunks.append(cur)
    return chunks


def _group_by_scene(keyframes: Sequence[Any], limit: int) -> list[list[Any]]:
    """按场景切分关键帧，每组最多 ``limit`` 帧（首尾优先）。
    做法：场景是天然的"镜头边界"，一个场景内抽的 2 帧讲的是同一件事，
    放进一次推理模型能看出"这段时间画面怎么变"。跨场景硬凑在一起反而
    会让它把两个场景讲成一个。
    """
    from core.models.vl import evenly_spaced  # noqa: PLC0415

    groups: list[list[Any]] = []
    current: list[Any] = []
    current_scene: int | None = None
    for kf in keyframes:
        sidx = _kf_scene(kf)
        if current_scene is None or sidx == current_scene:
            current.append(kf)
        else:
            groups.append(current)
            current = [kf]
        current_scene = sidx
    if current:
        groups.append(current)
    size = max(1, int(limit))
    return [evenly_spaced(group, size) for group in groups]


def _keyframe_payload(keyframe: Any, description: str, ocr: str) -> dict[str, Any]:
    return {
        "keyframe_id": getattr(keyframe, "keyframe_id", ""),
        "video_id": getattr(keyframe, "video_id", ""),
        "clip_id": getattr(keyframe, "clip_id", None),
        "scene_index": int(getattr(keyframe, "scene_index", 0) or 0),
        "timestamp_ms": int(getattr(keyframe, "timestamp_ms", 0) or 0),
        "path": str(getattr(keyframe, "path", "") or ""),
        "width": int(getattr(keyframe, "width", 0) or 0),
        "height": int(getattr(keyframe, "height", 0) or 0),
        "size_bytes": int(getattr(keyframe, "size_bytes", 0) or 0),
        "sharpness": float(getattr(keyframe, "sharpness", 0.0) or 0.0),
        "sharpness_ok": bool(getattr(keyframe, "sharpness_ok", True)),
        "quality_ok": bool(getattr(keyframe, "quality_ok", True)),
        "description": description,
        "ocr_text": ocr,
    }


def _synthetic_clips(video_id: str, duration_ms: int, span_ms: int) -> list[Any]:
    """无转录时按固定跨度造切片，保证视觉链路有落点。"""
    from core.rag.chunker import Clip  # noqa: PLC0415

    total = max(0, int(duration_ms or 0))
    if total <= 0:
        return []
    span = max(5_000, int(span_ms or 20_000))
    clips: list[Clip] = []
    index = 0
    cursor = 0
    while cursor < total:
        end = min(total, cursor + span)
        clips.append(
            Clip(
                clip_id=f"{video_id}-c{index:04d}",
                video_id=video_id,
                index=index,
                start_ms=cursor,
                end_ms=end,
                core_start_ms=cursor,
                core_end_ms=end,
                text="",
                core_text="",
            )
        )
        index += 1
        cursor = end
    return clips


def _enforce_answer_claims(result: AskResult, built: Any) -> str:
    """校验回答里的引用编号，返回清洗后的正文。

    ``claims.validate_claims`` 只关心"自由文本 + 证据列表"，不碰
    :class:`AskResult` 的结构，所以这里只做搬运：

    - 悬空编号（指向不存在证据的 ``[3]``）撕掉 —— 实测模型会顺着
      "01/02/03" 的惯性编出画面里根本没有的编号；
    - 孤立陈述句补一条最相近的出处（重合度够高才补）；
    - 每条问题都记进 ``notes``，由 API 层原样透出，不吞掉。

    校验本身失败时保留原回答 —— 宁可少一层保险，也不能因为校验崩掉
    让用户拿不到答案。
    """
    if not result.answer.strip():
        return result.answer
    try:
        from core.pipeline.claims import enforce_claims  # noqa: PLC0415

        cleaned, report = enforce_claims(result.answer, built.refs)
    except Exception as exc:  # noqa: BLE001 - 校验失败不该影响回答
        logger.debug("证据校验跳过（%s）", exc)
        return result.answer

    for issue in report.issues:
        result.notes.append(f"证据校验: {issue}")
    for fix in report.fixes:
        result.notes.append(f"证据校验: {fix}")
    if not report.ok:
        # 只有"没被修掉"的问题才标降级；自动补上的引用不算降级
        result.degraded = True
    return cleaned


def _evidence_only_answer(built: Any) -> str:
    """LLM 不可用时的回答：直接列出证据，不做改写，但保留引用编号。"""
    lines = ["（当前没有可用的语言模型，以下为检索到的原始证据）", ""]
    for item in built.items:
        lines.append(f"[{item.index}] {item.time_range}（{item.clip_id}）")
        lines.append(item.text or "（该片段无转录文本）")
        lines.append("")
    return "\n".join(lines).strip()


_REF_STRIP = re.compile(r"[\[［][0-9][0-9,，、\-\s]*[\]］]")
_QA_PREFIX = re.compile(r"^(?:\d{1,2}[.、)）]|[-*•·]|\((?:\d{1,2})\))\s*")


def _strip_chapter_refs(text: str) -> str:
    """抹掉章节摘要里的引用编号。

    章节正文里的 ``[1]`` 是 ``build_summary`` 给片段临时编的号，调用方拿不到
    那张映射表 —— 留着它读者以为是可点的引用，点开却是空。顺手清掉清理后
    残留的标点前空格。
    """
    cleaned = _REF_STRIP.sub("", text)
    # 编号删掉后会留下"  "这样的空洞，中文句子里读起来像错字；只收拾中文
    # 标点两侧与连续空格，英文句号后的空格保留。
    cleaned = re.sub(r"([。，；：！？、）」』】\n])[ \t]+", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+([。，；：！？、）」』】])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _flatten_qa_summary(text: str) -> str:
    """把"问答清单"折回叙述。

    本地小模型看到几条并列要点时，常自作主张改写成 ``1. 某问题？`` + 答案。
    那些问句是模型自己编的（视频里没人这么问），留着读者会以为是被采访的
    原文。这里丢掉问句、只留答案，再拼成一段；只有一条内容时原样返回。
    """
    answers: list[str] = []
    dropped = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        body = _QA_PREFIX.sub("", line).strip()
        if not body:
            continue
        if body.endswith(("？", "?")):
            dropped += 1
            continue
        answers.append(body)
    if dropped <= 0 or len(answers) <= 1:
        return text.strip()
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in "".join(answers))
    joiner = "" if has_cjk else " "
    return joiner.join(answers)


def _chapter_fallback(items: Sequence[Any], *, limit: int = 3) -> str:
    """模型不可用时的章节摘要：按时间取前几段原文拼接，不编造。"""
    picked = list(items)[:limit]
    if not picked:
        return ""
    parts: list[str] = []
    for item in picked:
        text = _item_text(item).strip()
        if not text:
            continue
        stamp = _ms_label(_get_field(item, "start_ms"))
        parts.append(f"{stamp} {text[:200]}")
    return "；".join(parts)


def _unlink_all(paths: Sequence[str]) -> int:
    """删掉给定文件，返回成功个数。单个失败只记日志 —— 清理阶段不该被打断。"""
    gone = 0
    for raw in paths:
        try:
            Path(raw).unlink(missing_ok=True)
            gone += 1
        except OSError:
            logger.debug("删除文件失败: %s", raw)
    return gone


def _unlink_owned(path: Path, roots: Sequence[Path]) -> bool:
    """仅当文件位于给定根目录之下才删除。

    用户在界面里拖进来的视频是自己的素材，删记录不能连原件一起删；
    只有下载缓存在 ``data/videos``、``data/temp`` 里的副本才归系统管。
    """
    try:
        target = path.resolve()
    except OSError:
        return False
    for root in roots:
        try:
            if target.is_relative_to(root.resolve()) and target.is_file():
                target.unlink()
                return True
        except (OSError, ValueError):
            continue
    return False


def _get_field(item: Any, name: str, default: Any = 0) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


# =============================================================================
# 组件装配（延迟导入，缺依赖即抛给 build() 记录降级）
# =============================================================================


def _build_vectors(cfg: AppConfig) -> Any:
    from core.rag.vector_store import VectorStore  # noqa: PLC0415

    return VectorStore.from_config(cfg)


def _build_embedder(cfg: AppConfig, manager: Any) -> Any:
    from core.models.embed import EmbeddingService  # noqa: PLC0415

    # 注意 manager 是 keyword-only，位置传会 TypeError
    return EmbeddingService.from_config(cfg, manager=manager)


def _build_asr(cfg: AppConfig, manager: Any) -> Any:
    from core.models.asr import AsrConfig, AsrEngine  # noqa: PLC0415

    return AsrEngine(AsrConfig.from_config(cfg), model_manager=manager)


def _build_scene(cfg: AppConfig) -> Any:
    from core.media.scene_detect import SceneDetectConfig, SceneDetector  # noqa: PLC0415

    # flattened() 里有 SceneDetectConfig 不认识的键（如 keyframe.format），过滤掉
    payload = cfg.media.scene_detect.flattened()
    known = set(SceneDetectConfig.__dataclass_fields__)
    return SceneDetector(SceneDetectConfig(**{k: v for k, v in payload.items() if k in known}))


def _build_chunker(cfg: AppConfig) -> Any:
    from core.rag.chunker import ChunkConfig, Chunker  # noqa: PLC0415

    return Chunker(ChunkConfig(**cfg.chunk.as_chunk_config()))


def _build_vl(cfg: AppConfig, manager: Any) -> Any:
    from core.models.vl import VlService  # noqa: PLC0415

    return VlService.from_config(cfg, manager)


def _build_llm(cfg: AppConfig, manager: Any) -> Any:
    from core.models.llm import LlmGateway  # noqa: PLC0415

    return LlmGateway.from_config(cfg, manager)


def _build_context(cfg: AppConfig) -> Any:
    from core.pipeline.context_builder import ContextBudget, ContextBuilder  # noqa: PLC0415
    from core.prompts import get_prompt_registry  # noqa: PLC0415

    budget = ContextBudget(
        max_tokens=cfg.retrieval.context_max_tokens
        if hasattr(cfg.retrieval, "context_max_tokens")
        else 4000,
        max_chars=cfg.retrieval.max_clip_chars * 6,
        max_items=cfg.retrieval.top_k + 2,
    )
    # 必须挂上提示词注册表：不挂的话 build_summary/qa_answer 全走内置兜底
    # 文案，configs/prompts.yaml 里精修的模板（含防注入声明）形同虚设 ——
    # 这正是"总结又短又干"的隐藏原因之一。
    try:
        return ContextBuilder(budget, prompts=get_prompt_registry())
    except Exception:  # noqa: BLE001 - 预算非法时退回默认
        logger.debug("ContextBuilder 构造失败，退回默认预算", exc_info=True)
        return ContextBuilder(None, prompts=get_prompt_registry())


def _build_downloader(cfg: AppConfig) -> Any:
    from core.media.downloader import DownloadConfig, MediaDownloader  # noqa: PLC0415

    return MediaDownloader(DownloadConfig.from_config(cfg))
