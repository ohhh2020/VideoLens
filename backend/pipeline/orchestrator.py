"""分析任务、问答调度、断点状态和生命周期。具体阶段见 stages/。"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from backend.concurrency import submit, parallel_pair
from backend.logging_setup import log_task
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Sequence

from backend.config import AppConfig, get_app_config
from backend.pipeline.factory import _build_llm, _build_context

from backend.pipeline.domain import (
    AnalysisReport,
    AnalyzeOptions,
    AskResult,
    PipelineError,
    Stage,
    StageReport,
    _INFERENCE_LOCK,
    _INFERENCE_POLL_S,
    _STAGE_LABEL,
    _STAGE_ORDER,
    _STAGE_PROGRESS,
    _STAGE_STATUS,
    _video_lock,
    compute_fingerprint,
    extract_citations,
)
from backend.pipeline.helpers import (
    _clip_payload,
    _enforce_answer_claims,
    _restore_clips,
    _restore_keyframes,
    _restore_transcript,
    _segment_payload,
    _unlink_all,
    _unlink_owned,
)

logger = logging.getLogger("zhiying.pipeline")

from backend.pipeline.stages.media import MediaStages
from backend.pipeline.stages.segmentation import SegmentationStages
from backend.pipeline.stages.vision import VisionStages
from backend.pipeline.stages.indexing import IndexingStages
from backend.pipeline.stages.retrieval import RetrievalStages
from backend.pipeline.stages.reports import ReportStages
from backend.pipeline.domain import Citation


class Orchestrator(
    MediaStages,
    SegmentationStages,
    VisionStages,
    IndexingStages,
    RetrievalStages,
    ReportStages,
):
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
        memory: Any = None,
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
        self._memory = memory
        # BM25 通道缓存：按 (video_id, 切片数, 末片ID) 记账，切片一变自然失效。
        # 不缓存的话每次提问都要对整库切片重新分词（1 小时视频约 180 片）。
        self._bm25_cache: OrderedDict[str, tuple[tuple[str, int, str], Any]] = (
            OrderedDict()
        )
        self._bm25_lock = threading.Lock()

    @property
    def repository(self) -> Any:
        """元数据仓储（列出视频、看详情、统计容量）。"""
        return self._repo

    def _preference_messages(self, messages, query, *, report=False):
        if self._memory is not None:
            messages = self._memory.messages(query, messages, report=report)
        if report:
            from backend.memory.service import REPORT_GUIDANCE

            messages = [{"role": "system", "content": REPORT_GUIDANCE}, *messages]
        return messages

    def _notify_memory(self):
        if self._memory is not None and not getattr(self, "_suppress_memory_notify", False):
            self._memory.notify()

    def _require_cloud_ready(self):
        pending = getattr(self, "_cloud_setup_required", ())
        if pending:
            raise PipelineError("请先在网页设置中填写" + "、".join(pending) + "的服务地址、模型名和 API Key，保存后通过菜单重启服务")

    @property
    def vectors(self) -> Any:
        """必需的向量库；装配失败会阻止服务启动。"""
        return self._vectors

    @property
    def config(self) -> AppConfig:
        return self._cfg

    def analyze(
        self, source: str, options: AnalyzeOptions | None = None
    ) -> AnalysisReport:
        """分析一个视频（本地路径或 URL）。**不抛异常**，失败写进 report。"""
        with log_task(
            f"analyze:{getattr(options, 'task_id', '') or Path(source).name[:40]}"
        ):
            if getattr(self, "_use_integrity", False) and self._cfg.integrity.enabled:
                from backend.analysis.runs import execute
                return execute(self, source, options or AnalyzeOptions())
            return self._analyze(source, options)

    def _analyze(self, source, options=None):
        opts = options or AnalyzeOptions()
        from backend.analysis.optimized import enabled, analyze
        if enabled(self):
            return analyze(self, source, opts)
        opts.validate()
        started = time.monotonic()
        report = AnalysisReport(duration_ms=0)
        report.degraded.extend(getattr(self, "_assembly_notes", []) or [])
        progress = opts.progress or self._on_progress

        video_id = ""
        ctx: dict[str, Any] = {}
        video_guard: threading.Lock | None = None
        workers = None
        scene_future = None
        scene_stage = None

        try:
            # ---------------- parse ----------------
            self._require_cloud_ready()
            stage = self._begin(report, Stage.PARSE, progress)
            media_source = self._parse_source(source, opts)
            video_id = media_source.video_id
            report.video_id = video_id
            report.title = opts.title or media_source.display_name
            ctx["source"] = media_source

            # 同视频串行：后来者等先到者完成后走快路径（复用结果/断点续跑），
            # 绝不并行跑两条全量流水线。见 _VIDEO_LOCKS 的注释。
            video_guard = None if getattr(self, "_isolated_run", False) else _video_lock(video_id)
            if video_guard is not None and not video_guard.acquire(timeout=0.5):
                logger.info("video_id=%s 已有分析在运行，排队等待其完成", video_id)
                report.warnings.append("同视频已有任务在运行，已排队等待（未重复计算）")
                video_guard.acquire()

            existing = self._repo.find_video(video_id)
            if (
                existing
                and existing.status == "ready"
                and not opts.force
                and existing.fingerprint == compute_fingerprint(self._cfg)
            ):
                self._finish(stage, message="视频已分析完成，跳过（force=True 可重跑）")
                saved_report = self._repo.get_video_report(video_id)
                if saved_report:
                    for name in AnalysisReport.__dataclass_fields__:
                        if (
                            name not in {"stages", "elapsed_ms"}
                            and name in saved_report
                        ):
                            setattr(report, name, saved_report[name])
                report.status = "ready"
                report.clip_count = len(self._repo.get_clips(video_id))
                report.keyframe_count = len(self._repo.get_keyframes(video_id))
                report.elapsed_ms = int((time.monotonic() - started) * 1000)
                report.stages.append(
                    StageReport(
                        stage=Stage.DONE, skipped=True, message="已完成，未重跑"
                    )
                )
                self._end(report, Stage.DONE, progress)
                # 老视频（v5 之前分析的）没有落库报告：趁机重建总结并补存，
                # 让「已分析完成，跳过」路径也能在 /ui 视频库看到完整报告。
                try:
                    if self._repo.get_video_report(video_id) is None:
                        if not opts.skip_summary:
                            report.summary, report.summary_backend = self._summarize(
                                video_id,
                                report.title or existing.title,
                                report.duration_ms or existing.duration_ms,
                                opts,
                            )
                        self._repo.set_video_report(video_id, report.to_dict())
                except Exception:  # noqa: BLE001 - 补存失败不影响"已完成"
                    logger.warning(
                        "旧视频报告补存失败 video_id=%s", video_id, exc_info=True
                    )
                return report
            # 断点必须在 _register_video **之前**读：注册会 upsert 视频行，
            # 把上一轮失败的 stage 覆写成空，恢复点就永远读不到了。
            resume_from = self._resume_point(video_id, opts)
            resume_ctx = (
                self._resume_context(video_id) if resume_from is not None else {}
            )
            self._register_video(media_source, opts)
            # 指纹不一致 → 旧切片/向量必须先清，否则新旧产物混在库里
            stale = self._apply_fingerprint(
                report, video_id, compute_fingerprint(self._cfg)
            )
            if stale:
                # 指纹变了说明产物已全部作废，刚体检通过的缓存也不能再用
                resume_from, resume_ctx = None, {}
            self._finish(
                stage, message=f"{media_source.kind}: {media_source.display_name}"
            )

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

            if self._cfg.limits.pipeline_parallel:
                workers = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="scene-detect"
                )
                scene_stage = StageReport(stage=Stage.SCENE)

                def detect_scene():
                    scene_stage.started_at = time.monotonic()
                    try:
                        return self._detect_scenes(local_path, probe_info)
                    finally:
                        # 主流程接收 future 时统一记录完成，避免同一次检测打印两次。
                        scene_stage.elapsed_ms = max(
                            0, int((time.monotonic() - scene_stage.started_at) * 1000)
                        )

                scene_future = submit(workers, detect_scene)

            # ---------------- audio + transcribe ----------------
            stage = self._begin(report, Stage.AUDIO, progress)
            audio_path, has_audio = self._extract_audio(
                local_path, video_id, probe_info
            )
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
                transcript = self._transcribe(
                    video_id, audio_path, local_path, probe_info, opts
                )
            if getattr(self, '_analysis_run', None) is not None and not reused:
                self._analysis_run.put('transcript_raw', {'source': transcript.source, 'language': transcript.language,
                    'segments': [segment.to_dict() for segment in transcript.segments]})
            ctx["transcript"] = transcript
            report.transcript_source = str(getattr(transcript, "source", "") or "")
            report.language = str(getattr(transcript, "language", "") or "")
            segments = list(getattr(transcript, "segments", []) or [])
            report.segment_count = len(segments)
            transcript_warnings = [
                str(w) for w in (getattr(transcript, "warnings", ()) or ()) if w
            ]
            report.warnings.extend(
                w for w in transcript_warnings if w not in report.warnings
            )
            if segments and not reused:
                self._repo.add_segments(
                    video_id, [_segment_payload(s) for s in segments]
                )
            if reused:
                stage.skipped = True
                report.warnings.append("转写复用库中已有结果，未重新识别")
            self._finish(
                stage,
                message=(
                    f"{len(segments)} 段，来源 {report.transcript_source or 'asr'}"
                    + ("（断点恢复）" if reused else "")
                    + (
                        ("；" + "；".join(transcript_warnings))
                        if not segments and transcript_warnings
                        else ""
                    )
                ),
                degraded=not segments,
            )
            if not segments:
                reason = "；".join(transcript_warnings) or "未检测到可用人声或字幕"
                report.degraded.append(
                    f"未生成转写文字：{reason}；切片仅依赖视觉与时长"
                )

            # ---------------- scene ----------------
            stage = self._begin(
                report, Stage.SCENE, progress, parallel_result=scene_future is not None
            )
            if scene_future is not None:
                scenes, detector_used, scene_warnings = scene_future.result()
                stage.started_at = scene_stage.started_at
                stage.elapsed_ms = scene_stage.elapsed_ms
            else:
                scenes, detector_used, scene_warnings = self._detect_scenes(
                    local_path, probe_info
                )
            ctx["scenes"] = scenes
            report.warnings.extend(scene_warnings)
            self._finish(
                stage,
                message=f"{len(scenes)} 个场景（{detector_used}）"
                + ("；复用与转写并行计算的结果，未再次检测" if scene_future is not None else ""),
                elapsed_ms=scene_stage.elapsed_ms if scene_stage else None,
            )

            # 内容画像（方案 A 最小实现）：高运动判定驱动视觉分派
            duration_s = max(1.0, (report.duration_ms or 0) / 1000.0)
            scene_density = len(scenes) / (duration_s / 60.0)
            speech_ms = sum(
                max(
                    0,
                    int(getattr(s, "end_ms", 0) or 0)
                    - int(getattr(s, "start_ms", 0) or 0),
                )
                for s in segments
            )
            high_motion = scene_density >= float(
                self._cfg.vl.high_motion_density_per_min
            )
            profile = {
                "scene_count": len(scenes),
                "scene_density_per_min": round(scene_density, 1),
                "speech_ratio": round(
                    min(1.0, speech_ms / max(1, report.duration_ms or 1)), 3
                ),
                "transcript_source": report.transcript_source,
                "high_motion": high_motion,
            }
            report.profile = profile
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
                stage.skipped = True
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
            encoded = None
            if not reused:
                keyframes, visual_scenes = self._prepare_keyframes(
                    local_path,
                    video_id,
                    scenes,
                    clips,
                    opts,
                    report,
                    high_motion=high_motion,
                )
                profile["frame_extract_ms"] = int(
                    (time.monotonic() - stage.started_at) * 1000
                )

                def describe():
                    started_vl = time.monotonic()
                    result = self._describe_and_store(
                        local_path,
                        video_id,
                        keyframes,
                        visual_scenes,
                        opts,
                        report,
                        high_motion,
                    )
                    profile["cloud_vision_ms"] = int(
                        (time.monotonic() - started_vl) * 1000
                    )
                    return result

                def encode_inputs():
                    started_encode = time.monotonic()
                    try:
                        return self._encode_index_inputs(clips, keyframes, report, opts)
                    except Exception as exc:
                        # 等待视觉 worker 正常收尾后，在索引阶段严格上报模型故障。
                        # 不让并发编码错误被误记为云端视觉失败。
                        return {
                            "error": exc,
                            "text_ms": 0,
                            "image_ms": 0,
                            "elapsed_ms": int(
                                (time.monotonic() - started_encode) * 1000
                            ),
                        }

                keyframes, encoded = parallel_pair(
                    describe,
                    encode_inputs,
                    enabled=self._cfg.limits.pipeline_parallel,
                    prefix="vision-index",
                )
            profile.setdefault("vision_mode_used", "reused" if reused else "")
            report.profile = profile
            ctx["keyframes"] = keyframes
            report.keyframe_count = len(keyframes)
            if reused:
                stage.skipped = True
                report.warnings.append("关键帧复用库中已有结果与描述，未重新抽帧")
            self._finish(
                stage,
                message=f"{len(keyframes)} 张关键帧"
                + ("（断点恢复）" if reused else ""),
                elapsed_ms=None
                if reused
                else profile["frame_extract_ms"] + profile["cloud_vision_ms"],
                degraded=not keyframes or opts.skip_vision,
            )

            # ---------------- index ----------------
            stage = self._begin(report, Stage.INDEX, progress)
            if encoded is None and not opts.skip_vectors:
                encoded = self._encode_index_inputs(clips, keyframes, report, opts)
            index_write_start = time.monotonic()
            if encoded:
                stage.started_at = index_write_start - encoded["elapsed_ms"] / 1000
                profile.update(
                    {
                        "text_encode_ms": encoded["text_ms"],
                        "image_encode_ms": encoded["image_ms"],
                        "encode_wall_ms": encoded["elapsed_ms"],
                    }
                )
            vectors_written = self._index(
                video_id, clips, keyframes, report, opts, encoded=encoded
            )
            index_write_ms = int((time.monotonic() - index_write_start) * 1000)
            if encoded:
                profile.update(
                    {
                        "text_encode_ms": encoded["text_ms"],
                        "image_encode_ms": encoded["image_ms"],
                        "encode_wall_ms": encoded["elapsed_ms"],
                        "index_write_ms": index_write_ms,
                    }
                )
            report.vector_count = vectors_written
            self._finish(
                stage,
                message=f"{vectors_written} 条向量",
                elapsed_ms=index_write_ms + (encoded["elapsed_ms"] if encoded else 0),
                degraded=vectors_written == 0 and not opts.skip_vectors,
            )

            # ---------------- summary ----------------
            stage = self._begin(report, Stage.SUMMARY, progress)
            summary, backend = self._summarize(
                video_id, report.title, report.duration_ms, opts
            )
            report.summary = summary
            report.summary_backend = backend
            self._finish(
                stage,
                message=f"{len(summary)} 字符（{backend}）",
                degraded=backend == "keywords",
            )

            # ---------------- done ----------------
            self._repo.update_video_status(
                video_id, "ready", stage="ready", progress=1.0, finished=True
            )
            if task_id:
                self._repo.update_task(
                    task_id,
                    status="done",
                    stage="done",
                    progress=1.0,
                    message="分析完成",
                )
            report.status = "ready"
            self._end(report, Stage.DONE, progress)

        except Exception as exc:  # noqa: BLE001 - 顶层兜底
            message = f"{type(exc).__name__}: {exc}"
            logger.exception("分析失败 video_id=%s", video_id or "?")
            report.status = "failed"
            report.errors.append(message)
            if report.stages and report.stages[-1].stage is not Stage.DONE:
                self._finish(report.stages[-1], message=message)
                report.stages[-1].ok = False
                report.stages[-1].error = message
                report.stages[-1].message = message
            report.stages.append(StageReport(stage=Stage.DONE, ok=False, error=message))
            if video_id:
                try:
                    self._repo.update_video_status(
                        video_id, "failed", stage="failed", error=message, finished=True
                    )
                except Exception:  # noqa: BLE001
                    logger.debug("记录失败状态时再次出错", exc_info=True)
        finally:
            if workers is not None:
                workers.shutdown(wait=True, cancel_futures=True)
            report.elapsed_ms = int((time.monotonic() - started) * 1000)
            if video_id:
                try:
                    # 缓存命中不覆盖上一轮完整分析的报告，只记录本次复用耗时。
                    if not any(
                        s.skipped and s.stage == Stage.DONE for s in report.stages
                    ):
                        self._repo.set_video_report(video_id, report.to_dict())
                except Exception:
                    logger.warning(
                        "分析报告落库失败 video_id=%s", video_id, exc_info=True
                    )
            self._log_timings(report)
            if video_guard is not None:
                video_guard.release()
            if report.status == "ready":
                self._notify_memory()

        return report

    def _log_timings(self, report):
        cached = any(
            item.stage == Stage.DONE and item.skipped for item in report.stages
        )
        rows = [f"分析耗时汇总 video={report.video_id or '?'} status={report.status}"]
        parallel = report.profile.get('pipeline') == 'audio||decode->vision;summary||index'
        if parallel and not cached:
            rows.append("  并行流程：音频/转写 与 抽帧→视觉/OCR并行；文本总结 与 向量索引并行")
            reused = report.profile.get('transcript_reused')
            if reused:
                rows.append("  转写复用：" + {'content-cache': '音频内容缓存', 'run-checkpoint': '本次运行检查点',
                                             'matched-subtitle': '匹配字幕'}.get(reused, reused))
        if cached:
            rows.append(
                "  本次复用已有分析结果，仅统计本次读取耗时，历史分析报告保持不变"
            )
        for item in report.stages:
            state = "失败" if not item.ok else "复用" if item.skipped else "完成"
            rows.append(
                f"  {_STAGE_LABEL[item.stage]:<12} {item.elapsed_ms / 1000:8.3f}s  {state}"
            )
        for key, label in (
            ("frame_extract_ms", "抽帧"),
            ("cloud_vision_ms", "云端视觉/OCR"),
            ("text_encode_ms", "BGE 文本编码"),
            ("image_encode_ms", "SigLIP 图像编码"),
            ("index_write_ms", "索引写入"),
        ):
            if not cached and key in report.profile:
                if parallel:
                    label = {'frame_extract_ms': '抽帧（含队列等待）',
                             'cloud_vision_ms': '取帧＋视觉/OCR墙钟'}.get(key, label)
                rows.append(f"  {label:<12} {report.profile[key] / 1000:8.3f}s")
        rows.append(
            f"  总耗时（实际经过） {report.elapsed_ms / 1000:.3f}s；并发阶段有重叠，各项不能相加当作总耗时"
        )
        logger.info("\n".join(rows))

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
        self._require_cloud_ready()
        for kind, payload in self._ask_impl(
            video_id,
            question,
            top_k=top_k,
            max_context_tokens=max_context_tokens,
            stream=False,
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
        self._require_cloud_ready()
        yield from self._ask_impl(
            video_id,
            question,
            top_k=top_k,
            max_context_tokens=max_context_tokens,
            stream=True,
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
        if getattr(self, '_use_integrity', False) and self._cfg.integrity.enabled:
            from backend.analysis.qa import ask
            yield from ask(self, video_id, question, top_k=top_k, max_context_tokens=max_context_tokens,
                           stream=stream, session_id=session_id)
            return
        started = time.monotonic()
        result = AskResult(question=question, video_id=video_id)
        text = (question or "").strip()
        if not text:
            result.notes.append("问题为空")
            yield ("final", result)
            return
        if stream:
            yield ("status", {"phase": "accepted", "message": "已收到问题，准备检索…"})
        video = self._repo.find_video(video_id)
        if video is None:
            result.notes.append(f"视频不存在: {video_id}")
            yield ("final", result)
            return

        # ---- 多轮（C 端对话）----
        # 1) 查询改写：追问常带指代/省略（"那第二个进球呢"），坏查询直接
        #    召回坏证据 —— 改写发生在检索之前，用上一轮问答补全。
        # 2) 历史前缀：最近 N 轮作为 messages 前缀传给 LLM（证据只挂本轮）。
        from backend.pipeline.chat_sessions import get_chat_store  # noqa: PLC0415

        store = get_chat_store()
        history_key = f"{video_id}:{session_id}" if session_id else video_id
        history = store.history(history_key) if self._cfg.qa.multi_turn else []
        if not history and self._cfg.qa.multi_turn:
            try:
                for turn in self._repo.list_qa(
                    video_id, session_id=session_id, limit=200
                )[-self._cfg.qa.history_window :]:
                    history.extend(
                        [
                            {"role": "user", "content": turn["question"][:400]},
                            {"role": "assistant", "content": turn["answer"][:800]},
                        ]
                    )
            except Exception:
                logger.debug("读取会话历史失败", exc_info=True)
        retrieval_query = text
        if history and self._cfg.qa.query_rewrite:
            rewritten = self._rewrite_query(text, history)
            if rewritten:
                retrieval_query = rewritten
                result.notes.append(f"追问已改写检索: {rewritten[:60]}")

        if stream:
            yield (
                "status",
                {"phase": "retrieval", "message": "正在检索文字与画面证据…"},
            )
        retrieval_start = time.monotonic()
        items = self._retrieve(video_id, retrieval_query, top_k)
        logger.info(
            "问答检索 %.3fs video=%s", time.monotonic() - retrieval_start, video_id
        )
        result.candidates = len(items)
        if not items:
            items = self._fallback_items(video_id)
            result.notes.append("检索无命中，改用片段摘录作为证据")
            result.degraded = True
        items = self._attach_visuals(video_id, items)

        builder = self._ctx or _build_context(self._cfg)
        if max_context_tokens:
            from backend.pipeline.context_builder import (  # noqa: PLC0415
                ContextBudget,
                ContextBuilder,
            )
            from backend.prompts import get_prompt_registry  # noqa: PLC0415

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
            yield (
                "meta",
                {
                    "video_id": video_id,
                    "candidates": result.candidates,
                    "context_chars": result.context_chars,
                    "backend_hint": getattr(self._llm, "primary", "")
                    if self._llm is not None
                    else "",
                },
            )

        llm = self._llm or _build_llm(self._cfg, None)
        answer_parts: list[str] = []
        generated = False
        resp: Any = None
        stream_fn = getattr(llm, "stream", None)
        # 多轮生成：历史轮作为前缀（只保留问答文本），证据 prompt 永远是
        # 最后一条 user 消息 —— 引用编号只对应本轮检索结果，跨轮不串。
        history_window = int(self._cfg.qa.history_window)
        prefix = history[-history_window * 2 :] if history else []
        gen_messages = self._preference_messages(
            [*prefix, {"role": "user", "content": built.text}], text
        )
        first_delta = None
        if stream:
            yield (
                "status",
                {"phase": "generation", "message": "证据已就绪，等待云端首字…"},
            )
        try:
            if stream and callable(stream_fn):
                with self._inference_slot("回答问题", None, local=False):
                    for piece in stream_fn(gen_messages):
                        if first_delta is None:
                            first_delta = time.monotonic()
                            logger.info(
                                "问答首字 %.3fs video=%s",
                                first_delta - started,
                                video_id,
                            )
                        answer_parts.append(str(piece))
                        yield ("delta", str(piece))
                raw_answer = "".join(answer_parts)
                generated = bool(raw_answer.strip())
                if not generated:
                    raise RuntimeError("流式输出为空")
            else:
                with self._inference_slot("回答问题", None, local=False):
                    resp = llm.complete(gen_messages)
                raw_answer = resp.text
                generated = bool(raw_answer.strip())
                if stream and generated:
                    # 后端不支持流式（如注入的 Fake LLM / keywords 兜底）：
                    # 一次性吐全文，接口形状不变，首 token 延迟=总延迟。
                    answer_parts = [raw_answer]
                    yield ("delta", raw_answer)
        except Exception as exc:
            raise PipelineError(f"云端问答失败：{exc}") from exc
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
                from backend.pipeline.chat_sessions import Turn  # noqa: PLC0415

                store.remember(
                    history_key,
                    Turn(
                        question=text,
                        answer=result.answer,
                        rewritten=retrieval_query if retrieval_query != text else "",
                        elapsed_ms=result.elapsed_ms,
                    ),
                )
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
        if persisted:
            self._notify_memory()
        result.session_id = session_id
        logger.info("问答完成 %.3fs video=%s", (time.monotonic() - started), video_id)
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
            if self._cfg.qa.query_rewrite_mode == "context":
                # 原问题优先，附加上轮主题；无需额外云端往返，也不凭空生成实体。
                return f"{question} {last_q[:100]}"[:200]
            from backend.prompts import get_prompt_registry  # noqa: PLC0415

            llm = self._llm or _build_llm(self._cfg, None)
            prompt = get_prompt_registry().render(
                "chat_query_rewrite",
                last_question=last_q[:300],
                last_answer=last_a[:600],
                question=question[:300],
            )
            with self._inference_slot("查询改写", None, local=False):
                resp = llm.complete(
                    [{"role": "user", "content": prompt}],
                    max_tokens=120,
                    temperature=0.0,
                    timeout=self._cfg.qa.query_rewrite_timeout_s,
                )
            out = str(resp.text or "").strip().strip('"“”‘’').strip()
            # 违约防护：改写结果异常长（模型开始解释）或与原问一样时弃用
            if not out or len(out) > 200 or out == question:
                return ""
            return out[:200]
        except Exception as exc:  # noqa: BLE001 - 改写是可选优化
            logger.debug("查询改写失败，使用原问题: %s", exc)
            return ""

    def delete(self, video_id: str, *, missing_ok=False) -> dict[str, Any]:
        """两个网页删除入口共用：视频、所有会话、索引与受管缓存一起清理。"""
        from backend.storage.lifecycle import video_guard
        from backend.storage.repository import RepositoryError
        with video_guard(video_id):
            lock = _video_lock(video_id)
            if not lock.acquire(blocking=False):
                raise PipelineError('该视频正在分析，请等待当前任务完成后删除')
            try:
                self._repo.ensure_video_idle(video_id)
                if not missing_ok and self._repo.find_video(video_id) is None:
                    return {'video_id': video_id, 'db_error': '视频不存在'}
                return self._delete_linked_video(video_id, missing_ok=missing_ok)
            except RepositoryError as exc:
                raise PipelineError(str(exc)) from exc
            finally:
                lock.release()

    def _delete_linked_video(self, video_id, *, missing_ok=False):
        from backend.pipeline.deletion import attachment_paths, purge_uploads
        from backend.pipeline.chat_sessions import get_chat_store
        conversations = self._repo._query('SELECT id FROM conversations WHERE video_id=?', (video_id,))
        uploads = attachment_paths(self._repo, video_id=video_id)
        with self._bm25_lock:
            self._bm25_cache.pop(video_id, None)
        removed: dict[str, Any] = {"video_id": video_id, 'conversation_ids': [row['id'] for row in conversations]}
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
            return removed
        try:
            doomed = self._repo.delete_video(video_id, missing_ok=missing_ok)
            removed["db"] = doomed
            get_chat_store().clear(video_id)
            # 库删了文件不能留：关键帧 JPG 与音频 WAV 都在按 video_id 命名的
            # 目录里，漏删一次就是几 MB 的孤儿数据，且再也无法从库里定位。
            removed["files"] = self.purge_files(video_id, doomed, local_path=local_path)
            if getattr(self, "_use_integrity", False):
                from backend.analysis.runs import purge_versions
                removed["analysis_versions"] = purge_versions(self, video_id)
                from backend.analysis.transcripts import purge
                removed['transcript_cache'] = purge(self, video_id)
            removed['uploads'] = purge_uploads(self._repo, self._cfg, uploads)
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
            from backend.pipeline.deletion import referenced_files, file_key
            referenced = referenced_files(self._repo) if hasattr(self, '_repo') else set()
            if file_key(local_path) not in referenced:
                _unlink_owned(
                    Path(local_path),
                    [Path(self._cfg.path(tag)) for tag in ("videos", "temp")]
                    + [Path(self._cfg.path("data")) / "uploads"],
                )
            if not Path(local_path).exists():
                removed["source"] = 1
        return removed

    @contextmanager
    def _inference_slot(
        self, label: str, report: AnalysisReport | None, *, local: bool = True
    ) -> Iterator[None]:
        """重模型临界区：跨实例串行，等待期间继续轮询而不是死等。

        为什么不在构造时排队：一次分析要走 ASR → 视觉 → 向量 → LLM 四段，
        中间夹着纯 CPU 的抽帧和切片。整段串行会让第二个任务连"能先干的活"
        都干不了；分段持锁才是真正省内存又不浪费时间的做法。
        """
        if not local or not self._cfg.limits.inference_serial:
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
            if keyframes and alive >= (1.0 if getattr(self, "_analysis_run", None) is not None else 0.8):
                ready["keyframes"] = keyframes
            else:
                logger.info(
                    "视频 %s 的关键帧只剩 %d/%d 个文件，放弃恢复视觉结果",
                    video_id,
                    len(keyframes),
                    len(kf_records),
                )
                for key in ("clips", "keyframes"):
                    ready.pop(key, None)
        return ready

    def _resume_point(self, video_id: str, opts: AnalyzeOptions) -> Stage | None:
        if not opts.resume or opts.force:
            return None
        if getattr(self, '_analysis_run', None) is not None:
            progress = self._analysis_run.get('progress') or {}
            for stage in Stage:
                if stage.value == progress.get('stage') and stage is not Stage.DONE:
                    return stage
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
        self,
        report: AnalysisReport,
        stage: Stage,
        progress: Callable[[str, float, str], None] | None,
        *,
        parallel_result: bool = False,
    ) -> StageReport:
        if getattr(self, '_analysis_run', None) is not None:
            previous = (self._analysis_run.get('progress') or {}).get('stage')
            previous_order = next((item.order for item in Stage if item.value == previous), -1)
            # 重试仍先经过解析/探测；不能把已完成的昂贵阶段检查点写回parse。
            if stage.order > previous_order:
                self._analysis_run.put('progress', {'stage': stage.value})
        item = StageReport(stage=stage, started_at=time.monotonic())
        report.stages.append(item)
        logger.info(
            "%s %s video=%s",
            "阶段接收并行结果" if parallel_result else "阶段开始",
            _STAGE_LABEL[stage], report.video_id or "?",
        )
        self._emit(progress, stage, _STAGE_PROGRESS[stage], _STAGE_LABEL[stage])
        if report.video_id:
            try:
                self._repo.update_video_status(
                    report.video_id,
                    _STAGE_STATUS[stage],
                    stage=stage.value,
                    progress=_STAGE_PROGRESS[stage],
                )
            except Exception:  # noqa: BLE001
                logger.debug("更新 video 阶段失败", exc_info=True)
        return item

    def _finish(
        self,
        item: StageReport,
        *,
        message: str = "",
        degraded: bool = False,
        elapsed_ms: int | None = None,
    ) -> None:
        item.message = message
        item.degraded = degraded
        # 显式传了就用传的；没传就按 _begin 记的起点现算。以前这里恒等于 0，
        # 于是所有阶段耗时都显示 0ms，"1h 视频 ≤30min" 的性能指标根本无从核对。
        if elapsed_ms is not None:
            item.elapsed_ms = elapsed_ms
        elif item.started_at:
            item.elapsed_ms = max(0, int((time.monotonic() - item.started_at) * 1000))
        logger.info(
            "%s %s %.3fs %s",
            "阶段完成" if item.ok else "阶段失败",
            _STAGE_LABEL[item.stage],
            item.elapsed_ms / 1000,
            message,
        )

    def _end(
        self,
        report: AnalysisReport,
        stage: Stage,
        progress: Callable[[str, float, str], None] | None,
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

    def status(self, video_id: str) -> dict[str, Any]:
        video = self._repo.find_video(video_id)
        if video is None:
            return {"video_id": video_id, "exists": False}
        return {"exists": True, **video.to_dict()}

    def close(self) -> None:
        for component in (
            self._memory,
            self._vectors,
            self._embedder,
            self._asr,
            self._vl,
            self._llm,
        ):
            close = getattr(component, "close", None) or getattr(
                component, "unload", None
            )
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
            ("asr", self._asr),
            ("vl", self._vl),
            ("embed", self._embedder),
            ("vectors", self._vectors),
            ("llm", self._llm),
        ):
            parts.append(f"{label}={'on' if comp is not None else 'off'}")
        return f"Orchestrator({', '.join(parts)})"

    @classmethod
    def from_config(cls, config=None, *, repository=None, on_progress=None):
        """严格装配并验证必需模型，失败阻止启动。"""
        from backend.pipeline.factory import build_orchestrator

        return build_orchestrator(
            config,
            repository=repository,
            on_progress=on_progress,
            orchestrator_class=cls,
        )


__all__ = [
    "Orchestrator",
    "AnalysisReport",
    "AnalyzeOptions",
    "AskResult",
    "Citation",
    "PipelineError",
    "Stage",
    "StageReport",
    "compute_fingerprint",
    "extract_citations",
]
