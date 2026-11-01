"""fast-v4：无镜头检测，音频与流式抽帧/视觉并行，报告与索引并行。"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import logging
import threading
import time

from backend.pipeline.domain import AnalysisReport, Stage, StageReport, _STAGE_LABEL
from backend.pipeline.helpers import _clip_payload, _keyframe_payload, _segment_payload

logger = logging.getLogger('zhiying.analysis')
VERSION = 'fast-v4'


def enabled(orch):
    ledger = getattr(orch, '_analysis_run', None)
    return bool(ledger and (ledger.get('policy') or {}).get('_analysis_strategy') == VERSION)


def analyze(orch, source, opts):
    from backend.analysis.decoding import stream_frames
    from backend.analysis.fast import describe_stream
    from backend.analysis import transcripts
    from backend.models.asr import Transcript
    from backend.models.http_transport import enable_pool

    started = time.monotonic()
    report = AnalysisReport()
    ledger, cfg, repo = orch._analysis_run, orch._cfg, orch._repo
    task_id, video_id = '', ''
    progress_lock, ratio = threading.Lock(), [0.0]
    def progress(phase, value, message):
        with progress_lock:
            ratio[0] = max(ratio[0], value)
            if opts.progress:
                opts.progress(phase, ratio[0], message)
    def measured(stage, fn):
        item = StageReport(stage=stage, started_at=time.monotonic())
        report.stages.append(item)
        logger.info('阶段开始 %s（并行） video=%s', _STAGE_LABEL[stage], video_id)
        try:
            result = fn()
            orch._finish(item)
            return result
        except Exception as exc:
            item.ok, item.error = False, str(exc)
            orch._finish(item, message=str(exc))
            raise
    try:
        orch._require_cloud_ready()
        stage = orch._begin(report, Stage.PARSE, progress)
        media = orch._parse_source(source, opts)
        video_id, report.video_id = media.video_id, media.video_id
        report.title = opts.title or media.display_name
        orch._register_video(media, opts)
        task_id = orch._start_task(video_id, opts)
        orch._finish(stage, message=media.kind.value)
        stage = orch._begin(report, Stage.PROBE, progress)
        path, info = orch._resolve_local(media, progress)
        report.duration_ms = info.duration_ms
        repo.update_video_meta(video_id, duration_ms=info.duration_ms, width=info.display_width,
            height=info.display_height, fps=info.fps, has_audio=info.has_audio, local_path=str(path))
        orch._finish(stage, message=f'时长{info.duration_ms}ms')
        report.profile.update(scene_detection='removed', source_width=info.display_width,
            source_height=info.display_height, pipeline='audio||decode->vision;summary||index')
        for backend in getattr(orch._llm, '_backends', []):
            enable_pool(backend)
        if getattr(orch._vl, '_runner', None):
            enable_pool(orch._vl._runner)

        def audio():
            progress('transcribe', .12, '语音与固定抽帧正在并行处理…')
            existing = ledger.get('transcript_raw')
            if existing and existing.get('segments'):
                report.profile['transcript_reused'] = 'run-checkpoint'
                return transcripts.restore(existing, video_id)
            subtitle = None
            if cfg.asr.fallback_to_subtitle:
                paths = orch._find_subtitles(path)
                if paths:
                    subtitle = orch._asr.transcribe_subtitles_only(video_id=video_id,
                        subtitles=paths, duration_ms=info.duration_ms)
                    if subtitle.is_empty or (subtitle.segments and subtitle.segments[-1].end_ms > info.duration_ms + 2000):
                        subtitle = None
            if subtitle is not None:
                report.profile['transcript_reused'] = 'matched-subtitle'
                result = measured(Stage.TRANSCRIBE, lambda: subtitle)
            elif not info.has_audio:
                result = measured(Stage.TRANSCRIBE, lambda: Transcript(video_id=video_id, source='no_audio'))
            else:
                audio_path, _ = measured(Stage.AUDIO, lambda: orch._extract_audio(path, video_id, info))
                if audio_path is None:
                    raise ValueError('音轨提取失败，未将有音轨文件当作无声视频')
                key, identity = transcripts.identity(orch, audio_path)
                cached = transcripts.read(orch, key, video_id) if opts.reuse_transcript else None
                report.profile['transcript_reused'] = 'content-cache' if cached else ''
                result = measured(Stage.TRANSCRIBE, lambda: cached or orch._transcribe(video_id, audio_path, path, info, opts))
                if cached is None:
                    transcripts.write(orch, key, identity, result)
            ledger.put('transcript_raw', result.to_dict())
            return result

        def vision():
            if opts.skip_vision:
                return []
            frames = []
            def produced():
                for frame in stream_frames(orch, path, video_id, info, report, opts):
                    frames.append(frame)
                    yield frame
            def recognize():
                return describe_stream(orch, produced(), report, progress=progress)
            measured(Stage.VISION, recognize)
            return sorted(frames, key=lambda frame: frame.timestamp_ms)

        # 任何一支失败都先等待另一支正常退出，才允许版本失败/清理，避免后台写半成品。
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='analysis-input') as pool:
            audio_future, visual_future = pool.submit(audio), pool.submit(vision)
            transcript = audio_future.result()
            report.transcript_source, report.language = transcript.source, transcript.language
            segments = list(transcript.segments)
            report.segment_count = len(segments)
            report.warnings.extend(transcript.warnings)
            if segments:
                repo.add_segments(video_id, [_segment_payload(segment) for segment in segments])
            else:
                report.warnings.append('没有语音文字，本次只依据画面与OCR')
            def chunk():
                result = orch._chunk(video_id, segments, info.duration_ms)
                if not result:
                    raise ValueError('文字切片为空')
                repo.add_clips(video_id, [_clip_payload(item) for item in result])
                return result
            clips = measured(Stage.CHUNK, chunk)
            frames = visual_future.result()
        report.clip_count, report.keyframe_count = len(clips), len(frames)
        rebound = []
        for frame in frames:
            clip = next((c for c in clips if c.core_start_ms <= frame.timestamp_ms < c.core_end_ms), None)
            rebound.append(replace(frame, clip_id=clip.clip_id if clip else None))
        visual = ledger.get('visual_results') or []
        by_frame = {}
        for row in visual:
            for key in row.get('source_ids', []):
                by_frame.setdefault(key, {})[row['kind']] = row['text']
        payloads = [_keyframe_payload(frame, by_frame.get(frame.keyframe_id, {}).get('visual', ''),
                    by_frame.get(frame.keyframe_id, {}).get('ocr', '')) for frame in rebound]
        if payloads:
            repo.add_keyframes(video_id, payloads)
            repo.set_keyframe_binding(video_id)
        progress('summary', .84, '完整事实选取、总结与向量索引正在并行处理…')
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='report-index') as pool:
            def index():
                return orch._index(video_id, clips, rebound, report, opts)
            indexing = pool.submit(measured, Stage.INDEX, index)
            def summary():
                return orch._summarize(video_id, report.title, info.duration_ms, opts)
            summarizing = pool.submit(measured, Stage.SUMMARY, summary)
            report.summary, report.summary_backend = summarizing.result()
            report.vector_count = indexing.result()
        repo.update_video_status(video_id, 'ready', stage='ready', progress=1.0, finished=True)
        if task_id:
            repo.update_task(task_id, status='done', stage='done', progress=1.0, message='分析完成')
        report.status = 'ready'
        progress('done', 1.0, '完成')
    except Exception as exc:
        logger.exception('并行视频分析失败 video=%s', video_id)
        report.status = 'failed'
        report.errors.append(f'{type(exc).__name__}: {exc}')
        if task_id:
            repo.update_task(task_id, status='failed', message=str(exc), error=str(exc))
    finally:
        report.stages.sort(key=lambda item: item.stage.order)
        report.elapsed_ms = int((time.monotonic() - started) * 1000)
        orch._log_timings(report)
    return report
