"""media 阶段实现，仅通过编排器状态与其他阶段协作。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable


from backend.pipeline.domain import AnalysisReport, AnalyzeOptions, PipelineError, Stage
from backend.pipeline.helpers import _download_progress

logger = logging.getLogger("zhiying.pipeline")


class MediaStages:
    def _parse_source(self, source: str, opts: AnalyzeOptions) -> Any:
        from backend.media.parser import ParseOptions, parse_source  # noqa: PLC0415

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
                    media_source.video_id,
                    "downloading",
                    stage=Stage.PARSE.value,
                    progress=0.0,
                    error=None,
                    finished=False,
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
            if self._vectors is not None and not getattr(self, "_isolated_run", False):
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
        from backend.media.ffmpeg_utils import probe_media  # noqa: PLC0415

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

    def _extract_audio(
        self, local_path: Path, video_id: str, probe_info: Any
    ) -> tuple[Path | None, bool]:
        from backend.media.ffmpeg_utils import extract_audio  # noqa: PLC0415

        if not getattr(probe_info, "has_audio", False):
            return None, False
        audio_dir = Path(self._cfg.path("audio")) / video_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        dest = audio_dir / "audio.wav"
        if dest.exists() and dest.stat().st_size > 1024:
            return dest, True
        try:
            extract_audio(
                local_path, dest, sample_rate=self._cfg.media.audio_sample_rate
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("音频抽取失败: %s", exc)
            return None, False
        return dest, True

    def _transcribe(self, video_id, audio_path, local_path, probe_info, opts):
        from backend.models.asr import Transcript

        if audio_path is None and not getattr(probe_info, "has_audio", True):
            return Transcript(video_id=video_id, source="no_audio")
        if self._asr is None:
            raise PipelineError("缺少必需 Whisper 组件")
        with self._inference_slot("语音转写", None):
            result = self._asr.transcribe(
                audio_path or local_path,
                video_id=video_id,
                subtitles=self._find_subtitles(local_path),
                duration_ms=int(getattr(probe_info, "duration_ms", 0) or 0),
                allow_subtitle=self._cfg.asr.fallback_to_subtitle,
                **({"max_batch": self._cfg.integrity.asr_batch_max} if getattr(self, "_analysis_run", None) is not None else {}),
            )
        if result.degraded and result.source == "empty":
            raise PipelineError("Whisper 失败：" + "；".join(result.warnings))
        return result

    def _find_subtitles(self, local_path: Path) -> list[Path]:
        """在同目录下找字幕文件，按优先级排序（外部字幕优先于内嵌字幕）。

        yt-dlp 会把字幕下到 ``<video_id>/`` 里（``source.zh-Hans.vtt``）；
        用户手工放同名 ``.srt`` 也在这里被捡到。
        """
        from backend.media.subtitles import SUPPORTED_SUBTITLE_SUFFIXES  # noqa: PLC0415

        order = {suffix: i for i, suffix in enumerate(self._cfg.asr.subtitle_priority)}
        try:
            original = local_path.resolve()
            candidates = [p for base in {local_path, original} for p in base.parent.iterdir()
                if p.is_file() and p.suffix.lstrip('.').lower() in SUPPORTED_SUBTITLE_SUFFIXES
                and (p.stem == base.stem or p.stem.startswith(base.stem + '.'))]
            candidates = list(dict.fromkeys(candidates))
        except OSError:
            return []
        candidates.sort(
            key=lambda p: (order.get(p.suffix.lstrip(".").lower(), 99), p.name)
        )
        return candidates
