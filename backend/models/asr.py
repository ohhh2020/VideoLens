"""faster-whisper-small 严格转写与字幕优先路径。

仅使用配置的设备和算力。模型缺失、加载或推理失败直接报错；
VAD 判定无人声时可返回空转写，但不会用空内容掩盖模型错误。
批量模式保留词级时间戳，时间戳统一为毫秒。
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

from backend.media.errors import MediaError
from backend.models._runtime import gpu_slot
from backend.media.subtitles import SubtitleCue, load_subtitles

logger = logging.getLogger("zhiying.models.asr")

__all__ = [
    "AsrConfig",
    "AsrEngine",
    "AsrError",
    "Transcript",
    "TranscriptSegment",
    "Word",
]

# whisper 在静音段常见的"幻觉"输出，中文与英文各一组
_HALLUCINATION_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^请不吝点赞[，,、\s]*订阅[，,、\s]*转发[，,、\s]*打赏.*$",
        r"^订阅[，,、\s]*?点赞[，,、\s]*?转发.*$",
        r"^(?:字幕|字幕由).{0,12}(?:提供|制作|生成).*$",
        r"^(?:thanks?|thank you)(?: for watching| so much for watching)?[.!\s]*$",
        r"^subtitles? (?:by|provided by).*$",
        r"^amara\.org.*$",
        r"^\.+$",
        r"^[\s\-—–_·。，、,.!?！？]*$",
    )
)


class AsrError(MediaError):
    """转写过程中的可识别错误（音轨缺失、模型加载失败、依赖缺失）。"""


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class AsrConfig:
    """ASR 行为配置。所有字段可由 ``config.asr`` + ``config.media`` 注入。"""

    model_name: str = "faster-whisper-small"
    device: str | None = None
    compute_type: str | None = None
    beam_size: int = 5
    language: str | None = None
    vad_filter: bool = True
    vad_min_silence_ms: int = 500
    subtitle_priority: tuple[str, ...] = ("srt", "vtt", "ass", "ssa")
    fallback_to_subtitle: bool = True
    min_language_probability: float = 0.5
    word_timestamps: bool = True
    cpu_threads: int = 4
    num_workers: int = 1
    batch_size: int = 4
    condition_on_previous_text: bool = False
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    initial_prompt: str = "以下是普通话视频的转写内容，请使用简体中文并添加标点。"
    max_repeat_ratio: float = 0.6
    """兼容旧配置字段；仅删除时间重叠且逐字相同的重复结果，不再按相似度丢弃。"""

    def validate(self) -> None:
        if self.device not in {None, "auto", "cuda", "cpu", "mps"}:
            raise ValueError(f"asr.device 非法: {self.device}")
        if self.beam_size < 1:
            raise ValueError(f"beam_size 应 >= 1: {self.beam_size}")
        if not 0.0 <= self.min_language_probability <= 1.0:
            raise ValueError("min_language_probability 应在 0..1")
        if self.cpu_threads < 1:
            raise ValueError("cpu_threads 应 >= 1")
        if self.batch_size < 1 or self.num_workers < 1:
            raise ValueError("batch_size 和 num_workers 应 >= 1")
        if not 0.0 <= self.max_repeat_ratio <= 1.0:
            raise ValueError("max_repeat_ratio 应在 0..1")

    @classmethod
    def from_config(cls, config: Any) -> AsrConfig:
        """从 :class:`backend.config.AppConfig` 构造。"""
        asr = getattr(config, "asr", None)
        models = getattr(config, "models", None)
        kwargs: dict[str, Any] = {}
        for name in (
            "model",
            "compute_type",
            "beam_size",
            "language",
            "vad_filter",
            "vad_min_silence_ms",
            "subtitle_priority",
            "fallback_to_subtitle",
            "min_language_probability",
            "cpu_threads",
            "num_workers",
            "batch_size",
        ):
            value = getattr(asr, name, None)
            if value is not None:
                kwargs["model_name" if name == "model" else name] = value
        device = getattr(asr, "device", None)
        if device is None:
            device = getattr(models, "device", None)
        if device:
            kwargs["device"] = device
        if isinstance(kwargs.get("subtitle_priority"), list):
            kwargs["subtitle_priority"] = tuple(kwargs["subtitle_priority"])
        return cls(**kwargs)


# --------------------------------------------------------------------------- #
# 输出结构
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Word:
    """词级时间戳（用于切片边界微调与逐字定位）。"""

    word: str
    start_ms: int
    end_ms: int
    probability: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "probability": round(self.probability, 4),
        }


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """一条转写片段（毫秒时间戳，单调不倒退）。"""

    index: int
    start_ms: int
    end_ms: int
    text: str
    words: tuple[Word, ...] = ()
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
        }
        if self.words:
            payload["words"] = [w.to_dict() for w in self.words]
        if self.avg_logprob is not None:
            payload["avg_logprob"] = round(self.avg_logprob, 4)
        if self.no_speech_prob is not None:
            payload["no_speech_prob"] = round(self.no_speech_prob, 4)
        return payload


@dataclass(frozen=True, slots=True)
class Transcript:
    """一次转写的完整结果，是 ``chunker`` 的直接输入。"""

    video_id: str
    segments: tuple[TranscriptSegment, ...] = ()
    language: str = ""
    language_probability: float = 0.0
    duration_ms: int = 0
    source: str = "whisper"
    """``whisper`` / ``subtitle`` / ``empty``"""
    model_name: str = ""
    device: str = ""
    compute_type: str = ""
    elapsed_ms: int = 0
    subtitle_path: Path | None = None
    dropped_segments: int = 0
    degraded: bool = False
    warnings: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def text(self) -> str:
        return " ".join(seg.text for seg in self.segments if seg.text).strip()

    @property
    def char_count(self) -> int:
        return sum(len(seg.text) for seg in self.segments)

    @property
    def is_empty(self) -> bool:
        return not self.segments

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "language": self.language,
            "language_probability": round(self.language_probability, 4),
            "duration_ms": self.duration_ms,
            "source": self.source,
            "model_name": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "elapsed_ms": self.elapsed_ms,
            "subtitle_path": str(self.subtitle_path) if self.subtitle_path else None,
            "dropped_segments": self.dropped_segments,
            "degraded": self.degraded,
            "warnings": list(self.warnings),
            "segments": [seg.to_dict() for seg in self.segments],
        }


# --------------------------------------------------------------------------- #
# 文本后处理
# --------------------------------------------------------------------------- #
def _ensure_cuda_dlls() -> None:
    """挂载 CUDA 运行库目录（项目 runtime/ 与 pip 用户级安装均覆盖）。"""
    from backend.models._runtime import ensure_cuda_dlls  # noqa: PLC0415

    ensure_cuda_dlls()


def _looks_like_hallucination(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return any(p.match(stripped) for p in _HALLUCINATION_PATTERNS)


def _clean_text(text: str) -> str:
    cleaned = re.sub(r"[ \t]{2,}", " ", text.replace("\n", " ")).strip()
    return cleaned


@lru_cache(maxsize=1)
def _cuda_usable() -> bool:
    """CUDA 是否**真的**可用（不是"有没有设备"）。

    判据以 **CTranslate2 优先**：跑转写的就是它，它数得到的卡就是真能用的卡。
    曾经反过来"torch 优先"，结果 torch 是 CPU 构建（+cpu 轮子）时，CTranslate2
    明明报 1 张卡也被判成不可用，白白全程 CPU —— 这正是"有 GPU 却全在
    CPU 上跑"的一个来源。torch 只作为没有 CT2 时的旁证。

    若 CUDA 运行库损坏（缺 cublas 等），首个 float16 档会在 load() 里抛错并
    明确报错并保留用户选择的设备，不会自动切换CPU。
    """
    from backend.models._runtime import ensure_cuda_dlls, ensure_torch_path  # noqa: PLC0415

    ensure_torch_path()
    ensure_cuda_dlls()
    try:
        import ctranslate2  # type: ignore[import-not-found]

        if ctranslate2.get_cuda_device_count() > 0:
            return True
    except Exception:  # noqa: BLE001 - 没装 CT2 不该发生（asr 依赖它），兜住
        pass
    try:
        import torch  # type: ignore[import-not-found]

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# 引擎
# --------------------------------------------------------------------------- #
class AsrEngine:
    """faster-whisper 封装。

    >>> engine = AsrEngine(AsrConfig(model_name="faster-whisper-small"))
    >>> tr = engine.transcribe("data/audio/v123.wav", video_id="v123")
    >>> tr.segments[0].start_ms
    0
    """

    def __init__(
        self, config: AsrConfig | None = None, *, model_manager: Any = None
    ) -> None:
        self.config = config or AsrConfig()
        self.config.validate()
        self._model_manager = model_manager
        self._transcribe_lock = threading.Lock()
        self._model: Any = None
        self._loaded_key: tuple[str, str, str] | None = None
        self._learned_batch = 4
        self._batch_peak = 0
        self.last_batch_size = 1
    # ---- 模型加载 --------------------------------------------------------- #
    def _manager(self) -> Any:
        if self._model_manager is None:
            from backend.models.manager import get_model_manager

            self._model_manager = get_model_manager()
        return self._model_manager

    def _model_path(self) -> Path:
        return Path(self._manager().ensure_model(self.config.model_name))

    def _device_plan(self):
        device = self.config.device
        if device == "auto":
            device = "cuda" if _cuda_usable() else "cpu"
        return [
            (
                device,
                self.config.compute_type or ("float16" if device == "cuda" else "int8"),
            )
        ]

    def load(self, *, force_reload: bool = False) -> Any:
        """加载（或复用）WhisperModel；指定设备失败立即报错，禁止降级。

        :raises AsrError: 指定的设备或精度不可用
        """
        try:
            import faster_whisper  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise AsrError(
                "未安装 faster-whisper，请执行 `python -m pip install -r requirements.txt`"
            ) from exc

        _ensure_cuda_dlls()
        model_path = self._model_path()
        plan = self._device_plan()
        key = (str(model_path), plan[0][0], plan[0][1])
        if self._model is not None and not force_reload and self._loaded_key == key:
            return self._model

        errors: list[str] = []
        for device, compute in plan:
            try:
                logger.info(
                    "加载 Whisper 模型 device=%s compute_type=%s path=%s",
                    device,
                    compute,
                    model_path.name,
                )
                model = faster_whisper.WhisperModel(
                    str(model_path),
                    device=device,
                    compute_type=compute,
                    cpu_threads=self.config.cpu_threads,
                    num_workers=self.config.num_workers,
                    local_files_only=True,
                )
            except Exception as exc:  # noqa: BLE001
                message = f"{device}/{compute}: {type(exc).__name__}: {exc}"
                errors.append(message)
                logger.error("Whisper 指定设备加载失败，禁止降级 —— %s", message[:240])
                continue
            self._model = model
            self._loaded_key = (str(model_path), device, compute)
            logger.info(
                "Whisper 就绪 device=%s compute_type=%s",
                device,
                compute,
            )
            return model

        raise AsrError(
            "Whisper 指定设备加载失败（禁止降级）:\n" + "\n".join(errors)
        )

    def unload(self) -> None:
        """释放模型引用（配合 ``ModelManager`` 的显存管理）。"""
        self._model = None
        self._loaded_key = None
        logger.debug("Whisper 模型引用已释放")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def active_device(self) -> tuple[str, str]:
        if self._loaded_key is None:
            return ("", "")
        return (self._loaded_key[1], self._loaded_key[2])

    # ---- 转写 ------------------------------------------------------------- #
    def transcribe(
        self,
        audio_path: str | Path,
        *,
        video_id: str,
        subtitles: Iterable[str | Path] = (),
        duration_ms: int | None = None,
        allow_subtitle: bool | None = None,
        max_batch: int | None = None,
    ) -> Transcript:
        """转写入口：字幕优先，失败退回 Whisper，再失败返回空结果。

        :param subtitles: 候选字幕文件（yt-dlp 下载的 ``.vtt`` 等）
        :param duration_ms: 已知时长，用于填充 ``Transcript.duration_ms``
        """
        started = time.monotonic()
        use_subtitle = (
            self.config.fallback_to_subtitle
            if allow_subtitle is None
            else allow_subtitle
        )
        warnings: list[str] = []

        if use_subtitle:
            transcript = self._from_subtitles(
                video_id, subtitles, duration_ms, warnings
            )
            if transcript is not None:
                return replace(
                    transcript, elapsed_ms=int((time.monotonic() - started) * 1000)
                )

        with self._transcribe_lock, gpu_slot(self._device_plan()[0][0]):
            transcript = self._from_whisper(
                audio_path,
                video_id=video_id,
                duration_ms=duration_ms,
                warnings=warnings,
                max_batch=max_batch,
            )
        return replace(transcript, elapsed_ms=int((time.monotonic() - started) * 1000))

    def transcribe_subtitles_only(
        self,
        video_id: str,
        subtitles: Iterable[str | Path],
        *,
        duration_ms: int | None = None,
    ) -> Transcript:
        """只走字幕路径（用于明确知道不该跑推理的场景，如批量导入）。"""
        warnings: list[str] = []
        transcript = self._from_subtitles(video_id, subtitles, duration_ms, warnings)
        if transcript is not None:
            return transcript
        return Transcript(
            video_id=video_id, source="empty", degraded=True, warnings=tuple(warnings)
        )

    # ---- 内部：字幕路径 --------------------------------------------------- #
    def _from_subtitles(
        self,
        video_id: str,
        subtitles: Iterable[str | Path],
        duration_ms: int | None,
        warnings: list[str],
    ) -> Transcript | None:
        paths = [Path(p) for p in subtitles if str(p)]
        if not paths:
            return None
        cues, hit, sub_warnings = load_subtitles(
            paths, preferred_order=self.config.subtitle_priority
        )
        warnings.extend(sub_warnings)
        if not cues or hit is None:
            return None
        segments = self._segments_from_cues(cues)
        if not segments:
            warnings.append("字幕内容全是空白或幻觉文本，已忽略")
            return None
        language = self._guess_language(segments)
        return Transcript(
            video_id=video_id,
            segments=segments,
            language=language,
            language_probability=1.0,
            duration_ms=duration_ms or segments[-1].end_ms,
            source="subtitle",
            subtitle_path=hit,
            model_name="subtitle",
            device="none",
            compute_type="none",
            warnings=tuple(warnings),
        )

    def _segments_from_cues(
        self, cues: Sequence[SubtitleCue]
    ) -> tuple[TranscriptSegment, ...]:
        segments: list[TranscriptSegment] = []
        previous = ""
        dropped = 0
        for cue in cues:
            text = _clean_text(cue.text)
            if _looks_like_hallucination(text):
                dropped += 1
                continue
            if segments and previous == text and cue.start_ms < segments[-1].end_ms:
                dropped += 1
                continue
            start = max(0, cue.start_ms)
            end = max(start + 1, cue.end_ms)
            segments.append(
                TranscriptSegment(
                    index=len(segments), start_ms=start, end_ms=end, text=text
                )
            )
            previous = text
        if dropped:
            logger.debug("字幕清洗丢弃 %d 条（幻觉/复读）", dropped)
        return tuple(segments)

    # ---- 内部：Whisper 路径 ----------------------------------------------- #
    def _from_whisper(
        self,
        audio_path: str | Path,
        *,
        video_id: str,
        duration_ms: int | None,
        warnings: list[str],
        max_batch: int | None = None,
    ) -> Transcript:
        path = Path(audio_path)
        if not path.is_file():
            raise AsrError(f"音频文件不存在: {path}")
        model = self.load()

        device, compute = self.active_device
        logger.info(
            "Whisper 整段音频转写开始 video=%s device=%s compute_type=%s batch_size=%d",
            video_id, device, compute, self.config.batch_size,
        )
        options: dict[str, Any] = {
            "beam_size": self.config.beam_size,
            "language": self.config.language,
            "word_timestamps": self.config.word_timestamps,
            "condition_on_previous_text": self.config.condition_on_previous_text,
            "temperature": list(self.config.temperature),
            "no_speech_threshold": self.config.no_speech_threshold,
            "log_prob_threshold": self.config.log_prob_threshold,
            "compression_ratio_threshold": self.config.compression_ratio_threshold,
            "vad_filter": self.config.vad_filter,
            "vad_parameters": {
                "min_silence_duration_ms": self.config.vad_min_silence_ms
            },
        }
        if self.config.language is None or self.config.language.startswith("zh"):
            options["initial_prompt"] = self.config.initial_prompt

        from backend.models.batching import is_oom, release_oom
        from backend.models.batching import GpuMemoryProbe
        maximum = max(1, min(8, max_batch or self.config.batch_size))
        batch = min(maximum, self._learned_batch, self.config.batch_size if not device.startswith('cuda') else maximum)
        while True:
            try:
                if batch > 1:
                    from faster_whisper import BatchedInferencePipeline
                    runner = BatchedInferencePipeline(model=model)
                    raw_segments, info = runner.transcribe(str(path), batch_size=batch, **options)
                else:
                    raw_segments, info = model.transcribe(str(path), **options)
                with GpuMemoryProbe(device) as probe:
                    segments_list = list(raw_segments)
                self.last_batch_size = batch
                self._batch_peak = max(self._batch_peak, probe.used)
                self._learned_batch = batch
                if device.startswith('cuda') and probe.minimum > max(256*1024**2, self._batch_peak) * 3:
                    self._learned_batch = min(maximum, batch * 2)
                logger.info('Whisper批次完成 actual_batch=%d peak_device_delta=%.1fMiB next_batch=%d', batch, probe.used/1024**2, self._learned_batch)
                break
            except Exception as exc:
                if not device.startswith('cuda') or not is_oom(exc) or batch == 1:
                    raise AsrError(f"Whisper 转写失败：{type(exc).__name__}: {exc}") from exc
                release_oom(exc)
                batch = max(1, batch // 2)
                logger.warning('Whisper显存不足，缩小批次至%d，保持CUDA设备', batch)
        language = str(getattr(info, "language", "") or "")
        probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        if probability and probability < self.config.min_language_probability:
            warnings.append(
                f"语言识别置信度偏低（{language or '未知'} {probability:.2f}），结果可能不准"
            )

        segments: list[TranscriptSegment] = []
        previous = ""
        dropped = 0
        last_end = 0
        for raw in segments_list:
            text = _clean_text(str(getattr(raw, "text", "") or ""))
            if _looks_like_hallucination(text):
                dropped += 1
                continue
            if previous and previous == text and int(round(float(getattr(raw, "start", 0.0)) * 1000)) < last_end:
                dropped += 1
                continue
            start = int(round(float(getattr(raw, "start", 0.0)) * 1000))
            end = int(round(float(getattr(raw, "end", 0.0)) * 1000))
            start = max(start, last_end)
            end = max(end, start + 1)
            words = self._words_of(raw, segment_end=end)
            if words:
                start = min(start, words[0].start_ms)
                end = max(end, words[-1].end_ms)
            segments.append(
                TranscriptSegment(
                    index=len(segments),
                    start_ms=start,
                    end_ms=end,
                    text=text,
                    words=words,
                    avg_logprob=self._float_or_none(getattr(raw, "avg_logprob", None)),
                    no_speech_prob=self._float_or_none(
                        getattr(raw, "no_speech_prob", None)
                    ),
                    compression_ratio=self._float_or_none(
                        getattr(raw, "compression_ratio", None)
                    ),
                )
            )
            previous = text
            last_end = end

        if dropped:
            logger.info("Whisper 清洗丢弃 %d 段（幻觉/复读）", dropped)
        if not segments:
            warnings.append("转写结果为空（可能整段无人声）")

        total = duration_ms or int(
            round(float(getattr(info, "duration", 0.0) or 0.0) * 1000)
        )
        return Transcript(
            video_id=video_id,
            segments=tuple(segments),
            language=language,
            language_probability=probability,
            duration_ms=total or (segments[-1].end_ms if segments else 0),
            source="whisper",
            model_name=self.config.model_name,
            device=device,
            compute_type=compute,
            dropped_segments=dropped,
            degraded=not segments,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _words_of(raw: Any, *, segment_end: int) -> tuple[Word, ...]:
        raw_words = getattr(raw, "words", None) or ()
        result: list[Word] = []
        for item in raw_words:
            token = str(getattr(item, "word", "") or "").strip()
            if not token:
                continue
            start = int(round(float(getattr(item, "start", 0.0) or 0.0) * 1000))
            end = int(round(float(getattr(item, "end", 0.0) or 0.0) * 1000))
            start = max(0, start)
            end = max(end, start + 1)
            if end > segment_end + 2_000:  # whisper 偶发的越界时间戳
                end = min(end, segment_end)
            result.append(
                Word(
                    word=token,
                    start_ms=start,
                    end_ms=end,
                    probability=float(getattr(item, "probability", 1.0) or 1.0),
                )
            )
        return tuple(result)

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _guess_language(segments: Sequence[TranscriptSegment]) -> str:
        sample = " ".join(seg.text for seg in segments[:20])
        if not sample:
            return ""
        chinese = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
        return "zh" if chinese / max(1, len(sample)) > 0.2 else "en"
