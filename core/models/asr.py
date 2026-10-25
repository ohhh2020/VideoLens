"""语音转写（faster-whisper），带"字幕优先 + 逐级降级"策略。

降级链（对应需求表4 故障处理）
------------------------------
1. **字幕优先**：若视频自带 SRT/VTT/ASS 字幕且 ``fallback_to_subtitle`` 开启，
   直接解析字幕 —— 零推理成本、时间轴比 ASR 更准。
2. **GPU 转写**：``device=cuda`` + ``compute_type=float16``。
3. **GPU 降级算力**：``float16`` 失败退 ``int8_float16`` → ``int8``（显存不足常见解法）。
4. **CPU 转写**：``device=cpu`` + ``compute_type=int8``（``float32`` 兜底）。
5. **空结果**：全部失败返回 ``Transcript(segments=[], degraded=True)``，由上层
   决定是"仅视觉理解"还是"报错"。**不抛给调用方**，因为一个视频转写失败不该
   让整条流水线炸掉。

时间戳
------
输出统一毫秒整数，且强制 ``end_ms > start_ms``、段间单调不倒退。
faster-whisper 的 ``segment.start`` 是秒（float），这里 ``round`` 到毫秒，
并在段首尾做一次单调性修复（超过 1ms 的倒退会被拉平）。
"""

from __future__ import annotations

import logging
import os
import sys
from difflib import SequenceMatcher
from functools import lru_cache
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

from core.media.errors import MediaError
from core.media.subtitles import SubtitleCue, load_subtitles

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

    model_name: str = "faster-whisper-large-v3"
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
    condition_on_previous_text: bool = False
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    initial_prompt: str = "以下是普通话视频的转写内容，请使用简体中文并添加标点。"
    max_repeat_ratio: float = 0.6
    """单段文本与上一段相似度超过该比例则判为复读幻觉（0.6 = 60%）。"""

    def validate(self) -> None:
        if self.device not in {None, "auto", "cuda", "cpu", "mps"}:
            raise ValueError(f"asr.device 非法: {self.device}")
        if self.beam_size < 1:
            raise ValueError(f"beam_size 应 >= 1: {self.beam_size}")
        if not 0.0 <= self.min_language_probability <= 1.0:
            raise ValueError("min_language_probability 应在 0..1")
        if self.cpu_threads < 1:
            raise ValueError("cpu_threads 应 >= 1")
        if not 0.0 <= self.max_repeat_ratio <= 1.0:
            raise ValueError("max_repeat_ratio 应在 0..1")

    @classmethod
    def from_config(cls, config: Any) -> AsrConfig:
        """从 :class:`core.config.AppConfig` 构造。"""
        asr = getattr(config, "asr", None)
        models = getattr(config, "models", None)
        kwargs: dict[str, Any] = {}
        for name in (
            "model", "compute_type", "beam_size", "language", "vad_filter",
            "vad_min_silence_ms", "subtitle_priority", "fallback_to_subtitle",
            "min_language_probability",
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
    from core.models._runtime import ensure_cuda_dlls  # noqa: PLC0415

    ensure_cuda_dlls()


def _looks_like_hallucination(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return any(p.match(stripped) for p in _HALLUCINATION_PATTERNS)


def _similarity(a: str, b: str) -> float:
    """复读识别：看"序列是否也一致"，不看"用了哪些字"。

    早期版本用字符集合 Jaccard，在中文上勉强可用，在英文上必失效 ——
    两段毫不相干的英文都由同样 26 个字母组成，集合重叠 0.85 以上，于是
    正常的第二句被当成复读丢弃（实测 11 秒英文音频只留下第一句，
    WER 从 4% 虚高到 55%）。``SequenceMatcher`` 比值要求顺序也吻合，
    真复读才会接近 1.0，正常相邻句通常在 0.3 以下。
    """
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


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
    沿设备计划快速降级到 CPU（秒级代价，不是分钟级），因此这里宁可乐观。
    """
    from core.models._runtime import ensure_cuda_dlls, ensure_torch_path  # noqa: PLC0415

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

    >>> engine = AsrEngine(AsrConfig(model_name="faster-whisper-large-v3"))
    >>> tr = engine.transcribe("data/audio/v123.wav", video_id="v123")
    >>> tr.segments[0].start_ms
    0
    """

    def __init__(self, config: AsrConfig | None = None, *, model_manager: Any = None) -> None:
        self.config = config or AsrConfig()
        self.config.validate()
        self._model_manager = model_manager
        self._model: Any = None
        self._loaded_key: tuple[str, str, str] | None = None
        # D1：转写期 CUDA 运行库炸掉（典型：cublas64_12.dll 缺失，cuBLAS 惰性
        # 加载，加载期探测不到）后置 True，强制后续按 CPU 计划重载重试。
        self._force_cpu: bool = False

    # ---- 模型加载 --------------------------------------------------------- #
    def _manager(self) -> Any:
        if self._model_manager is None:
            from core.models.manager import get_model_manager

            self._model_manager = get_model_manager()
        return self._model_manager

    def _model_path(self) -> Path:
        return Path(self._manager().ensure_model(self.config.model_name))

    def _device_plan(self) -> list[tuple[str, str]]:
        """按优先级给出 ``(device, compute_type)`` 尝试序列。

        ``_force_cpu``（转写期 CUDA 运行库炸过一次）后不再给 CUDA 档 ——
        cuBLAS/cuDNN 是惰性加载，加载期的设备探测发现不了它缺失。
        """
        requested = (self.config.device or "auto").lower()
        explicit = self.config.compute_type
        plan: list[tuple[str, str]] = []

        def add(device: str, compute: str) -> None:
            if (device, compute) not in plan:
                plan.append((device, compute))

        if requested in {"cuda", "auto"} and not self._force_cpu:
            # auto 必须先问一句"CUDA 到底能不能用"。CTranslate2 的
            # get_cuda_device_count() 只数设备、不验证运行库：本机它是 1，
            # 但 cublas64_12.dll 缺失，于是 float16/int8_float16/int8 三档
            # 各失败一次才轮到 CPU，白白多花十几秒还打一堆红字。
            if requested == "cuda" or _cuda_usable():
                for compute in (explicit or "float16", "int8_float16", "int8"):
                    if compute:
                        add("cuda", compute)
        if requested in {"cpu", "auto", "mps"} or self._force_cpu:
            if requested == "mps" and not self._force_cpu:
                add("mps", explicit or "float32")
            for compute in (explicit or "int8", "float32"):
                if compute:
                    add("cpu", compute)
        return plan or [("cpu", "int8")]

    def load(self, *, force_reload: bool = False) -> Any:
        """加载（或复用）WhisperModel；失败按设备计划逐级降级。

        :raises AsrError: 所有设备/算力组合都失败
        """
        try:
            import faster_whisper  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise AsrError("未安装 faster-whisper，请执行 `pip install faster-whisper`") from exc

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
                    device, compute, model_path.name,
                )
                model = faster_whisper.WhisperModel(
                    str(model_path),
                    device=device,
                    compute_type=compute,
                    cpu_threads=self.config.cpu_threads,
                    num_workers=self.config.num_workers,
                    local_files_only=True,
                )
            except Exception as exc:  # noqa: BLE001 - 任何加载失败都尝试下一档
                message = f"{device}/{compute}: {type(exc).__name__}: {exc}"
                errors.append(message)
                logger.warning("模型加载失败，降级下一档 —— %s", message[:240])
                continue
            self._model = model
            self._loaded_key = (str(model_path), device, compute)
            logger.info(
                "Whisper 就绪 device=%s compute_type=%s（前面失败 %d 档）",
                device, compute, len(errors),
            )
            return model

        raise AsrError("Whisper 模型加载失败，所有设备档位均不可用:\n" + "\n".join(errors))

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
    ) -> Transcript:
        """转写入口：字幕优先，失败退回 Whisper，再失败返回空结果。

        :param subtitles: 候选字幕文件（yt-dlp 下载的 ``.vtt`` 等）
        :param duration_ms: 已知时长，用于填充 ``Transcript.duration_ms``
        """
        started = time.monotonic()
        use_subtitle = (
            self.config.fallback_to_subtitle if allow_subtitle is None else allow_subtitle
        )
        warnings: list[str] = []

        if use_subtitle:
            transcript = self._from_subtitles(video_id, subtitles, duration_ms, warnings)
            if transcript is not None:
                return replace(transcript, elapsed_ms=int((time.monotonic() - started) * 1000))

        transcript = self._from_whisper(
            audio_path, video_id=video_id, duration_ms=duration_ms, warnings=warnings
        )
        return replace(transcript, elapsed_ms=int((time.monotonic() - started) * 1000))

    def transcribe_subtitles_only(
        self, video_id: str, subtitles: Iterable[str | Path], *, duration_ms: int | None = None
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
        cues, hit, sub_warnings = load_subtitles(paths, preferred_order=self.config.subtitle_priority)
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

    def _segments_from_cues(self, cues: Sequence[SubtitleCue]) -> tuple[TranscriptSegment, ...]:
        segments: list[TranscriptSegment] = []
        previous = ""
        dropped = 0
        for cue in cues:
            text = _clean_text(cue.text)
            if _looks_like_hallucination(text):
                dropped += 1
                continue
            if previous and _similarity(previous, text) >= self.config.max_repeat_ratio:
                dropped += 1
                continue
            start = max(0, cue.start_ms)
            end = max(start + 1, cue.end_ms)
            segments.append(
                TranscriptSegment(index=len(segments), start_ms=start, end_ms=end, text=text)
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
    ) -> Transcript:
        path = Path(audio_path)
        if not path.is_file():
            warnings.append(f"音频文件不存在，跳过转写: {path}")
            return Transcript(
                video_id=video_id, source="empty", degraded=True, warnings=tuple(warnings)
            )
        try:
            model = self.load()
        except AsrError as exc:
            warnings.append(str(exc))
            return Transcript(
                video_id=video_id, source="empty", degraded=True, warnings=tuple(warnings)
            )

        device, compute = self.active_device
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
            "vad_parameters": {"min_silence_duration_ms": self.config.vad_min_silence_ms},
        }
        if self.config.language is None or self.config.language.startswith("zh"):
            options["initial_prompt"] = self.config.initial_prompt

        try:
            raw_segments, info = model.transcribe(str(path), **options)
        except Exception as exc:  # noqa: BLE001 - 推理层异常一律降级
            # D1：CUDA 运行库（cublas/cudnn）惰性加载，加载期探测不到，
            # 第一次真转写才炸。这里卸载重载到 CPU 再试一次 ——
            # "退回 CPU 慢点出结果"远好于"空转录让整条语音通道报废"。
            if device != "cpu" and not self._force_cpu:
                logger.warning(
                    "转写在 %s/%s 上失败（%s: %s），退回 CPU 重试一次",
                    device, compute, type(exc).__name__, str(exc)[:160],
                )
                self.unload()
                self._force_cpu = True
                try:
                    model = self.load()
                    device, compute = self.active_device
                    raw_segments, info = model.transcribe(str(path), **options)
                except Exception as exc2:  # noqa: BLE001 - CPU 也失败才算真失败
                    message = f"转写失败（CPU 回退后仍失败: {type(exc2).__name__}）: {exc2}"
                    logger.warning(message[:300])
                    warnings.append(message)
                    return Transcript(
                        video_id=video_id, source="empty", device=device,
                        compute_type=compute, model_name=self.config.model_name,
                        degraded=True, warnings=tuple(warnings),
                    )
                warnings.append(
                    f"CUDA 转写失败（{type(exc).__name__}），已回退 {device}/{compute} 完成转写"
                )
            else:
                message = f"转写失败（{type(exc).__name__}）: {exc}"
                logger.warning(message[:300])
                warnings.append(message)
                return Transcript(
                    video_id=video_id, source="empty", device=device,
                    compute_type=compute, model_name=self.config.model_name,
                    degraded=True, warnings=tuple(warnings),
                )
        segments_list = list(raw_segments)

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
            if previous and _similarity(previous, text) >= self.config.max_repeat_ratio:
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
                    no_speech_prob=self._float_or_none(getattr(raw, "no_speech_prob", None)),
                    compression_ratio=self._float_or_none(getattr(raw, "compression_ratio", None)),
                )
            )
            previous = text
            last_end = end

        if dropped:
            logger.info("Whisper 清洗丢弃 %d 段（幻觉/复读）", dropped)
        if not segments:
            warnings.append("转写结果为空（可能整段无人声）")

        total = duration_ms or int(round(float(getattr(info, "duration", 0.0) or 0.0) * 1000))
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
