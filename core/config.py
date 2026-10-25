"""统一配置中心。

设计要点
--------
1. **单一事实源**：``configs/app_config.yaml`` 是唯一配置载体，代码内只保留
   同名的兜底默认值，二者字段一一对应。
2. **三级覆盖**（高 → 低）::

       环境变量  ZH_YING__SECTION__KEY=...   >   app_config.yaml   >   dataclass 默认值

   环境变量用双下划线表达层级，支持任意深度::

       ZH_YING__CHUNK__MIN_MS=12000
       ZH_YING__RETRIEVAL__WEIGHTS__TEXT=1.0
3. **严格模式**：YAML 里出现 dataclass 没有声明的键直接报
   :class:`UnknownConfigKeyError`，避免拼写错误静默生效（这类 bug 极难排查）。
4. **不持有状态**：本模块只做"读 + 校验 + 类型转换"，不碰数据库、不建连接、
   不加载模型。进程级单例由 :func:`get_app_config` 提供。
5. **路径不写死**：所有数据目录来自 ``paths`` 段，相对路径按项目根解析，
   绝对路径原样使用。

典型用法::

    from core.config import get_app_config

    cfg = get_app_config()
    frames_dir = cfg.path("frames")          # -> Path, 已按项目根解析
    top_k = cfg.retrieval.top_k
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import threading
import types
from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence, Union, get_args, get_origin, get_type_hints

import yaml

logger = logging.getLogger("zhiying.core.config")

__all__ = [
    "ENV_PREFIX",
    "AppConfig",
    "ApiSection",
    "AppSection",
    "AsrSection",
    "ChunkSection",
    "ConfigError",
    "ConfigFileError",
    "ConfigValueError",
    "DownloadSection",
    "EmbeddingSection",
    "GradioSection",
    "ReportSection",
    "QaSection",
    "KeyframeSection",
    "LimitsSection",
    "LlmSection",
    "LoggingSection",
    "McpSection",
    "MediaSection",
    "ModelsSection",
    "PathsSection",
    "RetrievalSection",
    "SceneSection",
    "StorageSection",
    "TargetsSection",
    "UnknownConfigKeyError",
    "VlSection",
    "detect_project_root",
    "get_app_config",
    "load_app_config",
    "reload_app_config",
    "set_app_config",
]

# --------------------------------------------------------------------------- #
# 异常体系：每一层都能被单独捕获，便于逐级降级
# --------------------------------------------------------------------------- #
ENV_PREFIX: str = "ZH_YING"
"""环境变量前缀。层级用双下划线分隔，例如 ``ZH_YING__CHUNK__MIN_MS``。"""

_ENV_PATTERN = re.compile(rf"^{ENV_PREFIX}(?:__|_)([A-Za-z0-9_]+)$")
_CONFIG_REL = Path("configs") / "app_config.yaml"


class ConfigError(RuntimeError):
    """配置相关错误的基类。"""


class ConfigFileError(ConfigError):
    """配置文件缺失或不是合法 YAML。"""


class ConfigValueError(ConfigError, ValueError):
    """字段值无法转换成声明的类型。"""

    def __init__(self, path: str, expected: str, actual: Any) -> None:
        self.path = path
        self.expected = expected
        self.actual = actual
        super().__init__(f"配置项 {path} 期望 {expected}，实际 {type(actual).__name__}({actual!r})")


class UnknownConfigKeyError(ConfigError, KeyError):
    """配置里出现未声明的键（通常是拼写错误）。"""

    def __init__(self, path: str, key: str, known: Sequence[str]) -> None:
        self.path = path
        self.key = key
        self.known = tuple(known)
        super().__init__(
            f"未知配置项 {path}.{key}；该段合法键为: {', '.join(sorted(self.known))}"
        )


# --------------------------------------------------------------------------- #
# 各配置段
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class AppSection:
    name: str = "zhiying"
    version: str = "0.1.0"
    root: str = "."


@dataclass(frozen=True, slots=True)
class PathsSection:
    data: str = "data"
    models: str = "models"
    videos: str = "data/videos"
    audio: str = "data/audio"
    frames: str = "data/frames"
    temp: str = "data/temp"
    sqlite: str = "data/zhiying.db"
    chroma: str = "data/chroma"
    logs: str = "logs"
    exports: str = "data/exports"

    def keys(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self))


@dataclass(frozen=True, slots=True)
class ModelsSection:
    config_file: str = "configs/models.yaml"
    device: str = "auto"
    allow_cpu_fallback: bool = True
    max_loaded_models: int = 2
    # False = 只定位已下载权重，缺了就降级；True = 允许推理时就地下载。
    # 默认 False：下载统一走 scripts/download_models.py，别让"点一下分析"触发几 GB 拉取。
    auto_download: bool = False

    def validate(self) -> None:
        if self.device not in {"auto", "cuda", "cpu", "mps"}:
            raise ConfigValueError("models.device", "auto|cuda|cpu|mps", self.device)
        if self.max_loaded_models < 1:
            raise ConfigValueError("models.max_loaded_models", ">= 1", self.max_loaded_models)


@dataclass(frozen=True, slots=True)
class DownloadSection:
    max_retries: int = 3
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    socket_timeout_s: float = 30.0
    retries_per_fragment: int = 10
    concurrent_fragments: int = 4
    resume: bool = True
    cookies_from_browser: str | None = None
    max_url_length: int = 2048
    #: 下载源的最高分辨率。分析链路对清晰度的真实需求：关键帧目标短边 720
    #: （只缩不放）、视觉模型输入 448px、场景检测还会再降采样 —— 1080p 源
    #: 只是让下载多花两三倍时间。720p 同时也是 B 站免登录可得的最高档。
    format_max_height: int = 720

    def validate(self) -> None:
        if self.max_retries < 0:
            raise ConfigValueError("media.download.max_retries", ">= 0", self.max_retries)
        if self.concurrent_fragments < 1:
            raise ConfigValueError(
                "media.download.concurrent_fragments", ">= 1", self.concurrent_fragments
            )
        if self.max_url_length < 1:
            raise ConfigValueError("media.download.max_url_length", ">= 1", self.max_url_length)
        if self.format_max_height < 144:
            raise ConfigValueError("media.download.format_max_height", ">= 144",
                                   self.format_max_height)


@dataclass(frozen=True, slots=True)
class KeyframeSection:
    min_short_side: int = 720
    max_jpeg_bytes: int = 200 * 1024
    jpeg_quality_min: int = 55
    jpeg_quality_max: int = 92
    jpeg_subsampling: int = 2
    format: str = "jpg"

    def validate(self) -> None:
        if self.min_short_side < 1:
            raise ConfigValueError("media.scene_detect.keyframe.min_short_side", ">= 1",
                                   self.min_short_side)
        if self.max_jpeg_bytes < 1024:
            raise ConfigValueError("media.scene_detect.keyframe.max_jpeg_bytes", ">= 1024",
                                   self.max_jpeg_bytes)
        if not 1 <= self.jpeg_quality_min <= self.jpeg_quality_max <= 100:
            raise ConfigValueError(
                "media.scene_detect.keyframe.jpeg_quality_*", "1 <= min <= max <= 100",
                (self.jpeg_quality_min, self.jpeg_quality_max),
            )
        if self.format not in {"jpg", "jpeg", "png"}:
            raise ConfigValueError("media.scene_detect.keyframe.format", "jpg|jpeg|png",
                                   self.format)


@dataclass(frozen=True, slots=True)
class SceneSection:
    detector: str = "content"
    threshold: float = 27.0
    min_scene_len_ms: int = 1_200
    frame_skip: int = 0
    candidate_cap: int = 6
    frames_per_scene: int = 2
    min_frames_per_scene: int = 1
    blur_threshold: float = 60.0
    drop_blurry: bool = False
    dedupe_frames: bool = True
    dedupe_max_diff: float = 1.5
    dedupe_color_distance: float = 30.0
    dedupe_probe_side: int = 64
    dedupe_grid: int = 4
    max_workers: int = 4
    keep_probe_dir: bool = False
    # ---- 长视频检测加速（D4）----
    # 时长超过该值的视频，检测时提高 frame_skip/downscale（切点精度略降，
    # 换取"检测耗时不超过视频时长"）。0 = 关闭。
    long_video_min_ms: int = 300_000
    long_video_frame_skip: int = 2
    long_video_downscale: int = 4
    # 长视频改走 ffmpeg 检测档：NVDEC 硬解 + 480px 降采样打分（实测比
    # PySceneDetect 逐帧软解快一个量级）。false 则维持 pyscedetect 加速档。
    long_video_use_ffmpeg: bool = True
    long_video_hwaccel: bool = True
    long_video_detect_width: int = 480
    keyframe: KeyframeSection = field(default_factory=KeyframeSection)

    def validate(self) -> None:
        if self.detector not in {"content", "adaptive", "hash", "ffmpeg", "fixed"}:
            raise ConfigValueError("media.scene_detect.detector",
                                   "content|adaptive|hash|ffmpeg|fixed", self.detector)
        if self.min_scene_len_ms < 1:
            raise ConfigValueError("media.scene_detect.min_scene_len_ms", ">= 1",
                                   self.min_scene_len_ms)
        if not 1 <= self.min_frames_per_scene <= self.frames_per_scene:
            raise ConfigValueError(
                "media.scene_detect.frames_per_scene", ">= min_frames_per_scene >= 1",
                self.frames_per_scene,
            )
        if self.candidate_cap < self.frames_per_scene:
            raise ConfigValueError(
                "media.scene_detect.candidate_cap", ">= frames_per_scene", self.candidate_cap
            )
        if self.max_workers < 1:
            raise ConfigValueError("media.scene_detect.max_workers", ">= 1", self.max_workers)
        if self.threshold <= 0:
            raise ConfigValueError("media.scene_detect.threshold", "> 0", self.threshold)
        self.keyframe.validate()

    def flattened(self) -> dict[str, Any]:
        """摊平成 ``SceneDetectConfig(**kwargs)`` 可直接消费的字典。"""
        data = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "keyframe"}
        data.update({f.name: getattr(self.keyframe, f.name) for f in fields(self.keyframe)})
        return data


@dataclass(frozen=True, slots=True)
class MediaSection:
    audio_sample_rate: int = 16_000
    audio_channels: int = 1
    audio_codec: str = "pcm_s16le"
    audio_format: str = "wav"
    ffmpeg_timeout_ms: int = 1_800_000
    ffprobe_timeout_ms: int = 30_000
    download: DownloadSection = field(default_factory=DownloadSection)
    scene_detect: SceneSection = field(default_factory=SceneSection)

    def validate(self) -> None:
        if self.audio_sample_rate != 16_000:
            # faster-whisper 内部固定按 16k 处理，其它采样率只会引入重采样误差
            raise ConfigValueError("media.audio_sample_rate", "16000", self.audio_sample_rate)
        if self.audio_channels != 1:
            raise ConfigValueError("media.audio_channels", "1", self.audio_channels)
        if self.ffmpeg_timeout_ms < 1_000:
            raise ConfigValueError("media.ffmpeg_timeout_ms", ">= 1000", self.ffmpeg_timeout_ms)
        self.download.validate()
        self.scene_detect.validate()


@dataclass(frozen=True, slots=True)
class AsrSection:
    model: str = "faster-whisper-large-v3-turbo"
    device: str | None = None
    compute_type: str | None = None
    beam_size: int = 5
    language: str | None = None
    vad_filter: bool = True
    vad_min_silence_ms: int = 500
    subtitle_priority: tuple[str, ...] = ("srt", "vtt", "ass", "ssa")
    fallback_to_subtitle: bool = True
    min_language_probability: float = 0.5

    def validate(self) -> None:
        if self.device not in {None, "cuda", "cpu", "mps"}:
            raise ConfigValueError("asr.device", "null|cuda|cpu|mps", self.device)
        if self.beam_size < 1:
            raise ConfigValueError("asr.beam_size", ">= 1", self.beam_size)
        if not 0.0 <= self.min_language_probability <= 1.0:
            raise ConfigValueError("asr.min_language_probability", "0..1",
                                   self.min_language_probability)


@dataclass(frozen=True, slots=True)
class VlSection:
    model: str = "qwen2-vl-2b-gguf"
    max_new_tokens: int = 512
    temperature: float = 0.1
    max_image_side: int = 448
    prompt_key: str = "keyframe_describe"
    max_retries: int = 2
    batch_size: int = 1
    # 多帧一次推理的帧数上限。实测 2 帧最优；4 帧起 2B 模型为了"找差异"
    # 会编造画面里不存在的内容（同一页被描述成三张不同幻灯片）。
    sequence_frames: int = 2
    sequence_temperature: float = 0.1
    # 按场景分组做连续理解；关掉退回逐帧独立描述（旧行为）
    sequence_mode: bool = True
    # -- 云端视觉（OpenAI 兼容协议）--------------------------------------
    # 打开后优先走云端，本地 GGUF 作为回退。界面里可以填地址/模型/密钥。
    cloud_enabled: bool = False
    cloud_base_url: str = ""
    cloud_model: str = "qwen-vl-plus"
    cloud_api_key: str = ""
    cloud_api_key_env: str = "ZH_YING_VL_API_KEY"
    cloud_timeout_s: int = 60
    cloud_max_retries: int = 1
    # 云端没这个限制（2B 模型的毛病），放宽到 8；本地仍锁 1..4
    cloud_sequence_frames: int = 8
    # ---- 视频段理解（云端 qwen-vl 直接吃短视频段，见 docs/演进计划）----
    #: auto=按场景密度分派（讲解→帧，运动→段）；frames=只用抽帧；segments=只用分段
    video_mode: str = "auto"
    #: 单段时长上限（秒）：超出按此再剪；也决定上传体积（480p 重编码后约 1~5MB）
    video_segment_max_s: int = 15
    #: 每视频最多走"段"的场景数（费用闸门），超出退回抽帧
    video_segment_budget: int = 40
    #: 场景密度超过该值（次/分钟）判为高运动视频：auto 下走段/降档
    high_motion_density_per_min: float = 20.0
    #: 高运动降档时把相邻场景合并到该跨度（毫秒）再抽帧，减少视觉调用量
    high_motion_scene_span_ms: int = 10_000

    def validate(self) -> None:
        if self.max_new_tokens < 1:
            raise ConfigValueError("vl.max_new_tokens", ">= 1", self.max_new_tokens)
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigValueError("vl.temperature", "0..2", self.temperature)
        if self.max_image_side < 224:
            raise ConfigValueError("vl.max_image_side", ">= 224", self.max_image_side)
        if not 1 <= self.sequence_frames <= 4:
            raise ConfigValueError("vl.sequence_frames", "1..4", self.sequence_frames)
        if not 1 <= self.cloud_sequence_frames <= 32:
            raise ConfigValueError("vl.cloud_sequence_frames", "1..32",
                                   self.cloud_sequence_frames)
        if self.cloud_enabled and not self.cloud_model:
            raise ConfigValueError("vl.cloud_model", "非空（cloud_enabled=true 时）",
                                   self.cloud_model)
        if self.video_mode not in {"auto", "frames", "segments"}:
            raise ConfigValueError("vl.video_mode", "auto|frames|segments", self.video_mode)
        if not 5 <= self.video_segment_max_s <= 60:
            raise ConfigValueError("vl.video_segment_max_s", "5..60", self.video_segment_max_s)
        if self.video_segment_budget < 1:
            raise ConfigValueError("vl.video_segment_budget", ">= 1", self.video_segment_budget)
        if self.high_motion_density_per_min <= 0:
            raise ConfigValueError("vl.high_motion_density_per_min", "> 0",
                                   self.high_motion_density_per_min)
        if self.high_motion_scene_span_ms < 2_000:
            raise ConfigValueError("vl.high_motion_scene_span_ms", ">= 2000",
                                   self.high_motion_scene_span_ms)


@dataclass(frozen=True, slots=True)
class EmbeddingSection:
    text_model: str = "bge-m3"
    image_model: str = "siglip2-base"
    text_dim: int = 1024
    image_dim: int = 768
    batch_size: int = 16
    normalize: bool = True
    max_text_chars: int = 8192

    def validate(self) -> None:
        if self.text_dim < 1 or self.image_dim < 1:
            raise ConfigValueError("embedding.text_dim/image_dim", ">= 1",
                                   (self.text_dim, self.image_dim))
        if self.batch_size < 1:
            raise ConfigValueError("embedding.batch_size", ">= 1", self.batch_size)


@dataclass(frozen=True, slots=True)
class Bm25Section:
    k1: float = 1.5
    b: float = 0.75
    tokenizer: str = "jieba"
    stopwords: tuple[str, ...] = (
        "的", "了", "是", "在", "和", "也", "就", "都", "而", "及", "与",
    )

    def validate(self) -> None:
        if self.k1 <= 0:
            raise ConfigValueError("retrieval.bm25.k1", "> 0", self.k1)
        if not 0.0 <= self.b <= 1.0:
            raise ConfigValueError("retrieval.bm25.b", "0..1", self.b)
        if self.tokenizer not in {"jieba", "char", "naive"}:
            raise ConfigValueError("retrieval.bm25.tokenizer", "jieba|char|naive",
                                   self.tokenizer)


@dataclass(frozen=True, slots=True)
class RetrievalSection:
    top_k: int = 6
    fetch_multiplier: int = 4
    rrf_k: int = 60
    #: 向量库后端：``chroma``（HNSW，需要 chromadb）或 ``sqlite``（零依赖，
    #: 线性扫描）。需求文档 2.4.3.4 要求两种都支持。
    backend: str = "chroma"
    weights: Mapping[str, float] = field(
        default_factory=lambda: {"text": 1.0, "image": 0.8, "bm25": 0.6}
    )
    min_score: float = 0.05
    channel_timeout_ms: int = 3_000
    max_clip_chars: int = 1_000
    include_keyframes: bool = True
    bm25: Bm25Section = field(default_factory=Bm25Section)

    def validate(self) -> None:
        if self.top_k < 1:
            raise ConfigValueError("retrieval.top_k", ">= 1", self.top_k)
        if self.fetch_multiplier < 1:
            raise ConfigValueError("retrieval.fetch_multiplier", ">= 1", self.fetch_multiplier)
        if self.rrf_k < 1:
            raise ConfigValueError("retrieval.rrf_k", ">= 1", self.rrf_k)
        if str(self.backend).lower() not in {"chroma", "sqlite"}:
            raise ConfigValueError("retrieval.backend", "chroma 或 sqlite", self.backend)
        if not self.weights:
            raise ConfigValueError("retrieval.weights", "非空映射", self.weights)
        for name, weight in self.weights.items():
            if weight < 0:
                raise ConfigValueError(f"retrieval.weights.{name}", ">= 0", weight)
        if sum(self.weights.values()) <= 0:
            raise ConfigValueError("retrieval.weights", "权重和 > 0", dict(self.weights))
        if not 0.0 <= self.min_score <= 1.0:
            raise ConfigValueError("retrieval.min_score", "0..1", self.min_score)
        self.bm25.validate()


@dataclass(frozen=True, slots=True)
class ChunkSection:
    """切片策略。**只放真正接进** :class:`core.rag.chunker.ChunkConfig` **的键**。

    早期版本这里还躺着 ``sentence_end`` / ``clause_end`` / ``weight_sentence`` /
    ``weight_clause`` / ``weight_silence`` / ``weight_target`` 六个字段，但
    chunker 的打分实现用的是模块内常量 ``_BOUNDARY_WEIGHTS`` 与
    ``_SOFT_BREAKS``，那六个键改了也不生效 —— 配置项骗人比没有更糟，
    已随 ``as_chunk_config()`` 一起下掉。要调边界权重请改 chunker 的常量或
    先给 ``ChunkConfig`` 加字段。
    """

    min_ms: int = 10_000
    max_ms: int = 30_000
    target_ms: int = 20_000
    overlap_ms: int = 1_500
    silence_ms: int = 600
    weight_tail_penalty: float = 0.6
    hard_split_ms: int = 30_000
    max_chars: int = 1_000

    def validate(self) -> None:
        if not 1_000 <= self.min_ms < self.max_ms:
            raise ConfigValueError("chunk.min_ms/max_ms", "1000 <= min < max",
                                   (self.min_ms, self.max_ms))
        if not self.min_ms <= self.target_ms <= self.max_ms:
            raise ConfigValueError("chunk.target_ms", "min <= target <= max", self.target_ms)
        if self.overlap_ms < 0 or self.overlap_ms * 2 >= self.min_ms:
            raise ConfigValueError("chunk.overlap_ms", "0 <= 2*overlap < min_ms",
                                   self.overlap_ms)
        if self.silence_ms < 0:
            raise ConfigValueError("chunk.silence_ms", ">= 0", self.silence_ms)
        if self.hard_split_ms < self.min_ms:
            raise ConfigValueError("chunk.hard_split_ms", ">= min_ms", self.hard_split_ms)
        if self.max_chars < 1:
            raise ConfigValueError("chunk.max_chars", ">= 1", self.max_chars)

    def as_chunk_config(self) -> dict[str, Any]:
        """``ChunkConfig(**cfg.chunk.as_chunk_config())`` 用的字典。

        两侧字段名并不一致（``silence_ms`` ↔ ``silence_threshold_ms``、
        ``hard_split_ms`` ↔ ``hard_max_ms``），所以这里做**显式映射**而不是
        ``{f.name: getattr(...)}`` —— 后者会把不存在的键灌进 dataclass 直接
        ``TypeError``，这个坑踩过一次。
        """
        return {
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "target_ms": self.target_ms,
            "overlap_ms": self.overlap_ms,
            "silence_threshold_ms": self.silence_ms,
            "hard_max_ms": self.hard_split_ms,
            "weight_tail_penalty": self.weight_tail_penalty,
            "max_chars": self.max_chars,
        }


@dataclass(frozen=True, slots=True)
class CloudLlmSection:
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    #: 界面上填的密钥。留空时回退读环境变量 ``api_key_env``。
    #: 存在 ``data/user_settings.json``（已在 .gitignore 里），不进版本控制。
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: int = 60
    max_retries: int = 2
    temperature: float = 0.3
    max_tokens: int = 2048

    def resolved_key(self) -> str:
        """密钥取值：配置里的明文优先，其次环境变量。"""
        key = (self.api_key or "").strip()
        if key:
            return key
        if self.api_key_env:
            return (os.environ.get(self.api_key_env) or "").strip()
        return ""


@dataclass(frozen=True, slots=True)
class LocalLlmSection:
    model: str = "qwen2.5-7b-gguf"
    n_ctx: int = 8192
    n_threads: int | None = None
    n_gpu_layers: int = -1
    temperature: float = 0.3
    max_tokens: int = 2048

    def validate(self) -> None:
        if self.n_ctx < 512:
            raise ConfigValueError("llm.local.n_ctx", ">= 512", self.n_ctx)


@dataclass(frozen=True, slots=True)
class LlmSection:
    mode: str = "auto"
    cloud: CloudLlmSection = field(default_factory=CloudLlmSection)
    local: LocalLlmSection = field(default_factory=LocalLlmSection)
    fallback_keywords: bool = True
    fallback_top_keywords: int = 20

    def validate(self) -> None:
        if self.mode not in {"auto", "cloud", "local"}:
            raise ConfigValueError("llm.mode", "auto|cloud|local", self.mode)
        self.local.validate()
        if self.fallback_top_keywords < 1:
            raise ConfigValueError("llm.fallback_top_keywords", ">= 1",
                                   self.fallback_top_keywords)


@dataclass(frozen=True, slots=True)
class SqliteSection:
    journal_mode: str = "WAL"
    busy_timeout_ms: int = 5_000
    pool_size: int = 4

    def validate(self) -> None:
        if self.journal_mode.upper() not in {"WAL", "DELETE", "TRUNCATE", "PERSIST", "MEMORY"}:
            raise ConfigValueError("storage.sqlite.journal_mode",
                                   "WAL|DELETE|TRUNCATE|PERSIST|MEMORY", self.journal_mode)


_CHROMA_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")


@dataclass(frozen=True, slots=True)
class ChromaSection:
    clip_collection: str = "clip_text"
    frame_collection: str = "frame_image"
    metric: str = "cosine"
    verify_dim: bool = True

    def validate(self) -> None:
        for name in (self.clip_collection, self.frame_collection):
            if not _CHROMA_NAME.match(name) or ".." in name:
                raise ConfigValueError(
                    "storage.chroma.<collection>",
                    "3~512 字符，仅 [a-zA-Z0-9._-]，首尾为字母数字且不含 '..'",
                    name,
                )
        if self.clip_collection == self.frame_collection:
            raise ConfigValueError("storage.chroma.<collection>", "两个集合名不能相同",
                                   self.clip_collection)
        if self.metric not in {"cosine", "l2", "ip"}:
            raise ConfigValueError("storage.chroma.metric", "cosine|l2|ip", self.metric)


@dataclass(frozen=True, slots=True)
class CapacitySection:
    max_videos: int = 200
    max_clips: int = 50_000
    max_transcript_segments: int = 500_000
    max_keyframes: int = 100_000
    max_keyframe_bytes: int = 21_474_836_480
    max_text_vectors: int = 50_000
    max_image_vectors: int = 100_000
    max_result_bytes: int = 5_368_709_120
    #: 分析产物（报告/导出文件）的登记行数上限；tasks 是簿记表不设硬闸，
    #: 由 ``Repository.prune_finished_tasks`` 按条数裁剪。
    max_artifacts: int = 5_000


@dataclass(frozen=True, slots=True)
class RetentionSection:
    """数据保留策略。由 ``scripts/gc.py`` 消费，服务进程自身不做定时删除。"""

    temp_hours: int = 24
    orphan_frames_days: int = 30
    #: 分析完成后保留多少天（0 = 永不按时间删）。到期后元数据、切片、关键帧、
    #: 音频、向量、报告一次性删干净 —— 需求文档表 3 的容量红线靠它维持。
    video_ttl_days: int = 0
    #: 容量触顶时从最久完成的分析开始回收，而不是让新任务直接失败。
    auto_prune: bool = True

    def validate(self) -> None:
        for name, value in (
            ("temp_hours", self.temp_hours),
            ("orphan_frames_days", self.orphan_frames_days),
            ("video_ttl_days", self.video_ttl_days),
        ):
            if value < 0:
                raise ConfigValueError(f"storage.retention.{name}", ">= 0", value)


@dataclass(frozen=True, slots=True)
class StorageSection:
    sqlite: SqliteSection = field(default_factory=SqliteSection)
    chroma: ChromaSection = field(default_factory=ChromaSection)
    limits: CapacitySection = field(default_factory=CapacitySection)
    retention: RetentionSection = field(default_factory=RetentionSection)

    def validate(self) -> None:
        self.sqlite.validate()
        self.chroma.validate()
        self.retention.validate()


@dataclass(frozen=True, slots=True)
class ApiSection:
    host: str = "127.0.0.1"
    port: int = 8000
    max_body_bytes: int = 4_294_967_296
    sse_heartbeat_s: int = 15
    auth_token_env: str = "ZHIYING_API_TOKEN"
    cors_origins: tuple[str, ...] = (
        "http://127.0.0.1:7860",
        "http://localhost:7860",
    )
    request_timeout_s: int = 300
    #: 把 Gradio 界面临时挂到本服务的 ``/ui``，让 ``http://host:port`` 打开就是界面。
    #: 关掉后根路径只返回一行提示，界面改走独立的 ``clients.gradio_ui``（7860）。
    serve_ui: bool = True

    def validate(self) -> None:
        if not 1 <= self.port <= 65_535:
            raise ConfigValueError("api.port", "1..65535", self.port)
        if self.sse_heartbeat_s < 1:
            raise ConfigValueError("api.sse_heartbeat_s", ">= 1", self.sse_heartbeat_s)


@dataclass(frozen=True, slots=True)
class McpSection:
    transport: str = "stdio"
    server_name: str = "zhiying"
    server_version: str = "0.1.0"
    call_timeout_s: int = 30

    def validate(self) -> None:
        if self.transport not in {"stdio", "http"}:
            raise ConfigValueError("mcp.transport", "stdio|http", self.transport)


@dataclass(frozen=True, slots=True)
class GradioSection:
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False
    theme: str = "soft"
    page_size: int = 20

    def validate(self) -> None:
        if not 1 <= self.port <= 65_535:
            raise ConfigValueError("gradio.port", "1..65535", self.port)


@dataclass(frozen=True, slots=True)
class LimitsSection:
    max_url_length: int = 2048
    max_duration_ms: int = 7_200_000
    max_file_bytes: int = 4_294_967_296
    max_question_chars: int = 500
    max_report_chars: int = 5_000
    min_keyframe_short_side: int = 720
    timestamp_tolerance_ms: int = 1_000
    keyframe_tolerance_ms: int = 100
    citation_tolerance_ms: int = 2_000
    #: 重模型串行：同一进程里同时只让一个 ASR / 视觉 / 向量 / 本地 LLM 干活。
    #: 关掉它可以让两个任务并行，但两套权重同时驻留会把 8G 机器挤到换页，
    #: 实测比串行更慢。除非机器内存 32G 以上，否则别关。
    inference_serial: bool = True

    def validate(self) -> None:
        for name in ("max_url_length", "max_duration_ms", "max_file_bytes",
                     "max_question_chars", "max_report_chars", "min_keyframe_short_side"):
            if getattr(self, name) < 1:
                raise ConfigValueError(f"limits.{name}", ">= 1", getattr(self, name))


@dataclass(frozen=True, slots=True)
class TargetsSection:
    parse_1h_cpu_ms: int = 1_800_000
    parse_1h_gpu_ms: int = 600_000
    asr_1h_cpu_ms: int = 2_400_000
    asr_1h_gpu_ms: int = 600_000
    qa_first_response_ms: int = 10_000
    qa_first_token_ms: int = 3_000
    mcp_call_ms: int = 30_000
    ui_response_ms: int = 200
    zh_cer_max: float = 0.15
    en_wer_max: float = 0.20


@dataclass(frozen=True, slots=True)
class QaSection:
    """多轮问答（C 端对话形态，见 core/pipeline/chat_sessions.py）。"""

    #: 多轮：把最近几轮问答作为 LLM 前缀。关掉即退回单轮问答。
    multi_turn: bool = True
    #: 查询改写：检索前把"那第二个进球呢"这类追问扩写成独立完整查询。
    query_rewrite: bool = True
    #: 每视频保留的对话轮数（内存 LRU，进程重启即清）。
    history_window: int = 8

    def validate(self) -> None:
        if self.history_window < 1:
            raise ConfigValueError("qa.history_window", ">= 1", self.history_window)


@dataclass(frozen=True, slots=True)
class ReportSection:
    """三层报告（改进 A/B）：综合总结 / 内容驱动的脉络 / 修字后的解说词。"""

    #: LLM 修正转写错字（保留原话，不增删）。False 时解说词直接用原文。
    polish_transcript: bool = True
    #: 每章最多提取的关键节点数（改进 B：内容驱动，禁止均匀切分）
    milestones_max_per_chapter: int = 5
    #: 单次送 LLM 修字的最大转写长度（字符），超出按章内再分块
    polish_block_chars: int = 4_000

    def validate(self) -> None:
        if self.milestones_max_per_chapter < 1:
            raise ConfigValueError("report.milestones_max_per_chapter", ">= 1",
                                   self.milestones_max_per_chapter)
        if self.polish_block_chars < 500:
            raise ConfigValueError("report.polish_block_chars", ">= 500",
                                   self.polish_block_chars)


@dataclass(frozen=True, slots=True)
class LoggingSection:
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file: str = "logs/zhiying.log"
    max_bytes: int = 10_485_760
    backup_count: int = 5
    redact_keys: tuple[str, ...] = (
        "api_key", "token", "password", "secret", "authorization", "cookie",
    )
    local_mode_strict: bool = True

    def validate(self) -> None:
        if self.level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigValueError("logging.level", "DEBUG|INFO|WARNING|ERROR|CRITICAL",
                                   self.level)
        if self.max_bytes < 1024:
            raise ConfigValueError("logging.max_bytes", ">= 1024", self.max_bytes)


# --------------------------------------------------------------------------- #
# 顶层配置
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class AppConfig:
    """完整的应用配置快照（不可变，线程安全共享）。"""

    app: AppSection = field(default_factory=AppSection)
    paths: PathsSection = field(default_factory=PathsSection)
    models: ModelsSection = field(default_factory=ModelsSection)
    media: MediaSection = field(default_factory=MediaSection)
    asr: AsrSection = field(default_factory=AsrSection)
    vl: VlSection = field(default_factory=VlSection)
    embedding: EmbeddingSection = field(default_factory=EmbeddingSection)
    retrieval: RetrievalSection = field(default_factory=RetrievalSection)
    chunk: ChunkSection = field(default_factory=ChunkSection)
    llm: LlmSection = field(default_factory=LlmSection)
    storage: StorageSection = field(default_factory=StorageSection)
    api: ApiSection = field(default_factory=ApiSection)
    mcp: McpSection = field(default_factory=McpSection)
    gradio: GradioSection = field(default_factory=GradioSection)
    report: ReportSection = field(default_factory=ReportSection)
    qa: QaSection = field(default_factory=QaSection)
    limits: LimitsSection = field(default_factory=LimitsSection)
    targets: TargetsSection = field(default_factory=TargetsSection)
    logging: LoggingSection = field(default_factory=LoggingSection)
    # 运行时注入，不来自 YAML
    root_dir: Path = field(default_factory=lambda: Path.cwd(), repr=False, compare=False)
    source_file: Path | None = field(default=None, repr=False, compare=False)

    # -- 路径解析 ----------------------------------------------------------- #
    def path(self, key: str) -> Path:
        """取 ``paths.<key>`` 的绝对路径。

        ``data`` 这类容器目录返回目录本身；``sqlite`` 返回文件路径。
        """
        known = self.paths.keys()
        if key not in known:
            raise UnknownConfigKeyError("paths", key, known)
        raw = getattr(self.paths, key)
        candidate = Path(os.path.expanduser(raw))
        if candidate.is_absolute():
            return candidate
        return (self.root_dir / candidate).resolve()

    def all_paths(self) -> dict[str, Path]:
        return {key: self.path(key) for key in self.paths.keys()}

    def ensure_dirs(self) -> dict[str, Path]:
        """创建所有目录类路径（``sqlite`` 只创建父目录）。"""
        created: dict[str, Path] = {}
        for key in self.paths.keys():
            target = self.path(key)
            if key == "sqlite":
                target.parent.mkdir(parents=True, exist_ok=True)
                created[key] = target
                continue
            target.mkdir(parents=True, exist_ok=True)
            created[key] = target
        return created

    # -- 校验 --------------------------------------------------------------- #
    def validate(self) -> None:
        """跨段一致性校验；任何不一致都抛 :class:`ConfigValueError`。"""
        self.models.validate()
        self.media.validate()
        self.asr.validate()
        self.vl.validate()
        self.embedding.validate()
        self.retrieval.validate()
        self.chunk.validate()
        self.llm.validate()
        self.storage.validate()
        self.api.validate()
        self.mcp.validate()
        self.gradio.validate()
        self.report.validate()
        self.qa.validate()
        self.limits.validate()
        self.logging.validate()

        # 跨段：关键帧短边不得低于全局下限
        if self.media.scene_detect.keyframe.min_short_side < self.limits.min_keyframe_short_side:
            logger.warning(
                "keyframe.min_short_side(%d) 低于 limits.min_keyframe_short_side(%d)，"
                "将以 limits 为准触发告警",
                self.media.scene_detect.keyframe.min_short_side,
                self.limits.min_keyframe_short_side,
            )
        # 跨段：切片硬切长度不得超过 chunk 上限
        if self.chunk.hard_split_ms > self.chunk.max_ms:
            raise ConfigValueError("chunk.hard_split_ms", "<= chunk.max_ms",
                                   self.chunk.hard_split_ms)
        # 跨段：下载 URL 上限与全局限制保持一致
        if self.media.download.max_url_length != self.limits.max_url_length:
            logger.info(
                "media.download.max_url_length(%d) != limits.max_url_length(%d)，"
                "下载器将取二者较小值",
                self.media.download.max_url_length,
                self.limits.max_url_length,
            )
        # 跨段：SQLite 与 Chroma 目录不能相同
        if self.path("sqlite").parent == self.path("chroma"):
            raise ConfigValueError("paths.sqlite/paths.chroma", "不应指向同一目录",
                                   str(self.path("chroma")))

    # -- 序列化 ------------------------------------------------------------- #
    def to_dict(self, *, include_runtime: bool = False) -> dict[str, Any]:
        data = _to_plain(self)
        if not include_runtime:
            data.pop("root_dir", None)
            data.pop("source_file", None)
        return data

    def diff_from_defaults(self) -> dict[str, tuple[Any, Any]]:
        """返回与 dataclass 默认值不同的字段，形如 ``{"chunk.min_ms": (10000, 12000)}``。"""
        default = AppConfig()
        flat_default = _flatten(default.to_dict(include_runtime=False))
        flat_actual = _flatten(self.to_dict(include_runtime=False))
        diff: dict[str, tuple[Any, Any]] = {}
        for key, value in flat_actual.items():
            if key in flat_default and flat_default[key] != value:
                diff[key] = (flat_default[key], value)
        return diff


# --------------------------------------------------------------------------- #
# YAML / env 读取
# --------------------------------------------------------------------------- #
def detect_project_root(start: str | Path | None = None) -> Path:
    """向上查找含 ``configs/app_config.yaml`` 的目录；找不到则返回起点。"""
    env_root = os.environ.get(f"{ENV_PREFIX}_PROJECT_ROOT")
    if env_root:
        return Path(os.path.expanduser(env_root)).resolve()
    origin = Path(start or Path.cwd()).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / _CONFIG_REL).is_file():
            return candidate
    return origin


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigFileError(f"配置文件不存在: {path}") from exc
    except OSError as exc:
        raise ConfigFileError(f"配置文件读取失败: {path} ({exc})") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigFileError(f"YAML 解析失败: {path}\n{exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigFileError(f"配置文件顶层必须是映射，实际为 {type(data).__name__}: {path}")
    return dict(data)


_DOTENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def _load_dotenv(path: Path, environ: MutableMapping[str, str]) -> int:
    """极简 .env 解析；已存在的环境变量不覆盖。返回注入条数。"""
    if not path.is_file():
        return 0
    injected = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _DOTENV_LINE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        if value and value[0] in {"'", '"'} and value[-1:] == value[0]:
            value = value[1:-1]
        else:
            value = value.split("#", 1)[0].strip()
        if key in environ:
            continue
        environ[key] = value
        injected += 1
    return injected


def _collect_env_overrides(environ: Mapping[str, str]) -> dict[tuple[str, ...], str]:
    """把 ``ZH_YING__A__B=1`` 收集成 ``{("a","b"): "1"}``。"""
    overrides: dict[tuple[str, ...], str] = {}
    for key, value in environ.items():
        match = _ENV_PATTERN.match(key)
        if not match:
            continue
        parts = tuple(part.lower() for part in match.group(1).split("__") if part)
        # 单下划线形式 `ZH_YING_CHUNK_MIN_MS` 无法无歧义拆分，只支持双下划线
        if len(parts) < 2:
            continue
        overrides[parts] = value
    return overrides


def _deep_set(data: MutableMapping[str, Any], path: Sequence[str], value: Any) -> None:
    cursor: MutableMapping[str, Any] = data
    for part in path[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[path[-1]] = value


def user_settings_path(root: str | Path | None = None) -> Path:
    """界面设置的落盘位置：``<root>/data/user_settings.json``。"""
    base = Path(root) if root is not None else detect_project_root()
    return Path(base) / "data" / "user_settings.json"


def _load_user_settings(path: Path) -> dict[str, Any]:
    """读取界面设置。文件不存在/损坏都只记日志，绝不让配置加载失败。"""
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("界面设置文件 %s 读取失败，已忽略: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("界面设置文件 %s 结构不是对象，已忽略", path)
        return {}

    flat: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not key.strip():
            continue
        # 允许两种写法：``{"llm.mode": "cloud"}`` 或嵌套 ``{"llm": {"mode": ...}}``
        if isinstance(value, dict):
            for sub, sub_value in value.items():
                if isinstance(sub, str) and not isinstance(sub_value, (dict, list)):
                    flat[f"{key}.{sub}"] = sub_value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            if value is not None:
                flat[key] = value
    return flat


# --------------------------------------------------------------------------- #
# 类型转换与 dataclass 构建
# --------------------------------------------------------------------------- #
_HINTS_CACHE: dict[type, dict[str, Any]] = {}
_TRUE = {"1", "true", "yes", "on", "y", "t"}
_FALSE = {"0", "false", "no", "off", "n", "f", ""}


def _hints(cls: type) -> dict[str, Any]:
    cached = _HINTS_CACHE.get(cls)
    if cached is None:
        cached = get_type_hints(cls)
        _HINTS_CACHE[cls] = cached
    return cached


def _coerce(value: Any, tp: Any, path: str) -> Any:
    """把 YAML/环境变量里的值转换成目标类型。"""
    origin = get_origin(tp)
    args = get_args(tp)

    # Optional[T] / T | None
    if origin is types.UnionType or origin is Union:
        if value is None:
            return None
        for branch in args:
            if branch is type(None):
                continue
            try:
                return _coerce(value, branch, path)
            except ConfigValueError:
                continue
        joined = " | ".join(str(a) for a in args if a is not type(None))
        raise ConfigValueError(path, joined, value)

    if value is None:
        raise ConfigValueError(path, str(tp), None)

    if tp is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        if isinstance(value, str):
            low = value.strip().lower()
            if low in _TRUE:
                return True
            if low in _FALSE:
                return False
        raise ConfigValueError(path, "bool", value)

    if tp is int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and float(value).is_integer():
            return int(value)
        if isinstance(value, str):
            text = value.strip().replace("_", "")
            try:
                return int(text, 0) if text[:2].lower() in {"0x", "0b", "0o"} else int(text)
            except ValueError:
                pass
        raise ConfigValueError(path, "int", value)

    if tp is float:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip().replace("_", ""))
            except ValueError:
                pass
        raise ConfigValueError(path, "float", value)

    if tp is str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, Path):
            return str(value)
        raise ConfigValueError(path, "str", value)

    if tp is Path:
        return Path(str(value))

    if origin in (tuple, list, set, frozenset):
        inner = args[0] if args else str
        if isinstance(value, str):
            text = value.strip()
            items: list[Any] = [] if not text else [
                part.strip() for part in re.split(r"[,;，；]", text) if part.strip()
            ]
        elif isinstance(value, (list, tuple, set, frozenset)):
            items = list(value)
        else:
            raise ConfigValueError(path, "list/tuple", value)
        converted = [_coerce(item, inner, f"{path}[{index}]") for index, item in enumerate(items)]
        if origin is tuple:
            return tuple(converted)
        if origin is list:
            return converted
        if origin is set:
            return set(converted)
        return frozenset(converted)

    if origin in (dict, AbcMapping) or tp is dict:
        key_type, value_type = (args + (str, Any))[:2] if args else (str, Any)
        if not isinstance(value, Mapping):
            raise ConfigValueError(path, "mapping", value)
        return {
            _coerce(k, key_type, f"{path}.<key>"): _coerce(v, value_type, f"{path}.{k}")
            for k, v in value.items()
        }

    if tp is Any:
        return value

    if is_dataclass(tp):
        return _build_dataclass(tp, value, path, strict=True)

    raise ConfigValueError(path, f"不支持的类型 {tp!r}", value)


def _build_dataclass(cls: type, data: Any, path: str, *, strict: bool) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ConfigValueError(path or "<root>", "mapping", data)
    hints = _hints(cls)
    known = [f.name for f in fields(cls)]
    if strict:
        for key in data:
            if key not in known:
                raise UnknownConfigKeyError(path or "<root>", str(key), known)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        field_path = f"{path}.{f.name}" if path else f.name
        kwargs[f.name] = _coerce(data[f.name], hints[f.name], field_path)
    return cls(**kwargs)


def _to_plain(obj: Any) -> Any:
    """递归转成可 JSON 序列化的结构。"""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_plain(item) for item in obj]
    return obj


def _flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(data, Mapping):
        for key, value in data.items():
            flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    else:
        flat[prefix] = data
    return flat


# --------------------------------------------------------------------------- #
# 加载入口
# --------------------------------------------------------------------------- #
def load_app_config(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    dotenv: Sequence[str | Path] = (),
    strict: bool = True,
    root: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    ignore_user_settings: bool = False,
) -> AppConfig:
    """加载并校验配置。

    :param config_path: YAML 路径；``None`` 表示按项目根自动探测。
    :param env: 环境变量映射，默认取 ``os.environ``（不修改进程环境）。
    :param dotenv: 额外读取的 .env 文件序列，已存在的变量不覆盖。
    :param strict: 是否把未声明的键视为错误。
    :param root: 显式指定项目根；``None`` 时按 ``app.root`` 或自动探测。
    :param overrides: 最高优先级的字典覆盖（测试/CLI 用），路径用 ``a.b.c``。
    :param ignore_user_settings: 跳过 ``data/user_settings.json``（界面设置）。
        测试"YAML 与内置默认值是否一致"时必须打开 —— 那个文件装的是用户运行时
        偏好，不属于配置骨架。
    """
    environ: dict[str, str] = dict(os.environ if env is None else env)

    detected_root = detect_project_root(root)
    for candidate in dotenv:
        dotenv_path = Path(candidate)
        if not dotenv_path.is_absolute():
            dotenv_path = detected_root / dotenv_path
        injected = _load_dotenv(dotenv_path, environ)
        if injected:
            logger.debug("从 %s 注入 %d 个环境变量", dotenv_path, injected)

    resolved = Path(config_path) if config_path is not None else detected_root / _CONFIG_REL
    if not resolved.is_absolute():
        resolved = (detected_root / resolved).resolve()

    raw = _load_yaml(resolved) if resolved.is_file() else {}
    if not resolved.is_file():
        logger.warning("配置文件 %s 不存在，全部使用内置默认值", resolved)

    # 界面里保存的设置放在 data/user_settings.json：比 YAML 高、比环境变量低。
    # 这样"用户在网页上填的 API 地址"能压过仓库里的默认值，但运维仍可用环境
    # 变量临时顶掉 —— 也避免把密钥写进受版本控制的 app_config.yaml。
    # 顺序很重要：先写界面设置，再让环境变量压过去。反过来就让"临时顶掉"失效了。
    settings_path = detected_root / "data" / "user_settings.json"
    settings = {} if ignore_user_settings else _load_user_settings(settings_path)
    for dotted, value in settings.items():
        _deep_set(raw, tuple(part for part in dotted.split(".") if part), value)
    if settings:
        logger.debug("应用 %d 条界面设置覆盖（%s）", len(settings), settings_path)

    env_overrides = _collect_env_overrides(environ)
    for path_parts, value in env_overrides.items():
        _deep_set(raw, path_parts, value)
    if env_overrides:
        logger.debug("应用 %d 条环境变量覆盖", len(env_overrides))

    if overrides:
        for dotted, value in overrides.items():
            _deep_set(raw, tuple(part for part in dotted.split(".") if part), value)

    # 严格模式先对顶层做一次检查，报错信息里带上文件路径
    config = _build_dataclass(AppConfig, raw, "", strict=strict)

    raw_root = config.app.root
    if root is not None:
        root_dir = Path(os.path.expanduser(str(root))).resolve()
    elif raw_root and raw_root != ".":
        candidate = Path(os.path.expanduser(raw_root))
        root_dir = candidate if candidate.is_absolute() else (resolved.parent.parent / candidate).resolve()
    else:
        root_dir = detected_root

    config = dataclasses.replace(config, root_dir=root_dir, source_file=resolved.resolve())
    config.validate()
    logger.debug(
        "配置加载完成: %s (root=%s, %d 条 env 覆盖)",
        config.source_file,
        config.root_dir,
        len(env_overrides),
    )
    return config


_LOCK = threading.RLock()
_CACHE: dict[str, AppConfig] = {}


def get_app_config(*, force_reload: bool = False, **kwargs: Any) -> AppConfig:
    """进程级单例。首次调用按默认参数加载，之后直接复用。"""
    key = "default"
    with _LOCK:
        if force_reload or key not in _CACHE:
            _CACHE[key] = load_app_config(**kwargs)
        return _CACHE[key]


def reload_app_config(**kwargs: Any) -> AppConfig:
    """强制重新加载（配置文件改动后调用）。"""
    return get_app_config(force_reload=True, **kwargs)


def set_app_config(config: AppConfig) -> AppConfig:
    """注入外部构造的配置（测试用）。"""
    with _LOCK:
        _CACHE["default"] = config
        return config
