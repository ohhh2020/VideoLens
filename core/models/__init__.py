"""模型推理子包（权重供应链 + 五个推理适配器）。

分层
----
::

    manager.py   权重供应链：定位 / 下载 / 校验 / manifest / 清理
        ↑        业务代码只调 ensure_model(name) -> Path，不拼路径
    ├── asr.py       faster-whisper 语音转写（含词级时间戳）
    ├── vl.py        Qwen2-VL 视觉理解（GGUF + mmproj）
    ├── embed.py     BGE-M3 文本向量 / SigLIP2 图文跨模态向量
    └── llm.py       云端 OpenAI 协议 / 本地 GGUF / jieba 关键词兜底

约定
----
* 所有适配器都接受"模型目录"而不是权重文件名 —— 文件名由适配器内部挑
  （``_pick_weights`` / ``_find_mmproj``），因为量化档位会随下载源变化。
* 所有适配器的 ``from_config(cfg, manager)`` 在权重缺失/依赖缺失时
  **返回降级实例**，不抛异常。降级状态可查 ``.degraded`` / ``.available``，
  并在结果对象里带 ``degraded=True`` 与 ``error``，供报告统计"降级项"。
* 推理器一律懒加载 + 可 ``close()``，配合 ``models.max_loaded_models``
  做显存淘汰。
"""

from __future__ import annotations

from core.models.asr import (
    AsrConfig,
    AsrEngine,
    AsrError,
    Transcript,
    TranscriptSegment,
    Word,
)
from core.models.embed import (
    DEFAULT_IMAGE_DIM,
    DEFAULT_TEXT_DIM,
    BgeM3TextEmbedder,
    EmbeddingError,
    EmbeddingService,
    HashingEmbedder,
    ImageEmbedder,
    SigLip2ImageEmbedder,
    TextEmbedder,
    detect_device,
)
from core.models.llm import (
    VALID_ROLES,
    KeywordBackend,
    LlamaCppChatBackend,
    LlmBackend,
    LlmError,
    LlmGateway,
    LlmMessage,
    LlmResponse,
    OpenAICompatibleBackend,
    detect_local_server,
    normalize_messages,
)
from core.models.manager import (
    PROJECT_ROOT,
    CacheConfig,
    ConfigError,
    DownloadReport,
    MissingDependencyError,
    ModelDownloadError,
    ModelIntegrityError,
    ModelManager,
    ModelManagerConfig,
    ModelManagerError,
    ModelSpec,
    ModelState,
    ModelStatus,
    NetworkConfig,
    SourceSpec,
    UnknownModelError,
    configure_logging,
    get_model_manager,
    reset_model_manager,
)
from core.models.vl import (
    DEFAULT_VL_PROMPT_KEY,
    NullVisionModel,
    QwenVlRunner,
    VlError,
    VlResult,
    VlService,
    split_description,
)

__all__ = [
    # 权重供应链
    "PROJECT_ROOT",
    "CacheConfig",
    "ConfigError",
    "DownloadReport",
    "MissingDependencyError",
    "ModelDownloadError",
    "ModelIntegrityError",
    "ModelManager",
    "ModelManagerConfig",
    "ModelManagerError",
    "ModelSpec",
    "ModelState",
    "ModelStatus",
    "NetworkConfig",
    "SourceSpec",
    "UnknownModelError",
    "configure_logging",
    "get_model_manager",
    "reset_model_manager",
    # ASR
    "AsrConfig",
    "AsrEngine",
    "AsrError",
    "Transcript",
    "TranscriptSegment",
    "Word",
    # 向量
    "DEFAULT_IMAGE_DIM",
    "DEFAULT_TEXT_DIM",
    "BgeM3TextEmbedder",
    "EmbeddingError",
    "EmbeddingService",
    "HashingEmbedder",
    "ImageEmbedder",
    "SigLip2ImageEmbedder",
    "TextEmbedder",
    "detect_device",
    # 视觉
    "DEFAULT_VL_PROMPT_KEY",
    "NullVisionModel",
    "QwenVlRunner",
    "VlError",
    "VlResult",
    "VlService",
    "split_description",
    # LLM
    "VALID_ROLES",
    "KeywordBackend",
    "LlamaCppChatBackend",
    "LlmBackend",
    "LlmError",
    "LlmGateway",
    "LlmMessage",
    "LlmResponse",
    "OpenAICompatibleBackend",
    "detect_local_server",
    "normalize_messages",
]
