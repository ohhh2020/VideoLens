"""后端组件装配；必需模型缺失、损坏或推理失败时抛错。"""

from __future__ import annotations

import logging
from typing import Any

from backend.config import AppConfig, get_app_config


def _build_vectors(cfg: AppConfig) -> Any:
    from backend.rag.vector_store import VectorStore  # noqa: PLC0415

    return VectorStore.from_config(cfg)


def _build_embedder(cfg: AppConfig, manager: Any) -> Any:
    from backend.models.embed import EmbeddingService  # noqa: PLC0415

    # 注意 manager 是 keyword-only，位置传会 TypeError
    return EmbeddingService.from_config(cfg, manager=manager)


def _build_asr(cfg: AppConfig, manager: Any) -> Any:
    from backend.models.asr import AsrConfig, AsrEngine  # noqa: PLC0415

    return AsrEngine(AsrConfig.from_config(cfg), model_manager=manager)


def _build_scene(cfg: AppConfig) -> Any:
    from backend.segmentation.scenes import SceneDetectConfig, SceneDetector  # noqa: PLC0415

    # flattened() 里有 SceneDetectConfig 不认识的键（如 keyframe.format），过滤掉
    payload = cfg.media.scene_detect.flattened()
    known = set(SceneDetectConfig.__dataclass_fields__)
    return SceneDetector(
        SceneDetectConfig(**{k: v for k, v in payload.items() if k in known})
    )


def _build_chunker(cfg: AppConfig) -> Any:
    from backend.segmentation.chunker import ChunkConfig, Chunker  # noqa: PLC0415

    return Chunker(ChunkConfig(**cfg.chunk.as_chunk_config()))


def _build_vl(cfg: AppConfig, manager: Any) -> Any:
    from backend.models.vl import VlService  # noqa: PLC0415

    return VlService.from_config(cfg)


def _build_llm(cfg: AppConfig, manager: Any) -> Any:
    from backend.models.llm import LlmGateway  # noqa: PLC0415

    return LlmGateway.from_config(cfg)


def _build_context(cfg: AppConfig) -> Any:
    from backend.pipeline.context_builder import ContextBudget, ContextBuilder  # noqa: PLC0415
    from backend.prompts import get_prompt_registry  # noqa: PLC0415

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
    from backend.media.downloader import DownloadConfig, MediaDownloader  # noqa: PLC0415

    return MediaDownloader(DownloadConfig.from_config(cfg))


logger = logging.getLogger("zhiying.pipeline.factory")


def build_orchestrator(
    config=None, *, repository=None, on_progress=None, orchestrator_class=None
):
    from backend.models.manager import ModelManager
    from backend.models.required import verify_required_models
    from backend.storage.repository import SqliteRepository
    from backend.pipeline.orchestrator import Orchestrator

    cfg = config or get_app_config()
    manager = ModelManager(
        config_path=cfg.root_dir / cfg.models.config_file,
        cache_root=cfg.path("models"),
        allow_download=False,
    )
    verify_required_models(manager)
    import torch
    from transformers.utils import logging as transformers_logging

    torch.set_num_threads(cfg.models.cpu_threads)
    transformers_logging.disable_default_handler()
    transformers_logging.enable_propagation()
    transformers_logging.disable_progress_bar()
    embed = _build_embedder(cfg, manager)
    asr = _build_asr(cfg, manager)
    try:
        # 使用真实编码验证权重可加载，错误不允许转为哈希/空转写。
        from PIL import Image

        embed.embed_query("启动验证")
        embed.embed_cross_modal("一张红色图片")
        embed.embed_image(Image.new("RGB", (224, 224), "red"))
        asr.load()
        from backend.models.devices import active_devices
        from types import SimpleNamespace

        logger.info(
            "本地模型已验证，实际设备=%s Whisper精度=%s",
            active_devices(SimpleNamespace(_embedder=embed, _asr=asr)),
            asr.active_device[1],
        )
        # 默认保留真实预热后的权重，避免首次分析/提问重复加载。
        if not cfg.models.keep_loaded:
            embed.unload()
            asr.unload()
        # 初次安装允许进入网页填写云端配置；本地模型依旧必须真实加载成功。
        # 未配置通道只接受设置操作，分析/问答会明确拒绝，不生成替代内容。
        import os
        pending = []
        cloud = cfg.llm.cloud
        if cloud.base_url and cloud.model and cloud.resolved_key():
            llm = _build_llm(cfg, manager)
        else:
            from backend.models.llm import LlmGateway, OpenAICompatibleBackend
            pending.append("云端大模型")
            llm = LlmGateway([OpenAICompatibleBackend("", "", "")])
        vision_key = cfg.vl.cloud_api_key or os.environ.get(cfg.vl.cloud_api_key_env, "")
        if cfg.vl.cloud_base_url and cfg.vl.cloud_model and vision_key:
            vl = _build_vl(cfg, manager)
        else:
            from backend.models.vl import VlService
            pending.append("云端视觉理解")
            vl = VlService(None, model_name="cloud:待配置", degraded_reason="请在网页设置中配置云端通道")
        if pending:
            logger.info("%s 尚未配置，请打开网页设置填写并保存，然后重启；分析与问答暂不可用", "、".join(pending))
        repo = repository or SqliteRepository.from_config(cfg)
        vectors = _build_vectors(cfg)
        from backend.memory.store import MemoryStore
        from backend.memory.service import PreferenceMemory
        from backend.runtime import runtime_path

        memory = PreferenceMemory(
            MemoryStore(runtime_path(cfg.memory.file,cfg.root_dir),embed,cfg.memory),
            repo,llm,cfg.memory,
        )
        orch = (orchestrator_class or Orchestrator)(
            repository=repo,
            vectors=vectors,
            embedder=embed,
            asr=asr,
            scene_detector=None,
            chunker=_build_chunker(cfg),
            vl=vl,
            llm=llm,
            downloader=_build_downloader(cfg),
            context_builder=_build_context(cfg),
            config=cfg,
            on_progress=on_progress,
            memory=memory,
        )
        orch._cloud_setup_required = tuple(pending)
        orch._use_integrity = True
        return orch
    except Exception:
        embed.unload()
        asr.unload()
        raise
