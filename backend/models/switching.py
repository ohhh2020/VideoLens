"""本地模型设备事务：等待当前推理、验证候选、保存设置后一次提交。"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from dataclasses import replace

from backend.models._runtime import gpu_slot
from backend.models.asr import AsrConfig, AsrEngine
from backend.models.devices import active_devices
from backend.models.embed import EmbeddingService, detect_device

logger = logging.getLogger("zhiying.models.switching")


class DeviceSwitchError(RuntimeError):
    """切换失败；旧模型及已保存配置均保留。"""


def _release(model):
    try:
        model.unload()
    except Exception:
        logger.warning("模型资源清理未完成，将由对象生命周期继续回收", exc_info=True)


def apply_local_devices(orch, config, persist):
    """调用方串行化设置请求；persist 失败也不替换运行中的模型。"""
    previous = getattr(orch, "_cfg", None)
    if previous is None:
        raise DeviceSwitchError("本地模型组件未就绪，无法切换设备")
    asr, embed = orch._asr, orch._embedder
    asr_config = AsrConfig.from_config(config)
    current = active_devices(orch)
    changed = []
    if asr_config != asr.config or not asr.is_loaded:
        changed.append("faster-whisper-small")
    if current["bge-m3"] != detect_device(config.embedding.text_device or config.models.device):
        changed.append("bge-m3")
    if current["siglip2-base"] != detect_device(config.embedding.image_device or config.models.device):
        changed.append("siglip2-base")
    runtime_config = replace(previous, asr=config.asr, embedding=config.embedding, ui=config.ui,
        integrity=config.integrity, media=config.media, limits=config.limits, report=config.report,
        chunk=config.chunk, retrieval=config.retrieval, vl=replace(previous.vl, max_image_side=config.vl.max_image_side))
    if not changed:
        persist()
        orch._cfg = runtime_config
        return []

    logger.info("本地模型设备切换等待当前推理结束：%s", ", ".join(changed))
    with ExitStack() as stack:
        # 顺序固定；推理入口也都先持有对应模型锁，再获取 GPU 锁。
        for lock in (asr._transcribe_lock, embed._text_lock, embed._image_lock):
            stack.enter_context(lock)
        stack.enter_context(gpu_slot("cuda"))
        manager = asr._manager()
        candidate_embed = candidate_asr = None
        prepared = []
        try:
            if "bge-m3" in changed or "siglip2-base" in changed:
                candidate_embed = EmbeddingService.from_config(config, manager=manager)
                if "bge-m3" not in changed:
                    candidate_embed._text = embed._text
                if "siglip2-base" not in changed:
                    candidate_embed._image = embed._image
                if "bge-m3" in changed:
                    prepared.append(candidate_embed._text)
                    candidate_embed.embed_query("本地模型设备切换验证")
                if "siglip2-base" in changed:
                    from PIL import Image

                    prepared.append(candidate_embed._image)
                    candidate_embed.embed_image(Image.new("RGB", (224, 224), "red"))
                    candidate_embed.embed_cross_modal("一张红色图片")
            if "faster-whisper-small" in changed:
                candidate_asr = AsrEngine(asr_config, model_manager=manager)
                candidate_asr.load()
            persist()
        except Exception as exc:
            for model in prepared:
                _release(model)
            if candidate_asr is not None:
                _release(candidate_asr)
            logger.warning("本地模型切换失败，保留原设备和设置：%s", type(exc).__name__)
            raise DeviceSwitchError(f"设备切换失败，原设置和模型已保留：{exc}") from exc

        retired = []
        if "bge-m3" in changed:
            retired.append(embed._text)
            embed._text = candidate_embed._text
        if "siglip2-base" in changed:
            retired.append(embed._image)
            embed._image = candidate_embed._image
        if candidate_asr is not None:
            # 保持服务对象身份不变，等待锁的请求读取提交后的新模型。
            retired_asr = asr._model
            asr._model, asr._loaded_key = candidate_asr._model, candidate_asr._loaded_key
            asr.config = candidate_asr.config
            candidate_asr._model = None
            from backend.models.asr_worker import AsrWorker
            if isinstance(retired_asr, AsrWorker):
                retired_asr.close()
            del retired_asr
        orch._cfg = runtime_config
        for model in retired:
            _release(model)
        logger.info("本地模型设备切换完成，实际设备=%s", active_devices(orch))
    return changed
