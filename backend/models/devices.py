"""网页设备偏好、实际生效设备和运行库能力。只读状态，不加载权重。"""

from backend.models._runtime import ensure_cuda_dlls, ensure_torch_path


def device_capabilities():
    result = {"torch_cuda": False, "asr_cuda": False, "gpu_name": "", "warnings": []}
    try:
        ensure_torch_path()
        ensure_cuda_dlls()
        import torch

        result["torch_cuda"] = bool(torch.cuda.is_available())
        if result["torch_cuda"]:
            result["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            result["warnings"].append(
                "BGE-M3、SigLIP2 使用 GPU 需要 CUDA 版 PyTorch 和可用的 NVIDIA 驱动。"
            )
    except Exception as exc:
        result["warnings"].append(f"PyTorch 设备检测失败：{type(exc).__name__}")
    try:
        import ctranslate2

        result["asr_cuda"] = ctranslate2.get_cuda_device_count() > 0
        if not result["asr_cuda"]:
            result["warnings"].append("Whisper 未检测到 CUDA 设备。")
    except Exception as exc:
        result["warnings"].append(f"CTranslate2 设备检测失败：{type(exc).__name__}")
    return result


def effective_settings(config, capabilities):
    def explicit(device):
        if device in {None, "auto"}:
            return "cuda" if capabilities["torch_cuda"] else "cpu"
        return device

    return {
        "asr.device": explicit(config.asr.device or config.models.device),
        "embedding.text_device": explicit(
            config.embedding.text_device or config.models.device
        ),
        "embedding.image_device": explicit(
            config.embedding.image_device or config.models.device
        ),
        "ui.report_mode": config.ui.report_mode,
        "ui.theme_mode": config.ui.theme_mode,
    }


def active_devices(orchestrator):
    def torch_device(tower):
        if tower is None:
            return "未加载"
        if hasattr(tower, "_model"):
            model = tower._model
            if model is None:
                return "未加载"
            # 读取权重所在设备，而非仅展示用户配置的目标设备。
            return str(model.device).split(":", 1)[0]
        return getattr(tower, "device", "未加载")

    embed = getattr(orchestrator, "_embedder", None)
    asr = getattr(orchestrator, "_asr", None)
    return {
        "faster-whisper-small": getattr(asr, "active_device", ("未加载", ""))[0]
        or "未加载",
        "bge-m3": torch_device(getattr(embed, "_text", None)),
        "siglip2-base": torch_device(getattr(embed, "_image", None)),
    }
