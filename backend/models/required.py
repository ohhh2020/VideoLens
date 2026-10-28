"""强制本地模型清单与启动前磁盘校验。"""
REQUIRED_MODELS = ("bge-m3", "faster-whisper-small", "siglip2-base")

def verify_required_models(manager):
    errors = []
    for name in REQUIRED_MODELS:
        state = manager.status(name)
        if not state.available:
            errors.append(f"{name}: {state.state.value}")
    if errors:
        raise RuntimeError("必需模型缺失或不完整，禁止启动：" + "；".join(errors) + "。请在 Conda 环境运行 scripts/download_models.py")
