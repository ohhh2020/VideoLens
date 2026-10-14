"""使用实际权重做最小推理；不联网下载，不用 Mock，逐项记录失败或未执行。

示例：conda run -n VideoLens python scripts/verify_real_models.py --audio data/fixtures/演讲.wav
添加 --cuda 可在完成 CPU 基线之后验证同一个模型的 CUDA 路径。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path
from collections.abc import Callable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from videolens.bootstrap import read_json, start_local, stop_local
from videolens.config import atomic_json, load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="真实模型 CPU/CUDA 推理验证")
    parser.add_argument("--audio", type=Path, help="真实人声 WAV/音频文件；缺失时 ASR 明确标记未执行")
    parser.add_argument("--cuda", action="store_true", help="分别尝试 CUDA 推理并记录显存峰值")
    parser.add_argument("--skip-local", action="store_true", help="不执行 llama.cpp 生成")
    parser.add_argument("--output", type=Path, default=Path("data/verification/real-models.json"))
    args = parser.parse_args()
    settings = load_settings()
    manifest = read_json(Path(settings.models_dir) / "model-manifest.json")
    report = {"kind": "real_model_inference", "platform": platform.platform(), "python": platform.python_version(),
              "cases": [], "note": "合成图像用于推理冒烟；不代表真实视频检索准确率。"}

    def run_case(name: str, function: Callable[[], dict]) -> None:
        """每项测试立即落盘，后续模型失败也不会丢掉前面已经取得的真实证据。"""
        start = time.perf_counter()
        case = {"name": name}
        try:
            case.update(function())
            case.setdefault("status", "passed")
        except Exception as exc:
            case.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        case["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        report["cases"].append(case)
        atomic_json(args.output, report)
        print(json.dumps(case, ensure_ascii=False), flush=True)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def model_path(role: str) -> str:
        """只接受部署清单里的固定快照，避免验证时悄悄下载不同权重。"""
        entry = manifest.get("models", {}).get(role)
        if not entry:
            raise RuntimeError(f"模型 {role} 没有安装清单")
        return entry["snapshot"]

    for device in ["cpu"] + (["cuda"] if args.cuda else []):
        def asr(device: str = device) -> dict:
            if not args.audio:
                return {"status": "not_run", "reason": "未提供真实语音；静音推理不能证明语音识别正确。"}
            import ctranslate2
            from videolens.models import ModelGateway
            selected = settings.model_copy(update={"device": device, "allow_model_download": False})
            gateway = ModelGateway(selected, lambda ref: None)
            try:
                transcript = gateway.transcribe(args.audio)
                actual_device = gateway._device
            finally:
                gateway.release()
            if not transcript or not any(s["text"].strip() for s in transcript):
                raise AssertionError("真实人声未输出任何转录")
            return {"segments": transcript, "device_requested": device, "device_actual": actual_device,
                    "status": "passed" if actual_device == device else "fallback",
                    "ctranslate2_version": ctranslate2.__version__}

        def text_embedding(device: str = device) -> dict:
            import numpy as np
            import torch
            from sentence_transformers import SentenceTransformer
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            model = SentenceTransformer(model_path("text"), device=device, local_files_only=True)
            model.max_seq_length = 256
            vectors = model.encode(["视频中老师介绍了牛顿第二定律", "牛顿第二定律的教学说明", "烹饪番茄炒鸡蛋"],
                                   normalize_embeddings=True)
            relevant, unrelated = float(vectors[0] @ vectors[1]), float(vectors[0] @ vectors[2])
            assert np.isfinite(vectors).all() and relevant > unrelated
            result = {"device": device, "shape": list(vectors.shape), "relevant_similarity": relevant,
                      "unrelated_similarity": unrelated, "torch_version": torch.__version__}
            if device == "cuda":
                result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                result["gpu"] = torch.cuda.get_device_name()
            return result

        def image_embedding(device: str = device) -> dict:
            import torch
            from PIL import Image, ImageDraw
            from transformers import AutoModel, AutoProcessor
            model = AutoModel.from_pretrained(model_path("image"), local_files_only=True).to(device).eval()
            processor = AutoProcessor.from_pretrained(model_path("image"), local_files_only=True)
            image = Image.new("RGB", (224, 224), "white")
            ImageDraw.Draw(image).rectangle((30, 30, 194, 194), fill="red")
            inputs = processor(text=["a red square on white background", "a blue ocean"], images=[image],
                               padding="max_length", return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = model(**inputs)
            values = output.logits_per_image.detach().cpu().tolist()
            assert torch.isfinite(output.logits_per_image).all()
            return {"device": device, "image_text_logits": values, "image_shape": list(output.image_embeds.shape),
                    "text_shape": list(output.text_embeds.shape), "comparison": "红色方块/蓝色海洋"}

        run_case(f"asr_{device}", asr)
        run_case(f"bge_m3_{device}", text_embedding)
        run_case(f"siglip2_{device}", image_embedding)

    if not args.skip_local:
        def local_generation() -> dict:
            import httpx
            owner = "verify-real-models"
            state = start_local(settings, owner=owner, wait_seconds=240)
            try:
                with httpx.Client(timeout=240, trust_env=False) as client:
                    response = client.post(settings.local_base_url.rstrip("/") + "/chat/completions", json={
                        "model": settings.local_model, "messages": [{"role": "user", "content": "请只回答中文词语：测试成功"}],
                        "max_tokens": 48, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
                    response.raise_for_status()
                    result = response.json()
                    answer = result["choices"][0]["message"].get("content", "")
                    assert answer.strip(), "llama.cpp 未产生可见回答"
                    return {"answer": answer, "usage": result.get("usage"), "gpu_layers": settings.llama_gpu_layers,
                            "note": "启动健康检查及真实 completion；正确率需另行评估。"}
            finally:
                if state.get("started"):
                    stop_local(settings, owner=owner)
        run_case("llama_cpp", local_generation)
    atomic_json(args.output, report)
    print(f"真实模型报告：{args.output.resolve()}")
    return 1 if any(case["status"] == "failed" for case in report["cases"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
