"""模型适配层：重型依赖按需导入，模型缺失必须明确报告，禁止伪造推理结果。

模型缓存统一放在 models_dir。环境配置工具写入的清单记录固定版本，本模块优先
读取该版本的 snapshot；关闭下载时，模型加载器只能读取本地文件。
"""
from __future__ import annotations

import base64
import functools
import gc
import importlib.util
import json
import os
import queue
import re
import threading
import wave
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

Check = Callable[[], None]


class ModelUnavailable(RuntimeError):
    """缺少模型、依赖或可用服务；业务层可据此进入明确标注的降级流程。"""


class _ControlSignal(BaseException):
    """取消/暂停来自业务层，不能被普通模型故障的 CPU 降级逻辑吞掉。"""
    def __init__(self, original: Exception):
        self.original = original


def _checkpoint(check: Check) -> None:
    try:
        check()
    except Exception as exc:
        raise _ControlSignal(exc) from exc


def _cancellable(function):
    """将内部取消信号还原为调用方原本的异常类型。"""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except _ControlSignal as exc:
            raise exc.original
    return wrapped


class ModelGateway:
    """一个网关只在内存中保留一个重型模型，防止 8GB 显存同时装入多套权重。"""

    def __init__(self, settings: Any, secret_getter: Callable[[str], str | None]):
        self.settings = settings
        self.secret_getter = secret_getter
        self._lock = threading.RLock()
        self._model = None
        self._processor = None
        self._kind: str | None = None
        self._device = "cpu"

    def _source(self, role: str, repo_id: str) -> str:
        """固定清单存在时使用其快照路径，避免无意切换到缓存中的另一个版本。"""
        manifest = Path(self.settings.models_dir) / "model-manifest.json"
        if manifest.is_file():
            try:
                item = json.loads(manifest.read_text(encoding="utf-8"))["models"].get(role, {})
                if item.get("repo_id") == repo_id and item.get("snapshot") and Path(item["snapshot"]).is_dir():
                    return item["snapshot"]
            except (ValueError, KeyError, TypeError, OSError):
                pass
        return repo_id

    def _preferred_device(self) -> str:
        if self.settings.device == "cpu":
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                # 能枚举 GPU 不等于二进制兼容这块显卡，因此先执行一次真实张量运算。
                value = (torch.ones(1, device="cuda") + 1).item()
                if value == 2:
                    return "cuda"
        except Exception:
            pass
        return "cpu"

    def release(self) -> None:
        """释放阶段模型；仅在 PyTorch 已加载时清理其显存缓存。"""
        with self._lock:
            self._model = None
            self._processor = None
            self._kind = None
            gc.collect()
            import sys
            if "torch" in sys.modules:
                try:
                    sys.modules["torch"].cuda.empty_cache()
                except Exception:
                    pass

    def _load(self, role: str, device: str) -> None:
        if self._kind == role and self._device == device and self._model is not None:
            return
        self.release()
        # 模型运行没有发送使用情况遥测的必要；此开关与允许下载权重独立。
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        directory = str(Path(self.settings.models_dir).resolve())
        offline = not self.settings.allow_model_download
        if role in {"text", "image"}:
            from transformers.utils import logging as transformers_logging
            transformers_logging.disable_default_handler()
            transformers_logging.enable_propagation()
        if role == "asr":
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self._source("asr", self.settings.asr_model), device=device,
                                       compute_type="float16" if device == "cuda" else "int8",
                                       download_root=directory, local_files_only=offline)
        elif role == "text":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._source("text", self.settings.text_embedding_model),
                                               cache_folder=directory, device=device,
                                               local_files_only=offline, trust_remote_code=False)
        elif role == "image":
            from transformers import AutoModel, AutoProcessor
            source = self._source("image", self.settings.image_embedding_model)
            self._processor = AutoProcessor.from_pretrained(source, cache_dir=directory, local_files_only=offline, trust_remote_code=False)
            self._model = AutoModel.from_pretrained(source, cache_dir=directory, local_files_only=offline,
                                                    trust_remote_code=False).to(device).eval()
        self._kind, self._device = role, device

    def _devices(self) -> list[str]:
        preferred = self._preferred_device()
        return ["cuda", "cpu"] if preferred == "cuda" else ["cpu"]

    def _asr_devices(self) -> list[str]:
        """CTranslate2 自带独立的 CUDA 后端，不能用 PyTorch CPU 版本否定它。

        能枚举显卡只决定是否尝试；成功仍以实际转写为准，失败会执行 CPU 重试。
        """
        if self.settings.device == "cpu":
            return ["cpu"]
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                return ["cuda", "cpu"]
        except Exception:
            pass
        return ["cpu"]

    @staticmethod
    def _decode_audio(audio: Path, check: Check):
        """直接读取规范 PCM，兼容 PyAV 19 删除 metadata_errors 参数后的版本。

        FFmpeg 阶段已输出 16kHz 单声道 PCM；将它转换为模型接受的 float32 数组，
        可避免 faster-whisper 再做一遍相同解码。其他格式使用当前 PyAV 公共接口。
        """
        import numpy as np
        _checkpoint(check)
        try:
            with wave.open(str(audio), "rb") as stream:
                if stream.getframerate() == 16000 and stream.getsampwidth() == 2 and stream.getnchannels() == 1:
                    chunks = []
                    while data := stream.readframes(16000 * 30):
                        _checkpoint(check)
                        chunks.append(np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0)
                    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
        except (wave.Error, EOFError):
            pass
        import av
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        chunks = []
        with av.open(str(audio), mode="r") as container:
            for frame in container.decode(audio=0):
                _checkpoint(check)
                for converted in resampler.resample(frame):
                    chunks.append(converted.to_ndarray().reshape(-1))
            for converted in resampler.resample(None):
                chunks.append(converted.to_ndarray().reshape(-1))
        return np.concatenate(chunks).astype(np.float32) if chunks else np.empty(0, dtype=np.float32)

    @_cancellable
    def transcribe(self, audio: Path, check: Check = lambda: None) -> list[dict]:
        """真实 ASR 转写；GPU 推理失败后从头在 CPU 重试，避免混合两份输出。"""
        with self._lock:
            _checkpoint(check)
            try:
                samples = self._decode_audio(Path(audio), check)
            except Exception as exc:
                raise ModelUnavailable("音频解码失败，请检查音频文件及 PyAV 依赖。") from exc
            for device in self._asr_devices():
                _checkpoint(check)
                try:
                    self._load("asr", device)
                    generator, _ = self._model.transcribe(samples, beam_size=5, vad_filter=True)
                    results = []
                    for segment in generator:
                        _checkpoint(check)
                        text = segment.text.strip()
                        if text and segment.end > segment.start:
                            results.append({"id": f"s{len(results) + 1:06d}", "start_ms": max(0, round(segment.start * 1000)),
                                            "end_ms": round(segment.end * 1000), "text": text})
                    _checkpoint(check)
                    return results
                except Exception:
                    self.release()
            raise ModelUnavailable("语音转写模型不可用。请下载 faster-whisper 模型并运行模型诊断；也可提供同名字幕。")

    @_cancellable
    def embed_texts(self, texts: list[str], check: Check = lambda: None) -> list[list[float]]:
        """BGE-M3 向量归一化，使余弦相似度的含义在建库与查询时一致。"""
        if not texts:
            return []
        with self._lock:
            _checkpoint(check)
            for device in self._devices():
                _checkpoint(check)
                try:
                    self._load("text", device)
                    output = []
                    for start in range(0, len(texts), 8):
                        _checkpoint(check)
                        values = self._model.encode(texts[start:start + 8], batch_size=8, normalize_embeddings=True, show_progress_bar=False)
                        output.extend(values.tolist())
                    _checkpoint(check)
                    return output
                except Exception:
                    self.release()
            raise ModelUnavailable("BGE-M3 文本向量模型不可用，已允许业务层使用关键词检索。请先完成模型下载。")

    def _image_features(self, values: list[str], is_text: bool, check: Check) -> list[list[float]]:
        with self._lock:
            _checkpoint(check)
            for device in self._devices():
                _checkpoint(check)
                try:
                    import torch
                    from PIL import Image
                    self._load("image", device)
                    output = []
                    for start in range(0, len(values), 8):
                        _checkpoint(check)
                        batch = values[start:start + 8]
                        if is_text:
                            inputs = self._processor(text=batch, padding="max_length", truncation=True, return_tensors="pt")
                        else:
                            images = []
                            for path in batch:
                                with Image.open(path) as image:
                                    images.append(image.convert("RGB"))
                            inputs = self._processor(images=images, return_tensors="pt")
                        inputs = {key: value.to(device) for key, value in inputs.items()}
                        with torch.inference_mode():
                            feature = self._model.get_text_features(**inputs) if is_text else self._model.get_image_features(**inputs)
                            # 部分 transformers 版本返回带 pooler_output 的结构，兼容两种公开返回值。
                            if not isinstance(feature, torch.Tensor):
                                feature = feature.pooler_output
                            feature = torch.nn.functional.normalize(feature.float(), p=2, dim=-1)
                            output.extend(feature.cpu().tolist())
                    _checkpoint(check)
                    return output
                except Exception:
                    self.release()
            raise ModelUnavailable("SigLIP2 图文向量模型不可用，请检查模型缓存、Pillow 和 transformers 依赖。")

    @_cancellable
    def embed_images(self, paths: list[str], check: Check = lambda: None) -> list[list[float]]:
        return self._image_features(paths, False, check) if paths else []

    @_cancellable
    def embed_image_query(self, text: str, check: Check = lambda: None) -> list[float]:
        return self._image_features([text], True, check)[0]

    def _provider(self, role: str) -> tuple[str, str, str | None, float]:
        provider = getattr(self.settings, f"{role}_provider")
        if not self.settings.cloud_enabled or not provider.enabled or not provider.base_url or not provider.model:
            raise ModelUnavailable("云端模型未启用或尚未配置。")
        key = self.secret_getter(provider.key_ref) if provider.key_ref else None
        return provider.base_url, provider.model, key, provider.timeout_seconds

    def _local_provider(self) -> tuple[str, str, None, float]:
        # 离线模式不允许把“本地服务”配置成远程主机来绕过云端开关。
        parsed = urlparse(self.settings.local_base_url)
        if not self.settings.local_enabled or parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ModelUnavailable("本地 llama.cpp 服务未启用或地址不是本机回环地址。")
        return self.settings.local_base_url, self.settings.local_model, None, 120

    def _http_stream(self, provider: tuple, messages: list[dict], check: Check) -> Iterator[str]:
        """独立读取线程等待网络，主线程每 200ms 响应取消；不把请求正文写入日志。"""
        import httpx
        base_url, model, key, timeout = provider
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        events: queue.Queue = queue.Queue()
        stopped = threading.Event()
        active: list[Any] = []
        payload = {"model": model, "messages": messages, "stream": True, "temperature": 0.2,
                   "max_tokens": getattr(self.settings, "output_tokens", 1536)}
        host = urlparse(base_url).hostname or ""
        # 新一代 OpenAI 模型使用 completion 预算，推理模式不接受 temperature。
        # 仅针对官方主机调整，避免强行改变其他兼容平台的参数语义。
        if host == "api.openai.com" and model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
            payload["max_completion_tokens"] = payload.pop("max_tokens")
            payload.pop("temperature")
            payload["reasoning_effort"] = "none" if model.startswith(("gpt-6-luna", "gpt-6-sol")) else "low"
        elif host.endswith("aliyuncs.com") and model.startswith("qwen"):
            payload["enable_thinking"] = False
        elif host == "api.deepseek.com":
            payload["thinking"] = {"type": "disabled"}
        if key is None and urlparse(base_url).hostname in {"localhost", "127.0.0.1", "::1"}:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        def reader() -> None:
            try:
                with httpx.Client(timeout=httpx.Timeout(float(timeout), connect=10), trust_env=False) as client:
                    with client.stream("POST", base_url.rstrip("/") + "/chat/completions", headers=headers, json=payload) as response:
                        active.append(response)
                        response.raise_for_status()
                        completed = False
                        for line in response.iter_lines():
                            if stopped.is_set():
                                return
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                completed = True
                                break
                            try:
                                item = json.loads(data)
                            except json.JSONDecodeError as exc:
                                raise ModelUnavailable("模型 SSE 数据格式无效。") from exc
                            if item.get("error"):
                                raise ModelUnavailable("模型服务返回流式错误。")
                            choices = item.get("choices", [])
                            if choices:
                                token = choices[0].get("delta", {}).get("content")
                                if token:
                                    events.put(("token", str(token)))
                                reason = choices[0].get("finish_reason")
                                if reason in {"stop", "eos_token"}:
                                    completed = True
                                elif reason in {"length", "content_filter"}:
                                    raise ModelUnavailable("模型回答被截断或过滤，未产生完整结果。")
                        if not completed:
                            raise ModelUnavailable("模型连接提前关闭，未收到正常结束标记。")
                events.put(("done", None))
            except Exception:
                events.put(("error", ModelUnavailable("模型服务请求失败。请检查连接、模型名、密钥、限额和超时设置。")))

        thread = threading.Thread(target=reader, name="videolens-model-stream", daemon=True)
        _checkpoint(check)
        thread.start()
        try:
            while True:
                _checkpoint(check)
                try:
                    kind, value = events.get(timeout=0.2)
                except queue.Empty:
                    continue
                if kind == "token":
                    yield value
                elif kind == "done":
                    return
                else:
                    raise value
        finally:
            stopped.set()
            for response in active:
                try:
                    response.close()
                except Exception:
                    pass

    def _http_generate(self, provider: tuple, messages: list[dict], check: Check) -> str:
        answer = "".join(self._http_stream(provider, messages, check)).strip()
        if not answer:
            raise ModelUnavailable("模型返回了空回答。")
        return re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip()

    @_cancellable
    def describe(self, frames: list[dict], check: Check = lambda: None) -> list[dict]:
        """逐帧调用视觉模型，输出分别绑定原始帧 ID 的画面描述和 OCR。"""
        provider = self._provider("vision")
        if not frames:
            return []
        self.release()
        results = []
        successes = 0
        for frame in frames:
            _checkpoint(check)
            content = base64.b64encode(Path(frame["path"]).read_bytes()).decode("ascii")
            messages = [{"role": "system", "content": "你是视频证据分析助手。只描述这张图片里能直接看到的内容，不猜测声音或上下文。返回 JSON 对象，且只有 description（中文画面描述）和 ocr（逐字抄录可读文字）两个字符串字段。"},
                        {"role": "user", "content": [{"type": "text", "text": f"分析帧 {frame['id']}，看不清的文字不要补造。"},
                                                       {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{content}"}}]}]
            result = dict(frame)
            try:
                answer = self._http_generate(provider, messages, check)
                answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip())
                parsed = json.loads(answer)
                if not isinstance(parsed, dict) or not isinstance(parsed.get("description", ""), str) or not isinstance(parsed.get("ocr", ""), str):
                    raise ValueError("视觉模型结构无效。")
                result.update(description=parsed.get("description", ""), ocr=parsed.get("ocr", ""))
                successes += 1
            except Exception:
                result.update(description="", ocr="", vision_error="此帧视觉理解失败，保留原始图片供人工查阅。")
            results.append(result)
        if not successes:
            raise ModelUnavailable("全部关键帧的视觉理解失败，将保留图片并使用字幕/语音通道。")
        return results

    @_cancellable
    def generate(self, messages: list[dict], check: Check = lambda: None, stream: bool = False):
        """云端生成失败时尝试本地服务；两者失败由业务层生成摘录报告。"""
        if stream:
            return self.stream(messages, check)
        result = "".join(self.stream(messages, check)).strip()
        return re.sub(r"<think>.*?</think>", "", result, flags=re.S).strip()

    def stream(self, messages: list[dict], check: Check = lambda: None) -> Iterator[str]:
        """若已输出部分回答后连接断开，立即报错，不混入另一模型续写冒充成功。"""
        try:
            self.release()
            providers = []
            try:
                providers.append(self._provider("text"))
            except ModelUnavailable:
                pass
            try:
                providers.append(self._local_provider())
            except ModelUnavailable:
                pass
            for provider in providers:
                _checkpoint(check)
                emitted = False
                try:
                    for token in self._http_stream(provider, messages, check):
                        emitted = True
                        yield token
                    if emitted:
                        return
                except Exception:
                    if emitted:
                        raise ModelUnavailable("模型回答中途断开。本次回答不完整，请重试。")
            raise ModelUnavailable("云端文本模型和本地 llama.cpp 均不可用，当前只能生成基础摘录报告。")
        except _ControlSignal as exc:
            raise exc.original

    def health(self) -> dict:
        """只查看安装与缓存状态，不因打开设置页面而加载数 GB 模型。"""
        dependencies = {name: importlib.util.find_spec(name) is not None for name in ("faster_whisper", "sentence_transformers", "transformers", "torch", "httpx")}
        roles = {"asr": self.settings.asr_model, "text": self.settings.text_embedding_model, "image": self.settings.image_embedding_model}
        cache = {}
        for role, repo in roles.items():
            source = self._source(role, repo)
            snapshots = Path(self.settings.models_dir) / ("models--" + repo.replace("/", "--")) / "snapshots"
            cache[role] = Path(source).is_dir() if source != repo else snapshots.exists() and any(snapshots.iterdir())
        return {"dependencies": dependencies, "cached_models": cache, "loaded_model": self._kind,
                "device": self._device if self._kind else "未加载", "cloud_enabled": self.settings.cloud_enabled,
                "local_enabled": self.settings.local_enabled, "note": "缓存/依赖存在仅表示可尝试加载，不代表真实推理已通过。"}
