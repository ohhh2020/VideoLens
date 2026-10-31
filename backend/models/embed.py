"""BGE-M3 与 SigLIP2 必需编码器，不提供哈希或缺失模型降级。"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from backend.config import get_app_config
from backend.models._runtime import gpu_slot

logger = logging.getLogger("zhiying.models.embed")

__all__ = [
    "DEFAULT_IMAGE_DIM",
    "DEFAULT_TEXT_DIM",
    "BgeM3TextEmbedder",
    "EmbeddingError",
    "EmbeddingService",
    "ImageEmbedder",
    "SigLip2ImageEmbedder",
    "TextEmbedder",
    "detect_device",
]

DEFAULT_TEXT_DIM: Final[int] = 1024
DEFAULT_IMAGE_DIM: Final[int] = 768


# =============================================================================
# 异常
# =============================================================================


class EmbeddingError(RuntimeError):
    """编码失败（模型加载、输入非法、推理报错）。"""


# =============================================================================
# 协议
# =============================================================================


@runtime_checkable
class TextEmbedder(Protocol):
    """文本编码器。返回 ``list[list[float]]``，与输入等长。"""

    name: str
    dim: int

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class ImageEmbedder(Protocol):
    """图像编码器。接受路径或 PIL 图像。"""

    name: str
    dim: int

    def embed_images(self, images: Sequence[str | Path | Any]) -> list[list[float]]: ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """SigLIP2 的文本侧编码（跨模态检索必需）。"""


# =============================================================================
# 工具
# =============================================================================


def detect_device(prefer: str = "auto") -> str:
    """探测可用设备，返回 ``"cuda"`` / ``"mps"`` / ``"cpu"``。

    ``prefer`` 为显式设备名时直接返回（不做可用性校验，交给调用方处理报错）。
    """
    want = (prefer or "auto").strip().lower()
    if want not in {"auto", ""}:
        return want
    from backend.models._runtime import ensure_torch_path  # noqa: PLC0415

    ensure_torch_path()
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 - 驱动异常不应阻断启动
        logger.debug("设备探测异常，回退 CPU", exc_info=True)
    return "cpu"


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm <= 1e-12:
        return vector
    return [x / norm for x in vector]


def _l2_normalize_matrix(matrix: list[list[float]]) -> list[list[float]]:
    try:
        import numpy as np  # noqa: PLC0415

        arr = np.asarray(matrix, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).astype("float32").tolist()
    except ImportError:
        return [_l2_normalize(row) for row in matrix]


def _feature_tensor(value: Any) -> Any:
    """把 transformers 的 ModelOutput 收敛成真正的特征张量。

    transformers 5.0 起 ``get_image_features`` / ``get_text_features`` 返回的是
    ``BaseModelOutputWithPooling`` 而不是 4.x 那样投影后的 tensor；直接塞进
    ``F.normalize`` 会炸在 ``'BaseModelOutputWithPooling' object has no attribute 'norm'``。
    同时 SigLIP 的投影头已经并进各塔的 ``head``，``pooler_output`` 就是最终嵌入。
    """
    if value is None:
        raise ValueError("模型返回空特征")
    pooled = getattr(value, "pooler_output", None)
    return pooled if pooled is not None else value


def _batch_ranges(total: int, size: int) -> list[tuple[int, int]]:
    step = max(1, int(size))
    return [(start, min(total, start + step)) for start in range(0, total, step)]


def _read_dim(embedder: Any, attr: str, default: int) -> int:
    """读取真实编码器或注入测试编码器的维度。"""
    for name in ("dim", attr):
        value = getattr(embedder, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return default


# =============================================================================
# BGE-M3
# =============================================================================


@dataclass
class BgeM3TextEmbedder:
    """BGE-M3 文本编码器（CLS 池化 + L2 归一化）。

    参数量约 568M，CPU 上单条 20 字查询约 60~120ms，可接受。
    批量编码按 ``batch_size`` 分批，避免长文本撑爆显存。
    """

    model_dir: Path
    device: str = "auto"
    batch_size: int = 16
    max_length: int = 8192
    name: str = "bge-m3"
    dim: int = DEFAULT_TEXT_DIM
    _model: Any = None
    _tokenizer: Any = None
    _lock: Any = None

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        if not self.model_dir.exists():
            raise EmbeddingError(f"BGE-M3 权重目录不存在: {self.model_dir}")
        self._lock = threading.Lock()
        self.device = detect_device(self.device)

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        with self._lock:
            if self._model is not None:
                return self._model, self._tokenizer
            try:
                from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
            except ImportError as exc:
                raise EmbeddingError(
                    "BGE-M3 编码需要 transformers：按 README.md 更新 Conda 环境"
                ) from exc
            logger.info("加载 BGE-M3: %s (device=%s)", self.model_dir, self.device)
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_dir), trust_remote_code=False, local_files_only=True
                )
                model = AutoModel.from_pretrained(
                    str(self.model_dir), trust_remote_code=False, local_files_only=True
                )
                model.eval()
                if self.device.startswith("cuda"):
                    import torch

                    model.to(device=self.device, dtype=torch.float16)
                else:
                    model.to(self.device)
            except Exception as exc:  # noqa: BLE001 - transformers 异常类型繁杂
                raise EmbeddingError(
                    f"加载 BGE-M3 失败({self.model_dir}): {exc}"
                ) from exc
            self._model, self._tokenizer = model, tokenizer
            return model, tokenizer

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        items = [str(t or "") for t in texts]
        if not items:
            return []
        model, tokenizer = self._load()
        import torch  # noqa: PLC0415

        out: list[list[float]] = []
        for start, end in _batch_ranges(len(items), self.batch_size):
            batch = items[start:end]
            try:
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max(8, int(self.max_length)),
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                with torch.no_grad():
                    outputs = model(**encoded)
                # BGE-M3：CLS 向量
                hidden = outputs.last_hidden_state[:, 0, :]
                hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
                out.extend(hidden.float().cpu().tolist())
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(
                    f"BGE-M3 编码失败（第 {start}-{end} 条）: {exc}"
                ) from exc
        return out

    def unload(self) -> None:
        """释放模型与显存（``models.max_loaded_models`` 淘汰时用）。"""
        with self._lock:
            self._model = None
            self._tokenizer = None
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("已卸载 BGE-M3")


# =============================================================================
# SigLIP2
# =============================================================================


@dataclass
class SigLip2ImageEmbedder:
    """SigLIP2 图文编码器（文本与图像投影到同一空间）。

    用 ``get_image_features`` / ``get_text_features`` 拿到**投影后**的向量，
    而不是池化前的 hidden state —— 跨模态检索必须用投影空间。

    按本地 config.json 选择 SiglipModel 或 Siglip2Model，兼容使用旧版模型格式
    发布的 SigLIP2 权重。配置序列化避开上游无效默认配置，实际权重不作修改。
    加载失败会抛出错误并阻止启动。
    """

    model_dir: Path
    device: str = "auto"
    batch_size: int = 16
    name: str = "siglip2-base"
    dim: int = DEFAULT_IMAGE_DIM
    _model: Any = None
    _processor: Any = None
    _lock: Any = None

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        if not self.model_dir.exists():
            raise EmbeddingError(f"SigLIP2 权重目录不存在: {self.model_dir}")
        self._lock = threading.Lock()
        self.device = detect_device(self.device)

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        with self._lock:
            if self._model is not None:
                return self._model, self._processor
            try:
                from backend.models.siglip_compat import load_siglip_config  # noqa: PLC0415
            except ImportError as exc:
                raise EmbeddingError(
                    "SigLIP2 编码需要 transformers：按 README.md 更新 Conda 环境"
                ) from exc
            logger.info("加载 SigLIP2: %s (device=%s)", self.model_dir, self.device)
            try:
                config, model_class, processor_class = load_siglip_config(self.model_dir)
                processor = processor_class.from_pretrained(
                    str(self.model_dir), config=config, local_files_only=True
                )
                model = model_class.from_pretrained(
                    str(self.model_dir), config=config, local_files_only=True
                )
                model.eval()
                if self.device.startswith("cuda"):
                    import torch

                    model.to(device=self.device, dtype=torch.float16)
                else:
                    model.to(self.device)
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(
                    f"加载 SigLIP2 失败({self.model_dir}): {exc}"
                ) from exc
            if not (
                hasattr(model, "get_image_features")
                and hasattr(model, "get_text_features")
            ):
                raise EmbeddingError(
                    f"{self.model_dir.name} 不是跨模态模型（缺少 get_image_features），"
                    f"请确认 models.yaml 里 siglip2 的 repo 配置"
                )
            self._model, self._processor = model, processor
            return model, processor

    def embed_images(self, images: Sequence[str | Path | Any]) -> list[list[float]]:
        if not images:
            return []
        model, processor = self._load()
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        out: list[list[float]] = []
        for start, end in _batch_ranges(len(images), self.batch_size):
            batch = images[start:end]
            opened: list[Any] = []
            try:
                for item in batch:
                    if isinstance(item, (str, Path)):
                        opened.append(Image.open(item).convert("RGB"))
                    else:
                        opened.append(
                            item.convert("RGB") if hasattr(item, "convert") else item
                        )
                inputs = processor(images=opened, return_tensors="pt")
                inputs = {
                    k: v.to(
                        device=self.device,
                        dtype=model.dtype if v.is_floating_point() else v.dtype,
                    )
                    for k, v in inputs.items()
                }
                with torch.no_grad():
                    feats = _feature_tensor(model.get_image_features(**inputs))
                feats = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
                out.extend(feats.float().cpu().tolist())
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(
                    f"SigLIP2 图像编码失败（第 {start}-{end} 张）: {exc}"
                ) from exc
            finally:
                for img in opened:
                    close = getattr(img, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:  # noqa: BLE001
                            pass
        return out

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        items = [str(t or "") for t in texts]
        if not items:
            return []
        model, processor = self._load()
        import torch  # noqa: PLC0415

        out: list[list[float]] = []
        for start, end in _batch_ranges(len(items), self.batch_size):
            batch = items[start:end]
            try:
                inputs = processor(
                    text=batch,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {
                    k: v.to(
                        device=self.device,
                        dtype=model.dtype if v.is_floating_point() else v.dtype,
                    )
                    for k, v in inputs.items()
                }
                with torch.no_grad():
                    feats = _feature_tensor(model.get_text_features(**inputs))
                feats = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
                out.extend(feats.float().cpu().tolist())
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(
                    f"SigLIP2 文本编码失败（第 {start}-{end} 条）: {exc}"
                ) from exc
        return out

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("已卸载 SigLIP2")


# =============================================================================
# 服务
# =============================================================================


class EmbeddingService:
    """统一的向量编码入口（供检索层与管道层调用）。

    Parameters
    ----------
    text_embedder / image_embedder:
        必须同时注入 BGE-M3 和 SigLIP2；缺失时抛错。
    normalize:
        是否对输出再做一次 L2 归一化（编码器内部通常已经做过，这里兜底）。
    max_text_chars:
        单条文本截断长度。BGE-M3 上限 8192 token，超长直接截字符更省事。
    """

    def __init__(
        self,
        text_embedder,
        image_embedder,
        *,
        batch_size=16,
        normalize=True,
        max_text_chars=8192,
    ):
        if text_embedder is None or image_embedder is None:
            raise EmbeddingError("必须同时提供 BGE-M3 和 SigLIP2 编码器")
        self._text, self._image = text_embedder, image_embedder
        self._batch_size = max(1, int(batch_size))
        self._normalize = bool(normalize)
        self._max_text_chars = max(1, int(max_text_chars))
        self._degraded_reason = ""
        from backend.models.batching import AdaptiveBatch
        self._text_batches = AdaptiveBatch()
        self._image_batches = AdaptiveBatch()
        self._cross_batches = AdaptiveBatch()
        self._text_lock = threading.Lock()
        self._image_lock = threading.Lock()

    # -- 构造 --

    @classmethod
    def from_config(cls, config=None, *, manager=None):
        cfg = config or get_app_config()
        if manager is None:
            raise EmbeddingError("必须提供 ModelManager；本地编码器不允许降级")
        e = cfg.embedding
        text = BgeM3TextEmbedder(
            manager.ensure_model(e.text_model),
            device=e.text_device or cfg.models.device,
            batch_size=cfg.integrity.gpu_batch_max
            if detect_device(e.text_device or cfg.models.device).startswith("cuda")
            else e.batch_size,
            max_length=e.max_text_chars,
            name=e.text_model,
            dim=e.text_dim,
        )
        image = SigLip2ImageEmbedder(
            manager.ensure_model(e.image_model),
            device=e.image_device or cfg.models.device,
            batch_size=cfg.integrity.gpu_batch_max
            if detect_device(e.image_device or cfg.models.device).startswith("cuda")
            else e.batch_size,
            name=e.image_model,
            dim=e.image_dim,
        )
        service = cls(
            text,
            image,
            batch_size=e.batch_size,
            normalize=e.normalize,
            max_text_chars=e.max_text_chars,
        )
        for controller in (service._text_batches, service._image_batches, service._cross_batches):
            controller.maximum = cfg.integrity.gpu_batch_max
        return service

    # -- 属性 --

    @property
    def text_dim(self) -> int:
        return _read_dim(self._text, "text_dim", DEFAULT_TEXT_DIM)

    @property
    def image_dim(self) -> int:
        return _read_dim(self._image, "image_dim", DEFAULT_IMAGE_DIM)

    @property
    def text_model_name(self) -> str:
        return str(getattr(self._text, "name", "unknown"))

    @property
    def image_model_name(self) -> str:
        return str(getattr(self._image, "name", "unknown"))

    @property
    def degraded(self) -> bool:
        return bool(self._degraded_reason)

    @property
    def degraded_reason(self) -> str:
        return self._degraded_reason

    def info(self) -> dict[str, Any]:
        return {
            "text_model": self.text_model_name,
            "text_dim": self.text_dim,
            "image_model": self.image_model_name,
            "image_dim": self.image_dim,
            "degraded": self.degraded,
            "degraded_reason": self._degraded_reason,
        }

    # -- 编码 --

    def embed_texts(self, texts: Sequence[str], *, max_batch=None) -> list[list[float]]:
        """批量文本编码。空输入返回 ``[]``，空字符串返回零向量。"""
        items = [self._truncate(t) for t in texts]
        if not items:
            return []
        with self._text_lock, gpu_slot(getattr(self._text, "device", "cpu")):
            if max_batch is not None:
                self._text_batches.maximum = max_batch
            vectors = self._text_batches.run(items, self._text.embed_texts, getattr(self._text, "device", "cpu"), lengths=[len(t) for t in items])
        if len(vectors) != len(items):
            raise EmbeddingError(
                f"文本编码器返回 {len(vectors)} 条向量，输入 {len(items)} 条"
            )
        self._check_dim(vectors, self.text_dim, "文本")
        return _l2_normalize_matrix(vectors) if self._normalize else vectors

    def embed_query(self, text: str) -> list[float]:
        """单条查询编码（BGE 系列查询不需要指令前缀，与文档同分布）。"""
        vectors = self.embed_texts([text])
        return vectors[0]

    def embed_images(self, images: Sequence[str | Path | Any], *, max_batch=None) -> list[list[float]]:
        """批量图像编码（路径或 PIL 对象）。"""
        items = list(images)
        if not items:
            return []
        with self._image_lock, gpu_slot(getattr(self._image, "device", "cpu")):
            if max_batch is not None:
                self._image_batches.maximum = max_batch
            vectors = self._image_batches.run(items, self._image.embed_images, getattr(self._image, "device", "cpu"))
        if len(vectors) != len(items):
            raise EmbeddingError(
                f"图像编码器返回 {len(vectors)} 条向量，输入 {len(items)} 张"
            )
        self._check_dim(vectors, self.image_dim, "图像")
        return _l2_normalize_matrix(vectors) if self._normalize else vectors

    def embed_image(self, image: str | Path | Any) -> list[float]:
        vectors = self.embed_images([image])
        return vectors[0]

    def embed_cross_modal(self, text: str) -> list[float]:
        """把文本编码进**图像**空间（SigLIP2 的文本塔）。

        ``VectorChannel`` 用文本 Query 检索关键帧图像向量时走这里，
        与 ``embed_query``（文本空间）不是同一条路。
        """
        with self._image_lock:
            # 先等设备切换锁，再读取模型，避免抓住即将被卸载的旧引用。
            embedder = self._image
            fn = getattr(embedder, "embed_cross_texts", None) or getattr(
                embedder, "embed_texts", None
            )
            if fn is None:
                raise EmbeddingError(
                    f"图像编码器 {self.image_model_name!r} 不支持文本侧编码，无法做跨模态检索"
                )
            with gpu_slot(getattr(embedder, "device", "cpu")):
                vectors = fn([self._truncate(text)])
        if not vectors:
            raise EmbeddingError("SigLIP2 返回空查询向量")
        self._check_dim(vectors, self.image_dim, "跨模态文本")
        return vectors[0]

    # -- 内部 --

    def _truncate(self, text: str) -> str:
        raw = str(text or "")
        return raw[: self._max_text_chars] if len(raw) > self._max_text_chars else raw

    @staticmethod
    def _check_dim(vectors: list[list[float]], expected: int, label: str) -> None:
        if not vectors:
            return
        got = len(vectors[0])
        if got != expected:
            raise EmbeddingError(
                f"{label}向量维度 {got} 与配置的 {expected} 不一致。"
                f"请检查 configs/app_config.yaml 的 embedding 段，"
                f"或换模型后重建向量集合。"
            )

    def unload(self) -> None:
        """释放两个编码器占用的显存/内存。"""
        for embedder in (self._text, self._image):
            unload = getattr(embedder, "unload", None)
            if callable(unload):
                try:
                    unload()
                except Exception:  # noqa: BLE001
                    logger.debug("卸载编码器时忽略异常", exc_info=True)

    def __repr__(self) -> str:
        state = "degraded" if self.degraded else "ready"
        return (
            f"EmbeddingService(text={self.text_model_name}@{self.text_dim}, "
            f"image={self.image_model_name}@{self.image_dim}, {state})"
        )
