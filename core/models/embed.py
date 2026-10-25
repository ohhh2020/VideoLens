"""知影 ZhiYing —— 向量编码服务（BGE-M3 文本 / SigLIP2 图文跨模态）。

职责
----
把"文本"和"图片"变成向量，别的什么都不做。写入 Chroma、算相似度、融合排序
分别属于 ``core.rag.vector_store`` 与 ``core.rag.hybrid_search``。

两个编码器，两种空间
--------------------
* **BGE-M3**（``bge-m3``，1024 维）：中文/多语言文本语义。池化用 CLS
  （``last_hidden_state[:, 0]``），随后 L2 归一化 —— BGE 系列的官方用法。
* **SigLIP2**（``siglip2-base``，768 维）：图文共享空间。文本查询与关键帧图像
  编码进同一空间，所以"架构图里的融合节点长什么样"能直接命中图片。

维度不同的两套向量**不能混在一个集合里**。``clip_text`` 存 1024 维，
``frame_image`` 存 768 维，各自独立配置集合（见 ``core.rag.vector_store``）。

降级链
------
需求文档表 4 要求"CPU 降级"与"模型不可用不白屏"。这里的降级顺序是：

1. 真实模型（``transformers`` + 权重目录，GPU 优先）
2. 真实模型但强制 CPU（``torch`` 可用、CUDA 显存不足时）
3. :class:`HashingEmbedder` —— 纯 Python 确定性哈希向量

第 3 档不是"能用的语义向量"，它只保证**整条链路可跑、接口形状一致、结果可复现**，
用于无网/无权重环境做联调与冒烟。日志里会明确打 WARNING，绝不悄悄降级。

典型用法
--------
>>> svc = EmbeddingService.from_config(get_app_config())
>>> svc.text_dim, svc.image_dim
(1024, 768)
>>> v = svc.embed_query("切片长度是怎么定的")
>>> len(v)
1024
>>> m = svc.embed_texts(["第一段", "第二段"])
>>> len(m), len(m[0])
(2, 1024)
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from core.config import AppConfig, get_app_config

logger = logging.getLogger("zhiying.models.embed")

__all__ = [
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
    from core.models._runtime import ensure_torch_path  # noqa: PLC0415

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
    """读编码器维度。

    真实编码器（BGE-M3 / SigLIP2）暴露统一的 ``dim`` 字段；
    降级的 :class:`HashingEmbedder` 用 ``text_dim`` / ``image_dim`` 区分两个空间。
    这里按这个优先级读，避免"Hashing 128 维被当成默认 1024 维"的静默错配。
    """
    for name in ("dim", attr):
        value = getattr(embedder, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return default


# =============================================================================
# 降级编码器
# =============================================================================


class HashingEmbedder:
    """确定性哈希向量（无模型依赖的降级通道）。

    文本侧：jieba 分词（可用时）→ 每个 token 的 MD5 落到某个维度 → 累加 → L2 归一化。
    这实际上是一种 "hashing trick" 词袋，**保留字面相似性**（同词同维度），
    但不具备语义泛化能力 —— 同义词不会靠近。用来联调足够。

    图像侧：缩到 32×24 灰度（768 个像素）后归一化。近似感知哈希，
    同画面不同编码会得到相近向量。

    两个维度都可以自定义，超出 768 的图像维度用哈希重采样补齐。
    """

    def __init__(
        self,
        *,
        text_dim: int = DEFAULT_TEXT_DIM,
        image_dim: int = DEFAULT_IMAGE_DIM,
        tokenizer: str = "jieba",
        name: str = "hashing",
    ) -> None:
        if text_dim <= 0 or image_dim <= 0:
            raise EmbeddingError("维度必须为正整数")
        self.name = name
        self.text_dim = int(text_dim)
        self.image_dim = int(image_dim)
        self._tokenizer_kind = tokenizer
        self._jieba: Any | None = None

    # -- 分词 --

    def _tokenize(self, text: str) -> list[str]:
        raw = (text or "").strip()
        if not raw:
            return []
        if self._tokenizer_kind == "jieba":
            if self._jieba is None:
                try:
                    import jieba  # noqa: PLC0415

                    jieba.setLogLevel(logging.WARNING)
                    self._jieba = jieba
                except ImportError:
                    logger.debug("jieba 不可用，hashing 编码退化为字符 bigram")
                    self._jieba = False
            if self._jieba:
                return [t for t in self._jieba.lcut(raw) if t.strip()]
        # 字符 + 二元组：中文无空格时也能提供区分度
        chars = [c for c in raw if not c.isspace()]
        bigrams = ["".join(chars[i : i + 2]) for i in range(len(chars) - 1)]
        return chars + bigrams

    @staticmethod
    def _hash_index(token: str, dim: int) -> int:
        digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).digest()
        return int.from_bytes(digest[:8], "big") % dim

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.text_dim
            for token in self._tokenize(text):
                vec[self._hash_index(token, self.text_dim)] += 1.0
            out.append(_l2_normalize(vec))
        return out

    def embed_images(self, images: Sequence[str | Path | Any]) -> list[list[float]]:
        out: list[list[float]] = []
        for item in images:
            out.append(self._image_vector(item))
        return out

    def embed_cross_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """把文本编码进**图像维度**的空间（跨模态通道的降级实现）。

        真实的 SigLIP2 有独立文本塔；哈希实现没有，于是把文本哈希到
        ``image_dim`` 上 —— 语义为零，但保证 ``VectorChannel`` 不会因为
        维度不一致而整条通道失效。
        """
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.image_dim
            for token in self._tokenize(text):
                vec[self._hash_index(token, self.image_dim)] += 1.0
            out.append(_l2_normalize(vec))
        return out

    def _image_vector(self, item: str | Path | Any) -> list[float]:
        try:
            from PIL import Image  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError("图像编码需要 Pillow：pip install Pillow") from exc
        if isinstance(item, (str, Path)):
            with Image.open(item) as img:
                thumb = img.convert("L").resize((32, 24))
                pixels = [float(p) / 255.0 for p in thumb.getdata()]
        else:
            thumb = item.convert("L").resize((32, 24))
            pixels = [float(p) / 255.0 for p in thumb.getdata()]
        if len(pixels) == self.image_dim:
            return _l2_normalize(pixels)
        # 维度不一致：按哈希重采样（保持相对排序的粗略近似）
        vec = [0.0] * self.image_dim
        for pos, value in enumerate(pixels):
            vec[self._hash_index(f"p{pos}", self.image_dim)] += value
        return _l2_normalize(vec)


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
                    "BGE-M3 编码需要 transformers：pip install -e \".[embed]\""
                ) from exc
            logger.info("加载 BGE-M3: %s (device=%s)", self.model_dir, self.device)
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_dir), trust_remote_code=True
                )
                model = AutoModel.from_pretrained(
                    str(self.model_dir), trust_remote_code=True
                )
                model.eval()
                model.to(self.device)
            except Exception as exc:  # noqa: BLE001 - transformers 异常类型繁杂
                raise EmbeddingError(f"加载 BGE-M3 失败({self.model_dir}): {exc}") from exc
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
                raise EmbeddingError(f"BGE-M3 编码失败（第 {start}-{end} 条）: {exc}") from exc
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

    transformers 对 SigLIP2 的支持从 4.52 起（``Siglip2Model``）。更早的版本
    会走 ``AutoModel`` 的通用路径；若仍不可用，:class:`EmbeddingService`
    会降级到哈希编码器并给出明确告警。
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
                from transformers import AutoModel, AutoProcessor  # noqa: PLC0415
            except ImportError as exc:
                raise EmbeddingError(
                    "SigLIP2 编码需要 transformers：pip install -e \".[embed]\""
                ) from exc
            logger.info("加载 SigLIP2: %s (device=%s)", self.model_dir, self.device)
            try:
                processor = AutoProcessor.from_pretrained(str(self.model_dir))
                model = AutoModel.from_pretrained(str(self.model_dir))
                model.eval()
                model.to(self.device)
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(f"加载 SigLIP2 失败({self.model_dir}): {exc}") from exc
            if not (hasattr(model, "get_image_features") and hasattr(model, "get_text_features")):
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
                        opened.append(item.convert("RGB") if hasattr(item, "convert") else item)
                inputs = processor(images=opened, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    feats = _feature_tensor(model.get_image_features(**inputs))
                feats = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
                out.extend(feats.float().cpu().tolist())
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(f"SigLIP2 图像编码失败（第 {start}-{end} 张）: {exc}") from exc
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
                    text=batch, padding="max_length", truncation=True, return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    feats = _feature_tensor(model.get_text_features(**inputs))
                feats = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
                out.extend(feats.float().cpu().tolist())
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(f"SigLIP2 文本编码失败（第 {start}-{end} 条）: {exc}") from exc
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
        注入编码器。为 ``None`` 时用 :class:`HashingEmbedder` 兜底。
    normalize:
        是否对输出再做一次 L2 归一化（编码器内部通常已经做过，这里兜底）。
    max_text_chars:
        单条文本截断长度。BGE-M3 上限 8192 token，超长直接截字符更省事。
    """

    def __init__(
        self,
        text_embedder: TextEmbedder | None = None,
        image_embedder: ImageEmbedder | None = None,
        *,
        batch_size: int = 16,
        normalize: bool = True,
        max_text_chars: int = 8192,
        degraded_reason: str = "",
    ) -> None:
        fallback = HashingEmbedder()
        # 只给一个降级器时，用它同时承担文本与图像两个空间（维度按各自属性取）
        if image_embedder is None and isinstance(text_embedder, HashingEmbedder):
            image_embedder = text_embedder
        self._text = text_embedder or fallback
        self._image = image_embedder or fallback
        self._batch_size = max(1, int(batch_size))
        self._normalize = bool(normalize)
        self._max_text_chars = max(1, int(max_text_chars))
        if not degraded_reason and (
            isinstance(self._text, HashingEmbedder) or isinstance(self._image, HashingEmbedder)
        ):
            degraded_reason = "未注入真实编码器，使用哈希降级（接口可用，语义精度为零）"
        self._degraded_reason = degraded_reason
        self._lock = threading.Lock()

    # -- 构造 --

    @classmethod
    def from_config(
        cls,
        config: AppConfig | None = None,
        *,
        manager: Any | None = None,
        allow_degraded: bool = True,
    ) -> EmbeddingService:
        """按配置加载两个真实编码器；任一失败则整体降级到哈希编码。

        ``manager`` 传 :class:`core.models.manager.ModelManager`，由其负责
        权重路径（本模块**不**拼任何权重路径）。
        """
        cfg = config or get_app_config()
        text_cfg = cfg.embedding
        reasons: list[str] = []
        text_emb: TextEmbedder | None = None
        image_emb: ImageEmbedder | None = None

        if manager is not None:
            # 文本侧
            try:
                text_dir = manager.ensure_model(text_cfg.text_model)
                text_emb = BgeM3TextEmbedder(
                    text_dir,
                    device=cfg.models.device,
                    batch_size=text_cfg.batch_size,
                    max_length=text_cfg.max_text_chars,
                    name=text_cfg.text_model,
                    dim=text_cfg.text_dim,
                )
                logger.info("文本编码器就绪: %s", text_cfg.text_model)
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"文本编码器({text_cfg.text_model})不可用: {exc}")
                logger.warning("文本编码器不可用，将降级: %s", exc)
            # 图像侧
            try:
                image_dir = manager.ensure_model(text_cfg.image_model)
                image_emb = SigLip2ImageEmbedder(
                    image_dir,
                    device=cfg.models.device,
                    batch_size=text_cfg.batch_size,
                    name=text_cfg.image_model,
                    dim=text_cfg.image_dim,
                )
                logger.info("图像编码器就绪: %s", text_cfg.image_model)
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"图像编码器({text_cfg.image_model})不可用: {exc}")
                logger.warning("图像编码器不可用，将降级: %s", exc)
        else:
            reasons.append("未提供 ModelManager，无法定位权重")

        if not allow_degraded and reasons:
            raise EmbeddingError("；".join(reasons))

        if reasons:
            logger.warning(
                "向量编码降级为哈希编码（链路可跑，语义精度下降）：%s", "；".join(reasons)
            )
            fallback_kwargs = {
                "text_dim": text_cfg.text_dim,
                "image_dim": text_cfg.image_dim,
            }
            return cls(
                text_emb or HashingEmbedder(**fallback_kwargs),
                image_emb or HashingEmbedder(**fallback_kwargs),
                batch_size=text_cfg.batch_size,
                normalize=text_cfg.normalize,
                max_text_chars=text_cfg.max_text_chars,
                degraded_reason="；".join(reasons),
            )

        return cls(
            text_emb,
            image_emb,
            batch_size=text_cfg.batch_size,
            normalize=text_cfg.normalize,
            max_text_chars=text_cfg.max_text_chars,
        )

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

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """批量文本编码。空输入返回 ``[]``，空字符串返回零向量。"""
        items = [self._truncate(t) for t in texts]
        if not items:
            return []
        with self._lock:
            vectors = self._text.embed_texts(items)
        if len(vectors) != len(items):
            raise EmbeddingError(
                f"文本编码器返回 {len(vectors)} 条向量，输入 {len(items)} 条"
            )
        self._check_dim(vectors, self.text_dim, "文本")
        return _l2_normalize_matrix(vectors) if self._normalize else vectors

    def embed_query(self, text: str) -> list[float]:
        """单条查询编码（BGE 系列查询不需要指令前缀，与文档同分布）。"""
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else [0.0] * self.text_dim

    def embed_images(self, images: Sequence[str | Path | Any]) -> list[list[float]]:
        """批量图像编码（路径或 PIL 对象）。"""
        items = list(images)
        if not items:
            return []
        with self._lock:
            vectors = self._image.embed_images(items)
        if len(vectors) != len(items):
            raise EmbeddingError(
                f"图像编码器返回 {len(vectors)} 条向量，输入 {len(items)} 张"
            )
        self._check_dim(vectors, self.image_dim, "图像")
        return _l2_normalize_matrix(vectors) if self._normalize else vectors

    def embed_image(self, image: str | Path | Any) -> list[float]:
        vectors = self.embed_images([image])
        return vectors[0] if vectors else [0.0] * self.image_dim

    def embed_cross_modal(self, text: str) -> list[float]:
        """把文本编码进**图像**空间（SigLIP2 的文本塔）。

        ``VectorChannel`` 用文本 Query 检索关键帧图像向量时走这里，
        与 ``embed_query``（文本空间）不是同一条路。
        """
        embedder = self._image
        # 优先用专门的跨模态文本塔（SigLIP2 的 get_text_features / 哈希降级实现）
        fn = getattr(embedder, "embed_cross_texts", None) or getattr(embedder, "embed_texts", None)
        if fn is None:
            raise EmbeddingError(
                f"图像编码器 {self.image_model_name!r} 不支持文本侧编码，无法做跨模态检索"
            )
        with self._lock:
            vectors = fn([self._truncate(text)])
        if not vectors:
            return [0.0] * self.image_dim
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
