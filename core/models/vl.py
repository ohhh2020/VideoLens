"""知影 ZhiYing —— 视觉语言理解（Qwen2-VL，GGUF + llama.cpp）。

职责
----
给一张关键帧，回一段中文描述 + 画面里可读的文字（OCR）。仅此而已。
抽帧在 ``core.media.scene_detect``，图片向量在 ``core.models.embed``，
把描述塞进切片在 ``core.pipeline.orchestrator``。

为什么走 llama.cpp 而不是 transformers
-------------------------------------
Qwen2-VL-2B 的 GGUF 量化版约 1.9GB，``llama.cpp`` 能在纯 CPU 上跑（约 1~3s/帧），
且与本地 LLM（Qwen2.5-7B GGUF）共用同一套运行时，少装一个 torch 栈。
代价是需要额外的 **mmproj**（多模态投影）文件 —— 它和主权重一起放在模型目录里，
由 :class:`core.models.manager.ModelManager` 按 ``models.yaml`` 下载。

降级链（需求文档表 4：视觉模型不可用 → 重试 2 次 → 纯文本总结）
--------------------------------------------------------------
1. 正常推理
2. 单帧重试 ``vl.max_retries`` 次（默认 2）
3. 仍失败 → 返回 ``degraded=True`` 的空结果，**不抛异常**

第 3 步是刻意的：一帧看不懂不该让整条 30 分钟的解析流程崩掉。
调用方拿到 ``degraded=True`` 后继续用转录文本做摘要即可，
但结果里会带 ``error`` 字段，最终报告会把它计入"降级项"。

用法
----
>>> from core.config import get_app_config
>>> from core.models.manager import ModelManager
>>> from core.models.vl import VlService
>>> cfg = get_app_config()
>>> svc = VlService.from_config(cfg, ModelManager(cfg.to_models_config()))
>>> r = svc.describe("data/frames/v1/kf_00000.jpg")
>>> r.ok, r.degraded
(True, False)
>>> r.description[:20]
'画面是一张深色背景的…'
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from core.config import AppConfig, get_app_config
from core.prompts import PromptRegistry, get_prompt_registry

logger = logging.getLogger("zhiying.models.vl")

__all__ = [
    "DEFAULT_VL_PROMPT_KEY",
    "NullVisionModel",
    "OpenAiVisionRunner",
    "QwenVlRunner",
    "VlError",
    "VlResult",
    "VlService",
    "split_description",
]

DEFAULT_VL_PROMPT_KEY: Final[str] = "keyframe_describe"
#: 连续画面（多帧一次推理）用的提示词键。
SEQUENCE_PROMPT_KEY: Final[str] = "keyframe_describe_sequence"

#: 描述与 OCR 的分隔标记（模型输出的常见写法）
_OCR_MARKERS: Final[tuple[str, ...]] = (
    "可读文字：",
    "可读文字:",
    "文字内容：",
    "文字内容:",
    "画面文字：",
    "画面文字:",
    "文字：",
    "文字:",
    "OCR：",
    "OCR:",
)

_MMPROJ_PATTERNS: Final[tuple[str, ...]] = ("*mmproj*.gguf", "*mmproj*.bin", "*mmproj*.gguf.*")


# =============================================================================
# 异常与结果
# =============================================================================


class VlError(RuntimeError):
    """视觉理解失败（加载、推理、输入非法）。"""


@dataclass(slots=True)
class VlResult:
    """单帧理解结果。

    ``degraded=True`` 表示这次没拿到有效描述（模型不可用或重试耗尽），
    ``description`` 为空、``error`` 说明原因。调用方据此走纯文本路径。
    """

    image_path: Path | None = None
    description: str = ""
    ocr_text: str = ""
    raw: str = ""
    model: str = ""
    prompt_key: str = DEFAULT_VL_PROMPT_KEY
    elapsed_ms: int = 0
    attempts: int = 0
    degraded: bool = False
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.description) and not self.degraded

    @property
    def has_ocr(self) -> bool:
        return bool(self.ocr_text.strip())

    def combined_text(self) -> str:
        """描述 + OCR 的合并文本（写向量、做摘要都用它）。"""
        parts = [self.description.strip()]
        if self.has_ocr:
            parts.append(f"[画面文字] {self.ocr_text.strip()}")
        return " ".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": str(self.image_path) if self.image_path else None,
            "description": self.description,
            "ocr_text": self.ocr_text,
            "model": self.model,
            "prompt_key": self.prompt_key,
            "elapsed_ms": self.elapsed_ms,
            "attempts": self.attempts,
            "degraded": self.degraded,
            "ok": self.ok,
            "error": self.error,
            **({"extra": self.extra} if self.extra else {}),
        }


# =============================================================================
# 文本加工
# =============================================================================


def split_description(raw: str) -> tuple[str, str]:
    """把模型输出拆成 ``(描述, OCR 文字)``。

    模型被要求"先描述再列文字"，但输出格式不稳定：有时带"可读文字："小标题，
    有时把文字混在描述里，有时干脆写"无"。这里做保守切分：

    * 找到第一个 OCR 标记 → 前半是描述、后半是文字；
    * 没有标记 → 全文当描述，OCR 为空（宁可不拆，也不要误伤描述）；
    * 文字部分只有"无"/"没有"/"n/a" → 视为无文字。
    """
    text = (raw or "").strip()
    if not text:
        return "", ""

    lowered = text.lower()
    cut = -1
    for marker in _OCR_MARKERS:
        pos = lowered.find(marker.lower())
        if pos >= 0 and (cut < 0 or pos < cut):
            cut = pos

    if cut < 0:
        return _clean_text(text), ""

    marker_len = 0
    for marker in _OCR_MARKERS:
        if lowered.startswith(marker.lower(), cut):
            marker_len = len(marker)
            break
    description = _clean_text(text[:cut])
    ocr = _clean_text(text[cut + marker_len :])
    if ocr.strip().lower() in {"无", "没有", "无文字", "none", "n/a", "na"}:
        ocr = ""
    return description, ocr


def _ms_label(value: int) -> str:
    """毫秒 → ``HH:MM:SS.mmm``。"""
    total = max(0, int(value))
    seconds, millis = divmod(total, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def evenly_spaced(items: Sequence[Any], limit: int) -> list[Any]:
    """从序列里均匀取 ``limit`` 个元素，**保留首尾**。

    时间窗内要压到 2 帧时，取"开头和结尾"比取"开头两帧"更能体现变化 ——
    后者两帧几乎一样，模型会误判为"画面没变"。
    """
    count = len(items)
    limit = max(1, int(limit))
    if count <= limit:
        return list(items)
    if limit == 1:
        return [items[count // 2]]
    step = (count - 1) / float(limit - 1)
    picked = [items[min(count - 1, int(round(i * step)))] for i in range(limit)]
    # 去重（round 可能撞到同一下标），不足时从剩余元素补齐
    seen: set[int] = set()
    unique: list[Any] = []
    for index in range(limit):
        pos = min(count - 1, int(round(index * step)))
        if pos not in seen:
            seen.add(pos)
            unique.append(items[pos])
    for item in items:
        if len(unique) >= limit:
            break
        if item not in unique:
            unique.append(item)
    return unique if len(unique) >= limit else picked


def _clean_text(text: str) -> str:
    """去掉列表符号与多余空行，保留段落结构。"""
    lines: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^[-*•·]\s*", "", stripped)
        stripped = re.sub(r"^\d+[.、)]\s*", "", stripped)
        lines.append(stripped)
    return "\n".join(lines).strip()


def _image_to_data_uri(path: Path, *, max_side: int = 1280, quality: int = 88) -> str:
    """读图 → 必要时缩小 → 转 data URI。

    缩小是为了控制上下文长度：Qwen2-VL 的视觉 token 数随面积增长，
    1280px 长边已经足够看清 PPT 字号，再大只是烧时间。
    """
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise VlError("视觉理解需要 Pillow：pip install Pillow") from exc

    if not path.exists():
        raise VlError(f"关键帧不存在: {path}")
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            width, height = img.size
            longest = max(width, height)
            if max_side > 0 and longest > max_side:
                scale = max_side / float(longest)
                img = img.resize(
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    Image.Resampling.LANCZOS,
                )
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
    except VlError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise VlError(f"读取关键帧失败({path}): {exc}") from exc

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# =============================================================================
# llama.cpp 运行时
# =============================================================================


def _pick_weights(model_dir: Path) -> Path:
    """在模型目录里挑主权重 GGUF（排除 mmproj）。

    优先量化等级：Q4_K_M > Q5_K_M > Q4_0 > Q8_0 > 第一个。2B 模型上
    Q4_K_M 是质量/速度的常用折中点。
    """
    candidates = [
        p
        for p in sorted(model_dir.glob("*.gguf"))
        if "mmproj" not in p.name.lower()
    ]
    if not candidates:
        raise VlError(f"模型目录里没有 .gguf 主权重: {model_dir}")
    for token in ("q4_k_m", "q5_k_m", "q4_0", "q8_0", "q6_k"):
        for path in candidates:
            if token in path.name.lower():
                return path
    return candidates[0]


def _find_mmproj(model_dir: Path) -> Path | None:
    for pattern in _MMPROJ_PATTERNS:
        for path in sorted(model_dir.glob(pattern)):
            if path.is_file():
                return path
    return None


class QwenVlRunner:
    """``llama.cpp`` 上的 Qwen2-VL 推理封装（懒加载 + 可卸载）。"""

    def __init__(
        self,
        model_path: Path,
        mmproj_path: Path,
        *,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        n_gpu_layers: int = -1,
        max_image_side: int = 448,
        verbose: bool = False,
        name: str = "",
    ) -> None:
        self.model_path = Path(model_path)
        self.mmproj_path = Path(mmproj_path)
        self.n_ctx = max(512, int(n_ctx))
        self.n_threads = n_threads
        self.n_gpu_layers = int(n_gpu_layers)
        self.max_image_side = max(224, int(max_image_side))
        self.verbose = bool(verbose)
        self.name = name or self.model_path.stem

        self._llm: Any | None = None
        self._lock = threading.Lock()

        for label, path in (("主权重", self.model_path), ("mmproj", self.mmproj_path)):
            if not path.exists():
                raise VlError(f"{label}文件不存在: {path}")

    # -- 加载 --

    def _load(self) -> Any:
        if self._llm is not None:
            return self._llm
        with self._lock:
            if self._llm is not None:
                return self._llm
            try:
                from llama_cpp import Llama  # noqa: PLC0415
                from llama_cpp import llama_chat_format as _lcf  # noqa: PLC0415
            except ImportError as exc:
                raise VlError(
                    "视觉理解需要 llama-cpp-python：pip install llama-cpp-python"
                ) from exc

            # Qwen25VLChatHandler 名字像"通吃"，实测对 Qwen2-VL-2B-Instruct 会立刻 EOS
            # （finish_reason=stop、completion_tokens=0、正文空串），因为两者 vision token
            # 的语义不同。MTMDChatHandler（llama.cpp 的通用 mtmd 通道）对 Qwen2-VL 正常。
            # 顺序按实测可用性排：MTMD > Qwen25VL > Qwen2VL。
            handler_cls: Any = None
            for _name in ("MTMDChatHandler", "Qwen25VLChatHandler", "Qwen2VLChatHandler"):
                handler_cls = getattr(_lcf, _name, None)
                if handler_cls is not None:
                    logger.debug("使用视觉 chat handler: %s", _name)
                    break
            if handler_cls is None:
                raise VlError(
                    "llama-cpp-python 缺少 Qwen-VL chat handler"
                    "（需要 MTMDChatHandler / Qwen25VLChatHandler / Qwen2VLChatHandler）"
                )

            logger.info(
                "加载 Qwen2-VL: %s (mmproj=%s, n_ctx=%d, n_gpu_layers=%d)",
                self.model_path.name,
                self.mmproj_path.name,
                self.n_ctx,
                self.n_gpu_layers,
            )
            started = time.monotonic()
            try:
                handler = handler_cls(
                    clip_model_path=str(self.mmproj_path), verbose=self.verbose
                )
                kwargs: dict[str, Any] = {
                    "model_path": str(self.model_path),
                    "chat_handler": handler,
                    "n_ctx": self.n_ctx,
                    "n_gpu_layers": self.n_gpu_layers,
                    "verbose": self.verbose,
                    "logits_all": False,
                }
                if self.n_threads:
                    kwargs["n_threads"] = int(self.n_threads)
                self._llm = Llama(**kwargs)
            except Exception as exc:  # noqa: BLE001
                raise VlError(f"加载 Qwen2-VL 失败({self.model_path.name}): {exc}") from exc
            logger.info("Qwen2-VL 就绪，耗时 %.1fs", time.monotonic() - started)
            return self._llm

    # -- 推理 --

    def describe(
        self,
        image_path: Path,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        return self.describe_multi(
            [image_path],
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def describe_multi(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        *,
        stamps: Sequence[int] | None = None,
        max_tokens: int = 384,
        temperature: float = 0.3,
    ) -> str:
        """多帧一次推理。

        实测（本机 CPU，Qwen2-VL-2B Q4_K_M + mmproj-f32, max_image_side=448）：

        ======  ========  ==========  ==================================
        帧数      耗时      生成 token   描述质量
        ======  ========  ==========  ==================================
        1        17.1s     174          准确但会说"似乎发生了变化"（脑补）
        2        14.8s     173          最好，会自己写"首先…接着…"
        4        20.4s     140          开始编造：把同一页说成两段不同内容
        6        26.7s     119          严重幻觉：编出"02/03 知演示"几个不存在的小节
        ======  ========  ==========  ==================================

        所以 2 帧是甜蜜点。逐帧调 6 次要 70 秒，多帧一次只要 27 秒；
        但帧数上去以后质量崩塌，调用方必须自己限制上限（见 VlService）。
        """
        llm = self._load()
        paths = [Path(p) for p in image_paths]
        if not paths:
            raise VlError("多帧推理至少需要一帧")

        content: list[dict[str, Any]] = []
        for index, path in enumerate(paths):
            if stamps is not None and index < len(stamps):
                content.append({"type": "text", "text": f"[{_ms_label(stamps[index])}]"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_to_data_uri(path, max_side=self.max_image_side)
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        try:
            response = llm.create_chat_completion(
                messages=[{"role": "user", "content": content}],
                max_tokens=max(1, int(max_tokens)),
                temperature=float(temperature),
                top_p=0.9,
                repeat_penalty=1.05,
            )
        except Exception as exc:  # noqa: BLE001
            raise VlError(f"Qwen2-VL 多帧推理失败({len(paths)} 帧): {exc}") from exc
        return _extract_content(response)

    def close(self) -> None:
        with self._lock:
            llm, self._llm = self._llm, None
        close = getattr(llm, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                logger.debug("关闭 llama.cpp 视觉模型时忽略异常", exc_info=True)
        if llm is not None:
            logger.info("已释放 Qwen2-VL: %s", self.model_path.name)


def _extract_content(response: Any) -> str:
    """从 ``create_chat_completion`` 的返回里取正文（兼容多种返回形态）。"""
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                chunks = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") in {None, "text"}
                ]
                return "".join(chunks).strip()
        text = response.get("content") or response.get("text")
        if isinstance(text, str):
            return text.strip()
    return str(response).strip()


# =============================================================================
# 云端视觉后端（OpenAI 兼容协议）
# =============================================================================


class OpenAiVisionRunner:
    """``POST {base_url}/chat/completions``，图片走 ``image_url`` + base64 data URI。

    与 :class:`QwenVlRunner` 同接口，所以 ``VlService`` 不用改任何逻辑就能
    换后端。用 ``urllib.request`` 而不是 httpx：本模块要在"只装了 stdlib"
    的环境里也能给出明确错误，而不是 ImportError 堆栈。

    国内可直接用（都是 OpenAI 兼容协议）：

    ==========================  ==========================================
    base_url                    model
    ==========================  ==========================================
    https://dashscope.aliyuncs.com/compatible-mode/v1   qwen-vl-plus
    https://api.openai.com/v1                            gpt-4o-mini
    http://127.0.0.1:8000/v1                             （本地 vLLM/Ollama）
    ==========================  ==========================================
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        *,
        timeout_s: float = 60.0,
        max_image_side: int = 448,
        max_retries: int = 1,
        name: str = "",
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or ""
        self.api_key = api_key or ""
        self.timeout_s = float(timeout_s)
        self.max_image_side = max(224, int(max_image_side))
        self.max_retries = max(0, int(max_retries))
        self.name = name or f"cloud:{self.model}"
        self._lock = threading.Lock()
        self._last_usage: dict[str, int] = {}

    # -- 协议 --

    @property
    def endpoint(self) -> str:
        base = self.base_url
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def available(self) -> bool:
        return bool(self.base_url) and bool(self.model)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "zhiying/0.1 (+local-first video assistant)",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @property
    def last_usage(self) -> dict[str, int]:
        """最近一次调用的 token 用量（界面用来显示成本）。"""
        with self._lock:
            return dict(self._last_usage)

    # -- 推理 --

    def describe(
        self,
        image_path: Path,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        return self.describe_multi(
            [Path(image_path)], prompt, max_tokens=max_tokens, temperature=temperature
        )

    def describe_multi(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        *,
        stamps: Sequence[int] | None = None,
        max_tokens: int = 384,
        temperature: float = 0.3,
    ) -> str:
        if not self.available():
            raise VlError("云端视觉后端未配置 base_url 或 model")
        paths = [Path(p) for p in image_paths]
        if not paths:
            raise VlError("多帧推理至少需要一帧")

        content: list[dict[str, Any]] = []
        for index, path in enumerate(paths):
            if stamps is not None and index < len(stamps):
                content.append({"type": "text", "text": f"[{_ms_label(stamps[index])}]"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_to_data_uri(path, max_side=self.max_image_side)
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max(1, int(max_tokens)),
            "temperature": float(temperature),
            "stream": False,
        }
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=payload, headers=self._headers(), method="POST"
        )

        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    raw = response.read()
                data = json.loads(raw.decode("utf-8"))
                return self._parse(data)
            except urllib.error.HTTPError as exc:
                detail = _read_error_body(exc)
                last_error = f"HTTP {exc.code}: {detail}"
                # 4xx（除 429 限流）重试没有意义，直接抛
                if 400 <= exc.code < 500 and exc.code != 429:
                    break
            except urllib.error.URLError as exc:
                last_error = f"网络错误: {exc.reason}"
            except TimeoutError:
                last_error = f"请求超时({self.timeout_s:.0f}s)"
            except json.JSONDecodeError as exc:
                last_error = f"响应不是合法 JSON: {exc}"
            except VlError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_retries:
                delay = min(4.0, 0.8 * (2**attempt))
                logger.warning("云端视觉第 %d 次失败(%s)，%.1fs 后重试",
                               attempt + 1, last_error, delay)
                time.sleep(delay)
        raise VlError(f"云端视觉调用失败({self.model}): {last_error}")

    def _parse(self, data: Any) -> str:
        if not isinstance(data, dict):
            raise VlError(f"云端视觉响应结构异常: {type(data).__name__}")
        choices = data.get("choices") or []
        if not choices:
            raise VlError(f"云端视觉响应没有 choices: {str(data)[:200]}")
        text = _extract_content(data)
        usage = data.get("usage") or {}
        with self._lock:
            self._last_usage = {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            }
        if not text:
            raise VlError("云端视觉返回空内容")
        return text

    def close(self) -> None:
        """无连接需要释放（每次请求都是新的 urllib 连接池条目）。"""
        logger.debug("云端视觉后端无状态，close() 空操作")


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    """从 HTTPError 里取出可读的错误正文（云端常见 JSON 错误信封）。"""
    try:
        raw = exc.read()
    except Exception:  # noqa: BLE001
        return str(exc)
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return str(exc)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text.strip()[:300] or str(exc)
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)[:300]
        if isinstance(error, str):
            return error[:300]
        if data.get("message"):
            return str(data["message"])[:300]
    return text.strip()[:300] or str(exc)


# =============================================================================
# 降级实现
# =============================================================================


class NullVisionModel:
    """视觉不可用时的占位实现：永远返回降级结果，永不抛异常。"""

    name = "null-vl"

    def describe(
        self, image_path: Path, prompt: str, *, max_tokens: int = 512, temperature: float = 0.1
    ) -> str:
        raise VlError("视觉模型未加载")

    def describe_multi(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        *,
        stamps: Sequence[int] | None = None,
        max_tokens: int = 384,
        temperature: float = 0.3,
    ) -> str:
        raise VlError("视觉模型未加载")

    def close(self) -> None:
        return None


# =============================================================================
# 服务
# =============================================================================


def _cloud_hint(vl_cfg: Any) -> str:
    """给降级原因补一句"云端为什么也没用上"，避免用户误判。"""
    if getattr(vl_cfg, "cloud_enabled", False):
        return "云端视觉已开启但未装配成功"
    return "云端视觉未启用"


class VlService:
    """视觉理解门面：重试、拆分描述/OCR、降级。

    Parameters
    ----------
    runner:
        真实推理器（或 :class:`NullVisionModel`）。``None`` 直接走降级。
    max_retries:
        单帧重试次数（需求文档要求 2 次）。
    """

    def __init__(
        self,
        runner: QwenVlRunner | NullVisionModel | None,
        *,
        prompts: PromptRegistry | None = None,
        prompt_key: str = DEFAULT_VL_PROMPT_KEY,
        max_retries: int = 2,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        sequence_frames: int = 2,
        sequence_temperature: float = 0.3,
        model_name: str = "",
        degraded_reason: str = "",
    ) -> None:
        self._runner = runner
        self._prompts = prompts or get_prompt_registry()
        self._prompt_key = prompt_key
        self._max_retries = max(0, int(max_retries))
        self._max_new_tokens = max(1, int(max_new_tokens))
        self._temperature = float(temperature)
        # 多帧一次推理：2 帧是实测最优（4 帧起小模型会编造差异）
        self._sequence_frames = max(1, int(sequence_frames))
        self._sequence_temperature = float(sequence_temperature)
        self._model_name = model_name or getattr(runner, "name", "unknown")
        self._degraded_reason = degraded_reason

    # -- 构造 --

    @classmethod
    def _try_cloud(
        cls,
        cfg: AppConfig,
        vl_cfg: Any,
        *,
        prompts: PromptRegistry | None = None,
    ) -> VlService | None:
        """尝试装配云端后端；配不全时返回 ``None`` 让调用方继续本地链路。"""
        base_url = (vl_cfg.cloud_base_url or "").strip()
        model = (vl_cfg.cloud_model or "").strip()
        if not base_url or not model:
            logger.warning(
                "已开启云端视觉但缺少参数（base_url=%r model=%r），改用本地",
                base_url, model,
            )
            return None

        api_key = (vl_cfg.cloud_api_key or "").strip()
        if not api_key and vl_cfg.cloud_api_key_env:
            api_key = (os.environ.get(vl_cfg.cloud_api_key_env) or "").strip()
        if not api_key:
            # 本地 vLLM / Ollama 常常不校验密钥，所以这里只提示不阻断
            logger.info(
                "云端视觉未提供密钥（配置项 vl.cloud_api_key 或环境变量 %s），"
                "按无鉴权端点尝试",
                vl_cfg.cloud_api_key_env,
            )

        try:
            runner = OpenAiVisionRunner(
                base_url,
                model,
                api_key,
                timeout_s=float(vl_cfg.cloud_timeout_s),
                max_image_side=vl_cfg.max_image_side,
                max_retries=vl_cfg.cloud_max_retries,
                name=f"cloud:{model}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("云端视觉后端构造失败，改用本地: %s", exc)
            return None

        frames = vl_cfg.cloud_sequence_frames
        logger.info("视觉理解走云端: %s (%s, 每轮最多 %d 帧)", model, base_url, frames)
        return cls(
            runner,
            prompts=prompts,
            prompt_key=vl_cfg.prompt_key,
            max_retries=vl_cfg.cloud_max_retries,
            max_new_tokens=vl_cfg.max_new_tokens,
            temperature=vl_cfg.temperature,
            sequence_frames=frames,
            sequence_temperature=vl_cfg.sequence_temperature,
            model_name=model,
        )

    @classmethod
    def from_config(
        cls,
        config: AppConfig | None = None,
        manager: Any | None = None,
        *,
        prompts: PromptRegistry | None = None,
        load: bool = True,
    ) -> VlService:
        """按配置装配。找不到权重/缺 llama.cpp 时返回降级服务，不抛异常。

        后端选择顺序（``vl.cloud_enabled`` 为真时）：

        1. **云端**：``vl.cloud_base_url`` + ``vl.cloud_model`` 齐备且能解析出
           密钥 → :class:`OpenAiVisionRunner`。这里**不**在装配期打网络探活，
           否则每次启动多几秒；第一帧失败时会由 ``VlService`` 的重试与降级接住。
        2. **本地**：``vl.model`` 的 GGUF + mmproj（原行为）。
        3. **降级**：纯文本。

        云端与本地都不满足时，``degraded_reason`` 会同时写明两边的原因 ——
        只报"本地不可用"会让用户以为是自己的模型没下好。
        """
        cfg = config or get_app_config()
        vl_cfg = cfg.vl

        if vl_cfg.cloud_enabled:
            cloud = cls._try_cloud(cfg, vl_cfg, prompts=prompts)
            if cloud is not None:
                return cloud

        common: dict[str, Any] = {
            "prompts": prompts,
            "prompt_key": vl_cfg.prompt_key,
            "max_retries": vl_cfg.max_retries,
            "max_new_tokens": vl_cfg.max_new_tokens,
            "temperature": vl_cfg.temperature,
            "sequence_frames": vl_cfg.sequence_frames,
            "sequence_temperature": vl_cfg.sequence_temperature,
            "model_name": vl_cfg.model,
        }

        if manager is None:
            logger.warning("未提供 ModelManager，视觉理解降级为纯文本")
            return cls(
                None,
                degraded_reason=_cloud_hint(vl_cfg)
                + "；未提供 ModelManager，无法定位本地视觉权重",
                **common,
            )

        try:
            model_dir = Path(manager.ensure_model(vl_cfg.model))
        except Exception as exc:  # noqa: BLE001
            logger.warning("视觉模型权重不可用，降级为纯文本: %s", exc)
            return cls(None, degraded_reason=f"权重不可用({vl_cfg.model}): {exc}", **common)

        try:
            weights = _pick_weights(model_dir)
            mmproj = _find_mmproj(model_dir)
        except VlError as exc:
            logger.warning("视觉模型文件不完整，降级为纯文本: %s", exc)
            return cls(None, degraded_reason=str(exc), **common)

        if mmproj is None:
            reason = (
                f"缺少 mmproj 多模态投影文件（{model_dir} 下没有 *mmproj*.gguf）。"
                f"Qwen2-VL 的 GGUF 推理必须同时提供主权重与 mmproj。"
            )
            logger.warning("视觉理解降级: %s", reason)
            return cls(None, degraded_reason=reason, **common)

        n_gpu_layers = -1 if cfg.models.device in {"auto", "cuda"} else 0
        try:
            runner = QwenVlRunner(
                weights,
                mmproj,
                n_ctx=max(4096, vl_cfg.max_new_tokens * 4),
                n_gpu_layers=n_gpu_layers,
                max_image_side=vl_cfg.max_image_side,
                name=vl_cfg.model,
            )
            if load:
                runner._load()  # noqa: SLF001 - 提前暴露权重损坏，而不是等到第一帧
        except Exception as exc:  # noqa: BLE001
            logger.warning("Qwen2-VL 加载失败，降级为纯文本: %s", exc)
            return cls(None, degraded_reason=f"加载失败: {exc}", **common)

        logger.info("视觉理解就绪: %s (%s)", vl_cfg.model, weights.name)
        return cls(runner, **common)

    # -- 属性 --

    @property
    def available(self) -> bool:
        return self._runner is not None and not self._degraded_reason

    @property
    def degraded_reason(self) -> str:
        return self._degraded_reason

    @property
    def model_name(self) -> str:
        return self._model_name

    def info(self) -> dict[str, Any]:
        return {
            "model": self._model_name,
            "available": self.available,
            "prompt_key": self._prompt_key,
            "max_retries": self._max_retries,
            "degraded_reason": self._degraded_reason,
        }

    # -- 理解 --

    def describe(
        self,
        image_path: str | Path,
        *,
        prompt: str | None = None,
        prompt_key: str | None = None,
        extra_vars: dict[str, Any] | None = None,
    ) -> VlResult:
        """理解单帧。**永远返回 VlResult**，失败时 ``degraded=True``。"""
        path = Path(image_path)
        key = prompt_key or self._prompt_key
        started = time.monotonic()

        if self._runner is None:
            return VlResult(
                image_path=path,
                model=self._model_name,
                prompt_key=key,
                degraded=True,
                error=self._degraded_reason or "视觉模型未加载",
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )

        text_prompt = prompt if prompt is not None else self._render(key, extra_vars)

        last_error = ""
        attempts = 0
        for attempt in range(1, self._max_retries + 2):  # 首次 + 重试
            attempts = attempt
            try:
                raw = self._runner.describe(
                    path,
                    text_prompt,
                    max_tokens=self._max_new_tokens,
                    temperature=self._temperature,
                )
            except Exception as exc:  # noqa: BLE001 - 单帧失败不阻断流程
                last_error = str(exc)
                logger.warning("视觉理解第 %d 次失败(%s): %s", attempt, path.name, exc)
                if attempt <= self._max_retries:
                    time.sleep(min(1.0, 0.2 * attempt))
                continue

            description, ocr = split_description(raw)
            if not description and not ocr:
                last_error = "模型返回空内容"
                logger.warning("视觉理解返回空内容(%s)，第 %d 次", path.name, attempt)
                continue
            return VlResult(
                image_path=path,
                description=description,
                ocr_text=ocr,
                raw=raw,
                model=self._model_name,
                prompt_key=key,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                attempts=attempts,
            )

        logger.warning("视觉理解降级(%s): %s", path.name, last_error)
        return VlResult(
            image_path=path,
            model=self._model_name,
            prompt_key=key,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            attempts=attempts,
            degraded=True,
            error=last_error or "视觉理解失败",
        )

    def describe_many(
        self,
        image_paths: Sequence[str | Path],
        *,
        prompt_key: str | None = None,
        multi_key: str = "keyframe_describe_multi",
    ) -> list[VlResult]:
        """批量理解：按 ``sequence_frames`` 分批做多帧推理。

        返回的列表长度是"批数"，不是"帧数" —— 一批产出一段连续叙述，
        由调用方决定怎么落到具体帧上。
        """
        _ = multi_key  # 保留旧参数名，实际已改走 describe_sequence
        paths = [Path(p) for p in image_paths]
        if not paths:
            return []
        size = max(1, self._sequence_frames)
        results: list[VlResult] = []
        for start in range(0, len(paths), size):
            batch = paths[start : start + size]
            results.append(
                self.describe_sequence(
                    [(p, 0) for p in batch],
                    prompt_key=prompt_key or "keyframe_describe_multi",
                )
            )
        return results

    def describe_sequence(
        self,
        frames: Sequence[tuple[str | Path, int]],
        *,
        span_ms: tuple[int, int] | None = None,
        prompt: str | None = None,
        prompt_key: str = SEQUENCE_PROMPT_KEY,
        max_frames: int | None = None,
    ) -> VlResult:
        """把一个时间窗内的多帧**一次**送进模型，得到一段连续叙述。

        与逐个 ``describe`` 的区别不是性能，是**语义**：单帧调用时模型
        看不到前后帧，会写出"接着画面似乎发生了变化"这类凭空推断；
        多帧一起给它，它自己就能对齐"起初…随后…"的时序。

        帧数上限由 :attr:`_sequence_frames` 控制（实测 2 帧最优，
        4 帧起小模型开始为了找差异而编造内容）。
        """
        pairs = [(Path(p), max(0, int(ms))) for p, ms in frames]
        if not pairs:
            return VlResult(
                model=self._model_name,
                prompt_key=prompt_key,
                degraded=True,
                error="没有可用的画面",
            )

        limit = max_frames if max_frames is not None else self._sequence_frames
        limit = max(1, int(limit))
        if len(pairs) > limit:
            pairs = evenly_spaced(pairs, limit)

        first_path, first_ms = pairs[0]
        lo = max(0, int(span_ms[0])) if span_ms else first_ms
        hi = int(span_ms[1]) if span_ms else pairs[-1][1]
        started = time.monotonic()

        if self._runner is None:
            return VlResult(
                image_path=first_path,
                model=self._model_name,
                prompt_key=prompt_key,
                degraded=True,
                error=self._degraded_reason or "视觉模型未加载",
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )

        if prompt is not None:
            text_prompt = prompt
        else:
            text_prompt = self._render(
                prompt_key,
                {
                    "frame_count": len(pairs),
                    "start_time": _ms_label(lo),
                    "end_time": _ms_label(hi),
                },
            )

        stamps = [ms for _, ms in pairs]
        paths = [p for p, _ in pairs]
        last_error = ""
        attempts = 0
        for attempt in range(1, self._max_retries + 2):
            attempts = attempt
            try:
                raw = self._runner.describe_multi(
                    paths,
                    text_prompt,
                    stamps=stamps,
                    max_tokens=self._max_new_tokens,
                    temperature=self._sequence_temperature,
                )
            except Exception as exc:  # noqa: BLE001 - 一段失败不阻断整个视频
                last_error = str(exc)
                logger.warning(
                    "连续画面理解第 %d 次失败(%d 帧): %s", attempt, len(paths), exc
                )
                if attempt <= self._max_retries:
                    time.sleep(min(1.0, 0.2 * attempt))
                continue

            description, ocr = split_description(raw)
            if not description and not ocr:
                last_error = "模型返回空内容"
                logger.warning("连续画面理解返回空内容(%d 帧)，第 %d 次", len(paths), attempt)
                continue
            return VlResult(
                image_path=first_path,
                description=description,
                ocr_text=ocr,
                raw=raw,
                model=self._model_name,
                prompt_key=prompt_key,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                attempts=attempts,
                extra={
                    "frame_count": len(paths),
                    "frame_paths": [str(p) for p in paths],
                    "stamps": stamps,
                    "span_ms": [lo, hi],
                },
            )

        logger.warning("连续画面理解降级(%d 帧): %s", len(paths), last_error)
        return VlResult(
            image_path=first_path,
            model=self._model_name,
            prompt_key=prompt_key,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            attempts=attempts,
            degraded=True,
            error=last_error or "连续画面理解失败",
        )

    def _render(self, key: str, variables: dict[str, Any] | None = None) -> str:
        try:
            return self._prompts.render(key, **(variables or {}))
        except KeyError:
            logger.warning("提示词 %r 不存在，使用任务描述兜底", key)
            return "请用一句话描述这一帧画面，并列出画面中所有可读文字。"

    # -- 生命周期 --

    def close(self) -> None:
        close = getattr(self._runner, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                logger.debug("释放视觉模型时忽略异常", exc_info=True)

    def __enter__(self) -> VlService:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "ready" if self.available else f"degraded({self._degraded_reason[:40]})"
        return f"VlService({self._model_name}, {state})"
