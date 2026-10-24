"""知影 ZhiYing —— LLM 网关（云端 OpenAI 协议 / 本地 GGUF / Ollama / 关键词兜底）。

四档后端，一条降级链
--------------------
需求文档表 4 的原始要求："云端超时 → 本地 llama.cpp/Ollama → jieba 关键词兜底"。
这一层就是那条链的落地点，业务代码只调 :class:`LlmGateway`，不关心后端是谁。

======================  ==========================================  ============
后端                    适用                                        实测延迟
======================  ==========================================  ============
``cloud``              有 API Key、网络可用、追求质量              2~8s
``local`` GGUF         llama.cpp，隐私敏感、离线                    30~90s (7B CPU)
``cloud``(本地服务)    Ollama / LM Studio 的 OpenAI 兼容端点        取决于部署
``keywords``           全部不可用，也要给出可读结果                20~80ms
======================  ==========================================  ============

``cloud`` 走 **OpenAI 兼容协议**（``POST {base_url}/chat/completions``），所以
DeepSeek / Moonshot / 通义 / vLLM / Ollama 的 ``/v1`` 都能直接用 —— 换 ``base_url``
与 ``model`` 即可，代码不动。

隐私约束（需求文档 3.4）
-----------------------
"本地模式下不产生携用户视频内容的外发请求"。:class:`LlmGateway` 在
``mode="local"`` 时**根本不构造** cloud 后端；``mode="auto"`` 下是否允许走云端
由 :class:`core.config.AppConfig` 的 ``storage.privacy.local_mode`` 与
``llm.mode`` 共同决定，不在这一层偷偷判断。

用法
----
>>> gw = LlmGateway.from_config()
>>> r = gw.complete([LlmMessage("user", "用一句话说明什么是 RRF 融合")])
>>> r.backend in {"cloud", "local", "keywords"}
True
>>> r.degraded
False
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from core.config import AppConfig, get_app_config

logger = logging.getLogger("zhiying.models.llm")

__all__ = [
    "KeywordBackend",
    "LlmBackend",
    "LlmError",
    "LlmGateway",
    "LlmMessage",
    "LlmResponse",
    "LlamaCppChatBackend",
    "OpenAICompatibleBackend",
    "VALID_ROLES",
    "detect_local_server",
    "local_server_backend",
]

VALID_ROLES: Final[frozenset[str]] = frozenset({"system", "user", "assistant"})


# =============================================================================
# 异常与数据结构
# =============================================================================


class LlmError(RuntimeError):
    """LLM 调用失败（配置缺失、网络错误、后端拒绝）。"""


@dataclass(frozen=True, slots=True)
class LlmMessage:
    """一条对话消息。``role`` 只允许 system/user/assistant。"""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise LlmError(f"非法 role {self.role!r}，只允许 {sorted(VALID_ROLES)}")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def normalize_messages(messages: Sequence[LlmMessage | dict[str, Any] | str]) -> list[LlmMessage]:
    """把松散输入统一成 :class:`LlmMessage` 列表（字符串按 user 处理）。"""
    out: list[LlmMessage] = []
    for item in messages:
        if isinstance(item, LlmMessage):
            out.append(item)
        elif isinstance(item, str):
            out.append(LlmMessage("user", item))
        elif isinstance(item, dict):
            role = str(item.get("role", "user"))
            content = item.get("content", "")
            if isinstance(content, list):  # 多段内容只取文本部分
                content = "".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict)
                )
            out.append(LlmMessage(role if role in VALID_ROLES else "user", str(content)))
        else:
            raise LlmError(f"无法识别的消息类型: {type(item).__name__}")
    if not out:
        raise LlmError("messages 不能为空")
    return out


@dataclass(slots=True)
class LlmResponse:
    """一次补全结果。``degraded=True`` 表示走的是兜底后端。"""

    text: str
    backend: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: int = 0
    finish_reason: str = ""
    degraded: bool = False
    error: str = ""
    attempts: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "backend": self.backend,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "elapsed_ms": self.elapsed_ms,
            "finish_reason": self.finish_reason,
            "degraded": self.degraded,
            "error": self.error,
            "attempts": list(self.attempts),
        }


# =============================================================================
# 协议
# =============================================================================


@runtime_checkable
class LlmBackend(Protocol):
    """LLM 后端。``complete`` 失败必须抛 :class:`LlmError`，由网关决定降级。"""

    name: str
    model: str

    def available(self) -> bool: ...

    def complete(
        self,
        messages: Sequence[LlmMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: Sequence[str] = (),
        timeout: float | None = None,
    ) -> LlmResponse: ...


# =============================================================================
# 云端：OpenAI 兼容协议
# =============================================================================


class OpenAICompatibleBackend:
    """``POST {base_url}/chat/completions``。

    用 ``urllib.request`` 而不是 httpx/requests：本模块要在"只装了 stdlib"的
    环境里也能给出明确错误（而不是 ImportError 堆栈）。流式用标准 SSE 解析。
    """

    name = "cloud"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.extra_headers = dict(extra_headers or {})
        self._lock = threading.Lock()

    def available(self) -> bool:
        return bool(self.base_url) and bool(self.model)

    @property
    def endpoint(self) -> str:
        base = self.base_url
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "zhiying/0.1 (+local-first video assistant)",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _payload(
        self,
        messages: Sequence[LlmMessage],
        max_tokens: int,
        temperature: float,
        stop: Sequence[str],
        stream: bool,
    ) -> bytes:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": bool(stream),
        }
        if stop:
            body["stop"] = list(stop)
        return json.dumps(body, ensure_ascii=False).encode("utf-8")

    def complete(
        self,
        messages: Sequence[LlmMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: Sequence[str] = (),
        timeout: float | None = None,
    ) -> LlmResponse:
        if not self.available():
            raise LlmError("云端后端未配置 base_url 或 model")
        started = time.monotonic()
        payload = self._payload(
            messages,
            max_tokens if max_tokens is not None else 2048,
            temperature if temperature is not None else 0.3,
            stop,
            stream=False,
        )
        request = urllib.request.Request(
            self.endpoint, data=payload, headers=self._headers(), method="POST"
        )
        timeout_s = float(timeout if timeout is not None else self.timeout_s)

        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout_s) as response:
                    raw = response.read()
                data = json.loads(raw.decode("utf-8"))
                return self._parse(data, started)
            except urllib.error.HTTPError as exc:
                detail = _read_error_body(exc)
                last_error = f"HTTP {exc.code}: {detail}"
                # 4xx（除 429）重试无意义
                if 400 <= exc.code < 500 and exc.code != 429:
                    break
            except urllib.error.URLError as exc:
                last_error = f"网络错误: {exc.reason}"
            except TimeoutError:
                last_error = f"请求超时({timeout_s:.0f}s)"
            except json.JSONDecodeError as exc:
                last_error = f"响应不是合法 JSON: {exc}"
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_retries:
                delay = min(4.0, 0.8 * (2**attempt))
                logger.warning("云端 LLM 第 %d 次失败(%s)，%.1fs 后重试", attempt + 1, last_error, delay)
                time.sleep(delay)
        raise LlmError(f"云端 LLM 调用失败: {last_error}")

    def _parse(self, data: Any, started: float) -> LlmResponse:
        if not isinstance(data, dict):
            raise LlmError(f"响应结构异常: {type(data).__name__}")
        choices = data.get("choices") or []
        if not choices:
            raise LlmError(f"响应没有 choices: {str(data)[:200]}")
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        if isinstance(text, list):
            text = "".join(str(p.get("text", "")) for p in text if isinstance(p, dict))
        usage = data.get("usage") or {}
        return LlmResponse(
            text=str(text).strip(),
            backend=self.name,
            model=str(data.get("model") or self.model),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            elapsed_ms=int((time.monotonic() - started) * 1000),
            finish_reason=str(choices[0].get("finish_reason") or ""),
        )

    def stream(
        self,
        messages: Sequence[LlmMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: Sequence[str] = (),
        timeout: float | None = None,
    ) -> Iterator[str]:
        """SSE 流式增量（供 FastAPI 的 ``/ask/stream`` 使用）。"""
        if not self.available():
            raise LlmError("云端后端未配置 base_url 或 model")
        payload = self._payload(
            messages,
            max_tokens if max_tokens is not None else 2048,
            temperature if temperature is not None else 0.3,
            stop,
            stream=True,
        )
        request = urllib.request.Request(
            self.endpoint, data=payload, headers=self._headers(), method="POST"
        )
        timeout_s = float(timeout if timeout is not None else self.timeout_s)
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    for choice in obj.get("choices") or []:
                        delta = choice.get("delta") or {}
                        piece = delta.get("content")
                        if piece:
                            yield str(piece)
        except urllib.error.HTTPError as exc:
            raise LlmError(f"云端流式请求失败 HTTP {exc.code}: {_read_error_body(exc)}") from exc
        except urllib.error.URLError as exc:
            raise LlmError(f"云端流式请求网络错误: {exc.reason}") from exc


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return str(exc.reason)
    try:
        obj = json.loads(body)
        error = obj.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or body)[:300]
        return str(error or body)[:300]
    except json.JSONDecodeError:
        return body[:300]


# =============================================================================
# 本地：llama.cpp GGUF
# =============================================================================


class LlamaCppChatBackend:
    """本地 GGUF 推理（``llama-cpp-python``）。

    默认全量上 GPU（``n_gpu_layers=-1``），无 CUDA 时自动退 ``0``（纯 CPU）。
    7B Q4_K_M 在 8 核 CPU 上约 4~8 token/s，够做"离线总结"。
    """

    name = "local"

    def __init__(
        self,
        model_path: Path,
        model: str = "",
        *,
        n_ctx: int = 8192,
        n_threads: int | None = None,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.model = model or self.model_path.stem
        self.n_ctx = max(512, int(n_ctx))
        self.n_threads = n_threads
        self.n_gpu_layers = int(n_gpu_layers)
        self.verbose = bool(verbose)
        self._llm: Any | None = None
        self._lock = threading.Lock()
        if not self.model_path.exists():
            raise LlmError(f"本地模型文件不存在: {self.model_path}")

    def available(self) -> bool:
        return self.model_path.exists()

    def _load(self) -> Any:
        if self._llm is not None:
            return self._llm
        with self._lock:
            if self._llm is not None:
                return self._llm
            try:
                from llama_cpp import Llama  # noqa: PLC0415
            except ImportError as exc:
                raise LlmError(
                    "本地推理需要 llama-cpp-python：pip install llama-cpp-python"
                ) from exc
            logger.info(
                "加载本地 LLM: %s (n_ctx=%d, n_gpu_layers=%d)",
                self.model_path.name,
                self.n_ctx,
                self.n_gpu_layers,
            )
            started = time.monotonic()
            kwargs: dict[str, Any] = {
                "model_path": str(self.model_path),
                "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
                "verbose": self.verbose,
            }
            if self.n_threads:
                kwargs["n_threads"] = int(self.n_threads)
            try:
                self._llm = Llama(**kwargs)
            except Exception as exc:  # noqa: BLE001
                raise LlmError(f"加载本地 LLM 失败({self.model_path.name}): {exc}") from exc
            logger.info("本地 LLM 就绪，耗时 %.1fs", time.monotonic() - started)
            return self._llm

    def complete(
        self,
        messages: Sequence[LlmMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: Sequence[str] = (),
        timeout: float | None = None,
    ) -> LlmResponse:
        llm = self._load()
        started = time.monotonic()
        kwargs: dict[str, Any] = {
            "messages": [m.to_dict() for m in messages],
            "max_tokens": int(max_tokens if max_tokens is not None else 2048),
            "temperature": float(temperature if temperature is not None else 0.3),
            "top_p": 0.9,
            "repeat_penalty": 1.05,
        }
        if stop:
            kwargs["stop"] = list(stop)
        try:
            response = llm.create_chat_completion(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise LlmError(f"本地 LLM 推理失败: {exc}") from exc

        text = ""
        finish_reason = ""
        usage: dict[str, Any] = {}
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                text = str((choices[0].get("message") or {}).get("content") or "")
                finish_reason = str(choices[0].get("finish_reason") or "")
            usage = response.get("usage") or {}
        return LlmResponse(
            text=text.strip(),
            backend=self.name,
            model=self.model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            elapsed_ms=int((time.monotonic() - started) * 1000),
            finish_reason=finish_reason,
        )

    def stream(
        self,
        messages: Sequence[LlmMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: Sequence[str] = (),
        timeout: float | None = None,
    ) -> Iterator[str]:
        llm = self._load()
        kwargs: dict[str, Any] = {
            "messages": [m.to_dict() for m in messages],
            "max_tokens": int(max_tokens if max_tokens is not None else 2048),
            "temperature": float(temperature if temperature is not None else 0.3),
            "stream": True,
        }
        if stop:
            kwargs["stop"] = list(stop)
        try:
            for chunk in llm.create_chat_completion(**kwargs):
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                piece = (choices[0].get("delta") or {}).get("content")
                if piece:
                    yield str(piece)
        except Exception as exc:  # noqa: BLE001
            raise LlmError(f"本地 LLM 流式推理失败: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            llm, self._llm = self._llm, None
        close = getattr(llm, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                logger.debug("关闭本地 LLM 时忽略异常", exc_info=True)


# =============================================================================
# 兜底：关键词摘要
# =============================================================================


class KeywordBackend:
    """jieba 关键词摘要 —— 永不失败的最后一道。

    不是"智能总结"，但它保证：无论模型和网络什么状态，用户点"生成总结"都能
    拿到一份包含时间区间和关键词的结果，而不是一个红色错误框。
    需求文档表 4 明确要求这一档存在。
    """

    name = "keywords"
    model = "jieba-extractive"

    #: 中文停用词（只保留区分度高的实词）
    STOPWORDS: Final[frozenset[str]] = frozenset(
        {
            "的", "了", "是", "在", "和", "也", "就", "都", "而", "及", "与", "着",
            "或", "一个", "我们", "你们", "他们", "这个", "那个", "这样", "那样",
            "什么", "怎么", "为什么", "可以", "没有", "不是", "还是", "但是",
            "因为", "所以", "如果", "然后", "现在", "这里", "那里", "自己",
        }
    )

    def __init__(self, *, top_keywords: int = 20, max_sentences: int = 6) -> None:
        self.top_keywords = max(1, int(top_keywords))
        self.max_sentences = max(1, int(max_sentences))
        self._jieba_analyse: Any | None = None
        self._jieba: Any | None = None

    def available(self) -> bool:
        return True

    # -- 分词 --

    def _ensure_jieba(self) -> None:
        if self._jieba is not None:
            return
        try:
            import jieba  # noqa: PLC0415
            import jieba.analyse  # noqa: PLC0415

            jieba.setLogLevel(logging.WARNING)
            self._jieba = jieba
            self._jieba_analyse = jieba.analyse
        except ImportError:
            logger.debug("jieba 不可用，关键词兜底退化为字符切分")
            self._jieba = False

    def extract_keywords(self, text: str, limit: int | None = None) -> list[str]:
        raw = (text or "").strip()
        if not raw:
            return []
        limit = max(1, int(limit if limit is not None else self.top_keywords))
        self._ensure_jieba()
        if self._jieba_analyse is not None:
            try:
                tags = self._jieba_analyse.extract_tags(
                    raw, topK=limit, withWeight=False, allowPOS=()
                )
                picked = [t for t in tags if t and t not in self.STOPWORDS]
                if picked:
                    return picked[:limit]
            except Exception:  # noqa: BLE001
                logger.debug("jieba.extract_tags 失败，退化为词频统计", exc_info=True)
        return self._frequency_keywords(raw, limit)

    def _frequency_keywords(self, text: str, limit: int) -> list[str]:
        import re  # noqa: PLC0415
        from collections import Counter  # noqa: PLC0415

        if self._jieba:
            tokens = [t for t in self._jieba.lcut(text) if len(t) >= 2]
        else:
            tokens = re.findall(r"[\u4e00-\u9fff]{2,4}|[A-Za-z][A-Za-z0-9_+#.-]{1,}", text)
        counted = Counter(t for t in tokens if t not in self.STOPWORDS)
        return [word for word, _ in counted.most_common(limit)]

    # -- 摘要 --

    def complete(
        self,
        messages: Sequence[LlmMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: Sequence[str] = (),
        timeout: float | None = None,
    ) -> LlmResponse:
        started = time.monotonic()
        normalized = normalize_messages(messages)
        joined = "\n".join(m.content for m in normalized if m.role in {"user", "system"})
        text = self.summarize(joined)
        return LlmResponse(
            text=text,
            backend=self.name,
            model=self.model,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            finish_reason="fallback",
            degraded=True,
        )

    def summarize(self, text: str) -> str:
        """抽取式摘要：关键词 + 首句 + 含关键词的句子。"""
        raw = (text or "").strip()
        if not raw:
            return "（无可用文本，无法生成摘要）"
        keywords = self.extract_keywords(raw)
        sentences = self._split_sentences(raw)
        picked = sentences[: self.max_sentences]
        lines = ["## 内容概览", ""]
        if keywords:
            lines.append(f"关键词：{'、'.join(keywords[: min(10, len(keywords))])}")
            lines.append("")
        if picked:
            lines.append("## 主要脉络")
            lines.append("")
            for index, sentence in enumerate(picked, start=1):
                lines.append(f"{index}. {sentence}")
            lines.append("")
        lines.append(
            f"> 本结果由关键词抽取生成（LLM 不可用时的兜底路径），未经过语义改写。"
            f"共 {len(keywords)} 个关键词、{len(sentences)} 个句子。"
        )
        return "\n".join(lines)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        import re  # noqa: PLC0415

        parts = re.split(r"(?<=[。！？!?；;\n])", text)
        out: list[str] = []
        for part in parts:
            cleaned = part.strip()
            if len(cleaned) < 8:
                continue
            out.append(cleaned if len(cleaned) <= 120 else cleaned[:117] + "...")
        return out

    def stream(
        self,
        messages: Sequence[LlmMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: Sequence[str] = (),
        timeout: float | None = None,
    ) -> Iterator[str]:
        yield self.complete(messages).text

    def close(self) -> None:
        return None


# =============================================================================
# 本地服务探测（Ollama / vLLM / LM Studio）
# =============================================================================


def detect_local_server(
    candidates: Sequence[str] = (
        "http://127.0.0.1:11434/v1",
        "http://127.0.0.1:1234/v1",
        "http://127.0.0.1:8000/v1",
    ),
    *,
    timeout: float = 0.6,
) -> str | None:
    """探测本机是否跑着 OpenAI 兼容服务（Ollama 等），返回 ``base_url``。

    只发 ``GET {base}/models`` 探活，不加载模型。命中即返回，全部失败返回 ``None``。
    """
    for base in candidates:
        url = f"{base.rstrip('/')}/models"
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status == 200:
                    logger.info("发现本地 OpenAI 兼容服务: %s", base)
                    return base
        except Exception:  # noqa: BLE001 - 探测失败属正常
            continue
    return None


def local_server_backend(base_url: str, *, timeout: float = 2.0) -> OpenAICompatibleBackend | None:
    """把探测到的本地服务变成可用后端：``GET /models`` 挑第一个模型名。

    没有模型名就无法构造请求体；列表为空或请求失败时返回 ``None``
    （该档缺席，降级链继续往下走）。
    """
    url = f"{base_url.rstrip('/')}/models"
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("读取本地服务模型列表失败 %s: %s", base_url, exc)
        return None
    for item in (data.get("data") or []) if isinstance(data, dict) else []:
        model_id = str((item or {}).get("id") or "").strip()
        if model_id:
            return OpenAICompatibleBackend(
                base_url, model_id, api_key="", timeout_s=90.0, max_retries=1
            )
    return None


# =============================================================================
# 网关
# =============================================================================


class LlmGateway:
    """按配置组装后端链，对外只暴露 ``complete`` / ``summarize`` / ``answer``。

    Parameters
    ----------
    backends:
        **有序**后端列表，前面的优先。降级即按此顺序往下走。
    mode:
        ``auto`` / ``cloud`` / ``local``，仅用于日志与结果标注。
    """

    def __init__(
        self,
        backends: Sequence[LlmBackend],
        *,
        mode: str = "auto",
        max_tokens: int = 2048,
        temperature: float = 0.3,
        timeout_s: float = 60.0,
    ) -> None:
        self._backends = list(backends)
        if not self._backends:
            self._backends = [KeywordBackend()]
        self.mode = mode
        self.max_tokens = max(1, int(max_tokens))
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()

    # -- 构造 --

    @classmethod
    def from_config(
        cls,
        config: AppConfig | None = None,
        manager: Any | None = None,
        *,
        allow_cloud: bool | None = None,
    ) -> LlmGateway:
        """按 ``llm.mode`` 组装。任何后端装配失败都跳过并记日志，最后兜底关键词。

        链序：``cloud``（有密钥）→ ``local`` GGUF（llama.cpp）→ 本地 OpenAI 兼容
        服务（Ollama / LM Studio，仅在 GGUF 档缺席时探测）→ ``keywords``。"""
        cfg = config or get_app_config()
        llm_cfg = cfg.llm
        cloud_cfg = llm_cfg.cloud
        local_cfg = llm_cfg.local

        backends: list[LlmBackend] = []
        want_cloud = llm_cfg.mode in {"auto", "cloud"}
        want_local = llm_cfg.mode in {"auto", "local"}
        if allow_cloud is False:
            want_cloud = False

        # 1) 云端
        if want_cloud:
            # 密钥来源：界面里填的（data/user_settings.json）优先，其次环境变量
            api_key = cloud_cfg.resolved_key()
            if not api_key:
                logger.info(
                    "云端 LLM 未配置密钥（llm.cloud.api_key 或环境变量 %s），跳过（mode=%s）",
                    cloud_cfg.api_key_env, llm_cfg.mode,
                )
            elif not cloud_cfg.base_url:
                logger.warning("llm.cloud.base_url 为空，跳过云端 LLM")
            else:
                backends.append(
                    OpenAICompatibleBackend(
                        cloud_cfg.base_url,
                        cloud_cfg.model,
                        api_key,
                        timeout_s=cloud_cfg.timeout_s,
                        max_retries=cloud_cfg.max_retries,
                    )
                )
                logger.info("云端 LLM 就绪: %s @ %s", cloud_cfg.model, cloud_cfg.base_url)

        # 2) 本地 GGUF
        if want_local and manager is not None:
            try:
                model_dir = Path(manager.ensure_model(local_cfg.model))
                weight = _pick_llm_weights(model_dir)
                n_gpu_layers = -1 if cfg.models.device in {"auto", "cuda"} else 0
                backends.append(
                    LlamaCppChatBackend(
                        weight,
                        local_cfg.model,
                        n_ctx=local_cfg.n_ctx,
                        n_threads=local_cfg.n_threads,
                        n_gpu_layers=n_gpu_layers,
                    )
                )
                logger.info("本地 LLM 就绪: %s (%s)", local_cfg.model, weight.name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("本地 LLM 不可用: %s", exc)

        # 2b) 本地 GGUF 没配上时探测 Ollama / LM Studio（OpenAI 兼容服务）。
        # 需求表 4 的链是 "云端 → llama.cpp → Ollama → 关键词"，这一档此前
        # 只有探测函数没有接线。只在 GGUF 档缺席时探测，避免正常路径
        # 每次装配多付 ~2 秒探活。
        if want_local and not any(getattr(b, "name", "") == "local" for b in backends):
            try:
                base = detect_local_server()
            except Exception:  # noqa: BLE001 - 探测本身不该让装配失败
                base = None
            if base:
                backend = local_server_backend(base)
                if backend is not None:
                    backends.append(backend)
                    logger.info("Ollama 档就绪: %s @ %s", backend.model, base)

        # 3) 兜底
        if llm_cfg.fallback_keywords or not backends:
            backends.append(KeywordBackend(top_keywords=llm_cfg.fallback_top_keywords))

        return cls(
            backends,
            mode=llm_cfg.mode,
            max_tokens=cloud_cfg.max_tokens,
            temperature=cloud_cfg.temperature,
            timeout_s=float(cloud_cfg.timeout_s),
        )

    # -- 属性 --

    @property
    def backend_names(self) -> list[str]:
        return [b.name for b in self._backends]

    @property
    def primary(self) -> str:
        return self._backends[0].name if self._backends else "none"

    def info(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "backends": self.backend_names,
            "primary": self.primary,
            "models": [getattr(b, "model", "") for b in self._backends],
        }

    # -- 调用 --

    def complete(
        self,
        messages: Sequence[LlmMessage | dict[str, Any] | str],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: Sequence[str] = (),
        timeout: float | None = None,
        require_quality: bool = False,
    ) -> LlmResponse:
        """按后端顺序尝试，第一个成功即返回。

        ``require_quality=True`` 时不允许落到关键词兜底（例如"必须给出一段
        连贯中文总结"的场景），此时全部失败会抛 :class:`LlmError`。
        """
        normalized = normalize_messages(messages)
        attempts: list[str] = []
        last_error = ""

        for backend in self._backends:
            name = getattr(backend, "name", backend.__class__.__name__)
            if require_quality and name == "keywords":
                continue
            try:
                if not backend.available():
                    attempts.append(f"{name}:unavailable")
                    continue
                with self._lock:
                    response = backend.complete(
                        normalized,
                        max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                        temperature=temperature if temperature is not None else self.temperature,
                        stop=stop,
                        timeout=timeout if timeout is not None else self.timeout_s,
                    )
                response.attempts = [*attempts, f"{name}:ok"]
                response.degraded = name == "keywords"
                if not response.text.strip():
                    last_error = f"{name} 返回空内容"
                    attempts.append(f"{name}:empty")
                    continue
                if attempts:
                    logger.info("LLM 降级到 %s（前序失败: %s）", name, ", ".join(attempts))
                return response
            except Exception as exc:  # noqa: BLE001 - 降级链的核心
                last_error = str(exc)
                attempts.append(f"{name}:failed({type(exc).__name__})")
                logger.warning("LLM 后端 %s 失败: %s", name, exc)

        raise LlmError(f"所有 LLM 后端均失败（{', '.join(attempts) or '无可用后端'}）: {last_error}")

    def stream(
        self,
        messages: Sequence[LlmMessage | dict[str, Any] | str],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: Sequence[str] = (),
        timeout: float | None = None,
    ) -> Iterator[str]:
        """流式输出。优先用第一个可用后端的 stream，失败则退化为整段输出。"""
        normalized = normalize_messages(messages)
        for backend in self._backends:
            name = getattr(backend, "name", "")
            if name == "keywords" or not backend.available():
                continue
            stream_fn = getattr(backend, "stream", None)
            if not callable(stream_fn):
                continue
            try:
                produced = False
                for piece in stream_fn(
                    normalized,
                    max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                    temperature=temperature if temperature is not None else self.temperature,
                    stop=stop,
                    timeout=timeout if timeout is not None else self.timeout_s,
                ):
                    produced = True
                    yield piece
                if produced:
                    return
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM 流式后端 %s 失败，尝试下一个: %s", name, exc)
        # 全部失败：用非流式兜底整段吐出
        try:
            yield self.complete(normalized).text
        except LlmError as exc:
            yield f"（生成失败：{exc}）"

    # -- 业务便捷方法 --

    def summarize(self, text: str, *, title: str = "视频内容") -> LlmResponse:
        """片段/整段摘要。用 ``clip_summarize`` 的语义，但不绑死提示词 key。"""
        prompt = (
            f"请用中文概括下面这段视频语音转录在讲什么（不超过 60 字），"
            f"再提取 3 个关键词，格式为「摘要：… / 关键词：a,b,c」。\n\n"
            f"《{title}》\n---\n{text}\n---"
        )
        return self.complete([LlmMessage("user", prompt)])

    def answer(self, question: str, context: str, *, video_title: str = "该视频") -> LlmResponse:
        """基于检索证据回答（提示词由调用方用 ``qa_answer`` 渲染时更佳）。"""
        prompt = (
            f"你是视频内容问答助手。只依据下面检索到的证据回答，"
            f"每条结论后用 [编号] 标注来源；证据不足时说明视频里没有提到。\n\n"
            f"【证据】\n{context}\n\n【问题】\n{question}\n\n"
            f"（视频：{video_title}）"
        )
        return self.complete([LlmMessage("user", prompt)])

    def extract_keywords(self, text: str, *, limit: int = 10) -> list[str]:
        """关键词提取。走关键词后端即可，不必占用 LLM。"""
        for backend in self._backends:
            fn = getattr(backend, "extract_keywords", None)
            if callable(fn):
                try:
                    result = fn(text, limit)
                    if result:
                        return list(result)
                except Exception:  # noqa: BLE001
                    logger.debug("关键词提取失败，尝试下一个后端", exc_info=True)
        return []

    # -- 生命周期 --

    def close(self) -> None:
        for backend in self._backends:
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    logger.debug("关闭 LLM 后端时忽略异常", exc_info=True)

    def __enter__(self) -> LlmGateway:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"LlmGateway(mode={self.mode}, backends={self.backend_names})"


def _pick_llm_weights(model_dir: Path) -> Path:
    """挑本地 LLM 主权重：Q4_K_M > Q5_K_M > Q4_0 > 第一个。"""
    candidates = [p for p in sorted(model_dir.glob("*.gguf")) if "mmproj" not in p.name.lower()]
    if not candidates:
        raise LlmError(f"模型目录里没有 .gguf 主权重: {model_dir}")
    for token in ("q4_k_m", "q5_k_m", "q4_0", "q8_0", "q6_k"):
        for path in candidates:
            if token in path.name.lower():
                return path
    return candidates[0]


def iter_text_pieces(text: str, *, size: int = 40) -> Iterable[str]:
    """把整段文本切成小块（给非流式后端的伪流式回放用）。"""
    step = max(1, int(size))
    for start in range(0, len(text), step):
        yield text[start : start + step]
