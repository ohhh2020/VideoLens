"""文本总结与问答仅使用云端 OpenAI 兼容协议。"""

from __future__ import annotations

import json
import re
import logging
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, runtime_checkable

from backend.config import get_app_config
from backend.credentials import secret_input_error

logger = logging.getLogger("zhiying.models.llm")


VALID_ROLES: Final[frozenset[str]] = frozenset({"system", "user", "assistant"})


# =============================================================================
# 异常与数据结构
# =============================================================================


class LlmError(RuntimeError):
    """LLM 调用失败（配置缺失、网络错误、后端拒绝）。"""


class LlmAuthenticationError(LlmError):
    """凭据不合法；修改配置后重建后端才可恢复。"""


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


def normalize_messages(
    messages: Sequence[LlmMessage | dict[str, Any] | str],
) -> list[LlmMessage]:
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
            out.append(
                LlmMessage(role if role in VALID_ROLES else "user", str(content))
            )
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
        self._auth_error = secret_input_error(self.api_key)
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
        if self._auth_error:
            raise LlmAuthenticationError(self._auth_error)
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
        json_mode: bool = False,
    ) -> bytes:
        body: dict[str, Any] = {
            "model": self.model,
            **({"response_format": {"type": "json_object"}} if json_mode else {}),
            "enable_thinking": False,
            "messages": [m.to_dict() for m in messages],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": bool(stream),
            **({"stream_options": {"include_usage": True}} if stream else {}),
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
        json_mode: bool = False,
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
            json_mode=json_mode,
        )
        request = urllib.request.Request(
            self.endpoint, data=payload, headers=self._headers(), method="POST"
        )
        timeout_s = float(timeout if timeout is not None else self.timeout_s)

        deadline = started + timeout_s if timeout is not None else None
        last_error = ""
        from backend.analysis.cloud import MANAGED, http_error
        retries = 0 if MANAGED.get() else self.max_retries
        for attempt in range(retries + 1):
            remaining = (
                deadline - time.monotonic() if deadline is not None else timeout_s
            )
            if remaining <= 0:
                break
            try:
                with urllib.request.urlopen(
                    request, timeout=max(0.01, remaining)
                ) as response:
                    raw = response.read()
                data = json.loads(raw.decode("utf-8"))
                return self._parse(data, started)
            except urllib.error.HTTPError as exc:
                detail = _read_error_body(exc)
                last_error = f"HTTP {exc.code}: {detail}"
                if MANAGED.get() and exc.code not in {401, 403}:
                    raise http_error(LlmError(last_error), exc) from exc
                # 4xx（除 429）重试无意义
                if exc.code in {401, 403}:
                    self._auth_error = f"云端认证失败 HTTP {exc.code}，请核对完整 API Key、服务地址和权限，保存后重启服务"
                    raise LlmAuthenticationError(self._auth_error) from exc
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
            if attempt < retries:
                delay = min(4.0, 0.8 * (2**attempt))
                logger.warning(
                    "云端 LLM 第 %d 次失败(%s)，%.1fs 后重试",
                    attempt + 1,
                    last_error,
                    delay,
                )
                if deadline is not None:
                    delay = min(delay, max(0.0, deadline - time.monotonic()))
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

    def stream_events(
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
        finished = False
        metadata = {"finish_reason": "", "prompt_tokens": 0, "completion_tokens": 0}
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        finished = True
                        break
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        metadata.update({key: int(obj["usage"].get(key) or 0) for key in ("prompt_tokens", "completion_tokens")})
                    for choice in obj.get("choices") or []:
                        if choice.get("finish_reason"):
                            metadata["finish_reason"] = str(choice["finish_reason"])
                            finished = True
                        delta = choice.get("delta") or {}
                        piece = delta.get("content")
                        if piece:
                            yield ("delta", str(piece))
                if not finished:
                    raise LlmError("云端流提前结束，未收到完成标志")
                yield ("metadata", metadata)
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                self._auth_error = f"云端认证失败 HTTP {exc.code}，请核对完整 API Key、服务地址和权限，保存后重启服务"
                raise LlmAuthenticationError(self._auth_error) from exc
            from backend.analysis.cloud import http_error
            raise http_error(LlmError(f"云端流式请求失败 HTTP {exc.code}: {_read_error_body(exc)}"), exc) from exc
        except urllib.error.URLError as exc:
            raise LlmError(f"云端流式请求网络错误: {exc.reason}") from exc

    def stream(self, messages, **kwargs):
        for kind, value in self.stream_events(messages, **kwargs):
            if kind == 'delta':
                yield value


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
# =============================================================================


# =============================================================================
# 兜底：关键词摘要
# =============================================================================


# =============================================================================
# =============================================================================


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
        backends,
        *,
        mode="cloud",
        max_tokens=2048,
        temperature=0.3,
        timeout_s=60.0,
        max_concurrency=4,
    ):
        self._backends = list(backends)
        if len(self._backends) != 1:
            raise LlmError("必须配置一个云端文本模型，不允许本地或关键词降级")
        self.mode = "cloud"
        self.max_tokens, self.temperature, self.timeout_s = (
            max_tokens,
            temperature,
            timeout_s,
        )
        self._slots = threading.BoundedSemaphore(max(1, max_concurrency))

    # -- 构造 --

    @classmethod
    def from_config(cls, config=None):
        cfg = (config or get_app_config()).llm.cloud
        key = cfg.resolved_key()
        if not key or secret_input_error(key):
            raise LlmAuthenticationError("请配置完整云端文本模型 API Key")
        backend = OpenAICompatibleBackend(
            cfg.base_url,
            cfg.model,
            key,
            timeout_s=cfg.timeout_s,
            max_retries=cfg.max_retries,
        )
        gateway = cls(
            [backend],
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            timeout_s=cfg.timeout_s,
            max_concurrency=cfg.max_concurrency,
        )
        gateway.policy = (config or get_app_config()).integrity
        return gateway

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

    def complete(self, messages, **kwargs):
        from backend.analysis.cloud import MANAGED, request
        from backend.analysis.policy import IntegritySection
        interactive = kwargs.pop("interactive", False)
        if MANAGED.get():
            return self._complete(messages, **kwargs)
        result = request(None, getattr(self, 'policy', IntegritySection()), '文本生成', {},
                         lambda: self._complete(messages, **kwargs), interactive=interactive)
        return LlmResponse(**{name: getattr(result, name) for name in LlmResponse.__dataclass_fields__ if hasattr(result, name)})

    def _complete(
        self,
        messages,
        *,
        max_tokens=None,
        temperature=None,
        stop=(),
        timeout=None,
        require_quality=False,
        json_mode=False,
    ):
        """直接调用唯一的云端模型，失败即抛出，不使用替代内容。"""
        backend = self._backends[0]
        if not backend.available():
            raise LlmError("云端文本模型配置不可用")
        with self._slots:
            result = backend.complete(
                normalize_messages(messages),
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                temperature=self.temperature if temperature is None else temperature,
                stop=stop,
                **({"json_mode": True} if json_mode else {}),
                timeout=self.timeout_s if timeout is None else timeout,
            )
        if not result.text.strip():
            raise LlmError("云端文本模型返回空内容")
        result.backend, result.degraded, result.attempts = "cloud", False, ["cloud:ok"]
        return result

    def stream_events(self, messages, *, max_tokens=None, temperature=None, stop=(), timeout=None):
        from backend.analysis.cloud import SCHEDULER, MANAGED
        from backend.analysis.policy import IntegritySection
        backend = self._backends[0]
        if not backend.available():
            raise LlmError('云端文本模型配置不可用')
        with SCHEDULER.slot(getattr(self, 'policy', IntegritySection()), interactive=True), self._slots:
            token = MANAGED.set(True)
            try:
                kwargs = dict(max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                    temperature=self.temperature if temperature is None else temperature, stop=stop,
                    timeout=self.timeout_s if timeout is None else timeout)
                if callable(getattr(backend, 'stream_events', None)):
                    yield from backend.stream_events(normalize_messages(messages), **kwargs)
                else:
                    for piece in backend.stream(normalize_messages(messages), **kwargs):
                        yield 'delta', piece
                    yield 'metadata', {'finish_reason': '', 'prompt_tokens': 0, 'completion_tokens': 0}
            finally:
                MANAGED.reset(token)

    def stream(self, messages, **kwargs):
        for kind, value in self.stream_events(messages, **kwargs):
            if kind == 'delta':
                yield value

    # -- 业务便捷方法 --

    def summarize(self, text: str, *, title: str = "视频内容") -> LlmResponse:
        """片段/整段摘要。用 ``clip_summarize`` 的语义，但不绑死提示词 key。"""
        prompt = (
            f"请用中文概括下面这段视频语音转录在讲什么（不超过 60 字），"
            f"再提取 3 个关键词，格式为「摘要：… / 关键词：a,b,c」。\n\n"
            f"《{title}》\n---\n{text}\n---"
        )
        return self.complete([LlmMessage("user", prompt)])

    def answer(
        self, question: str, context: str, *, video_title: str = "该视频"
    ) -> LlmResponse:
        """基于检索证据回答（提示词由调用方用 ``qa_answer`` 渲染时更佳）。"""
        prompt = (
            f"你是视频内容问答助手。只依据下面检索到的证据回答，"
            f"每条结论后用 [编号] 标注来源；证据不足时说明视频里没有提到。\n\n"
            f"【证据】\n{context}\n\n【问题】\n{question}\n\n"
            f"（视频：{video_title}）"
        )
        return self.complete([LlmMessage("user", prompt)])

    def extract_keywords(self, text, *, limit=10):
        response = self.complete(
            [
                LlmMessage(
                    "user", f"从下面文字提取最多{limit}个关键词，以逗号分隔：\n{text}"
                )
            ]
        )
        return [x.strip() for x in re.split(r"[,，、\n]", response.text) if x.strip()][
            :limit
        ]

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


def iter_text_pieces(text: str, *, size: int = 40) -> Iterable[str]:
    """把整段文本切成小块（给非流式后端的伪流式回放用）。"""
    step = max(1, int(size))
    for start in range(0, len(text), step):
        yield text[start : start + step]
