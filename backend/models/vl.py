"""视觉理解仅走云端多图/视频协议，本地仅做媒体预处理。"""
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

from backend.config import AppConfig, get_app_config
from backend.credentials import secret_input_error
from backend.prompts import PromptRegistry, get_prompt_registry

logger = logging.getLogger("zhiying.models.vl")


DEFAULT_VL_PROMPT_KEY: Final[str] = "keyframe_describe"
#: 连续画面（多帧一次推理）用的提示词键。
SEQUENCE_PROMPT_KEY: Final[str] = "keyframe_describe_sequence"
#: 视频段理解（云端整段输入）用的提示词键。
SEGMENT_PROMPT_KEY: Final[str] = "video_segment_describe"

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



# =============================================================================
# 异常与结果
# =============================================================================


class VlError(RuntimeError):
    """视觉理解失败（加载、推理、输入非法）。"""


class VlAuthenticationError(VlError):
    """认证失败，不重试其它帧或视频段。"""


class VlModelNotFoundError(VlError):
    """服务中不存在指定模型，需修改模型 ID。"""


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
        raise VlError("视觉理解需要 Pillow：python -m pip install -r requirements.txt") from exc

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
            if path.suffix.lower() == ".png" and max_side == 0:
                img.save(buf, format="PNG")
            else:
                img.save(buf, format="JPEG", quality=quality, optimize=True)
    except VlError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise VlError(f"读取关键帧失败({path}): {exc}") from exc

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "png" if path.suffix.lower() == ".png" and max_side == 0 else "jpeg"
    return f"data:image/{mime};base64,{b64}"


# =============================================================================
# =============================================================================








class VisionText(str):
    def __new__(cls, text, data=None):
        obj = super().__new__(cls, text)
        data = data or {}
        obj.finish_reason = str((data.get('choices') or [{}])[0].get('finish_reason') or '')
        usage = data.get('usage') or {}
        obj.prompt_tokens = int(usage.get('prompt_tokens') or 0)
        obj.completion_tokens = int(usage.get('completion_tokens') or 0)
        return obj


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

    由 VlService 统一调度多帧请求。
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
        self._auth_error = secret_input_error(self.api_key)
        self._model_error = ""
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
        if self._auth_error:
            raise VlAuthenticationError(self._auth_error)
        if self._model_error:
            raise VlModelNotFoundError(self._model_error)
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
        native: bool = False,
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
                        "url": _image_to_data_uri(path, max_side=0 if native else self.max_image_side, quality=92)
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        body = {
            "model": self.model,
            "enable_thinking": False,
            "messages": [{"role": "user", "content": content}],
            **({"response_format": {"type": "json_object"}} if native else {}),
            "max_tokens": max(1, int(max_tokens)),
            "temperature": float(temperature),
            "stream": False,
        }
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=payload, headers=self._headers(), method="POST"
        )

        last_error = ""
        from backend.analysis.cloud import MANAGED, http_error
        retries = 0 if MANAGED.get() else self.max_retries
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    raw = response.read()
                data = json.loads(raw.decode("utf-8"))
                return self._parse(data)
            except urllib.error.HTTPError as exc:
                detail = _read_error_body(exc)
                last_error = f"HTTP {exc.code}: {detail}"
                if MANAGED.get() and exc.code not in {401, 403, 404}:
                    raise http_error(VlError(last_error), exc) from exc
                # 4xx（除 429 限流）重试没有意义，直接抛
                if exc.code == 404:
                    self._model_error = f"视觉模型不存在 HTTP 404：{self.model}；请从当前服务的模型清单选择可用模型并重启服务"
                    raise VlModelNotFoundError(self._model_error) from exc
                if exc.code in {401, 403}:
                    self._auth_error = f"云端视觉认证失败 HTTP {exc.code}，请核对完整 API Key、服务地址和权限，保存后重启服务"
                    raise VlAuthenticationError(self._auth_error) from exc
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
            if attempt < retries:
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
        return VisionText(text, data)

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
# 云端视频段理解（DashScope qwen-vl 直接吃短视频段）
# =============================================================================


class QwenVideoRunner:
    """把一个时间窗的**短视频段**直接发给 qwen-vl（`video_url`，base64）。

    与逐帧图片模式（:class:`OpenAiVisionRunner`）的本质差别不是省钱省时，
    是**语义**：两张静帧区分不出"一次进攻"和"另一次进攻"，连续画面才行。
    设计取舍见 docs/演进计划：分段而非整段 —— 上传体积可控（480p 重编码
    后 1~5MB）、段边界即场景边界（天然对齐引用时间戳）、单段失败只损一段。

    仅 DashScope 系（及任何接受 ``video_url`` 的 OpenAI 兼容端点）可用；
    其它服务没有该扩展字段，VlService 会探测并自动落回逐帧模式。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        *,
        timeout_s: float = 120.0,
        max_retries: int = 1,
        segment_max_s: float = 15.0,
        max_segment_mb: float = 5.0,
        name: str = "",
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or ""
        self.api_key = api_key or ""
        self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))
        self._encoded_local = threading.local()
        self.segment_max_s = max(5.0, float(segment_max_s))
        self.max_segment_bytes = int(max_segment_mb * 1024 * 1024)
        self.name = name or f"cloud-video:{self.model}"
        self._auth_error = secret_input_error(self.api_key)

    # -- 协议 ------------------------------------------------------------ #

    @property
    def endpoint(self) -> str:
        base = self.base_url
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    def available(self) -> bool:
        return bool(self.base_url) and bool(self.model)

    # -- 段编码 ---------------------------------------------------------- #

    def _ffmpeg(self) -> str:
        from backend.media.ffmpeg_utils import resolve_binary  # noqa: PLC0415

        return resolve_binary("ffmpeg")

    def encode_segment(self, video_path: Path, start_ms: int, end_ms: int, dest: Path) -> Path:
        """剪出 ``[start_ms, end_ms)`` 的 mp4 小段，控制在体积上限内。

        使用精确裁剪重编码，并用FFprobe验证输出时长。音轨另由ASR处理。
        """
        import subprocess  # noqa: PLC0415

        start = max(0.0, int(start_ms) / 1000.0)
        duration = (int(end_ms) - int(start_ms)) / 1000.0
        if duration <= 0:
            raise VlError("视频段起止时间无效")
        if duration > self.segment_max_s + 0.001:
            raise VlError("视频段超过单窗上限，必须先按时间拆分")
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = str(video_path)
        cmd_reencode = [
            self._ffmpeg(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", src,
            "-vf", "scale='if(gt(iw,ih),min(1280,iw),-2)':'if(gt(iw,ih),-2,min(1280,ih))'",
            "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-an",
            str(dest),
        ]
        proc = subprocess.run(  # noqa: S603
            cmd_reencode, capture_output=True, timeout=180, check=False
        )
        if proc.returncode != 0 or not dest.is_file():
            detail = (proc.stderr or b"").decode("utf-8", errors="replace")[-200:]
            raise VlError(f"剪视频段失败({start_ms}-{end_ms}ms): {detail or '未知'}")
        if dest.stat().st_size > self.max_segment_bytes:
            raise VlError(
                f"视频段 {dest.name} 重编码后仍 {dest.stat().st_size // 1024}KB"
                f"（上限 {self.max_segment_bytes // 1024}KB），请调小 vl.video_segment_max_s"
            )
        from backend.media.ffmpeg_utils import probe_media
        actual = probe_media(dest)
        if not actual.has_video or abs(actual.duration_ms - duration * 1000) > 120:
            raise VlError('视频段实际编码范围与请求不符，禁止生成误导引用')
        self._encoded_local.duration_ms = actual.duration_ms
        return dest

    # -- 推理 ------------------------------------------------------------ #

    def describe_segment(self, video_path: Path, start_ms: int, end_ms: int,
                         prompt: str) -> str:
        if self._auth_error:
            raise VlAuthenticationError(self._auth_error)
        if not self.available():
            raise VlError("云端视频段后端未配置 base_url 或 model")
        import base64 as _b64  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="zhiying-seg-") as tmp:
            seg = Path(tmp) / "segment.mp4"
            self.encode_segment(video_path, start_ms, end_ms, seg)
            encoded_duration = getattr(self._encoded_local, "duration_ms", end_ms - start_ms)
            data_uri = "data:video/mp4;base64," + _b64.b64encode(
                seg.read_bytes()
            ).decode("ascii")

        body = {
            "model": self.model,
            "enable_thinking": False,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": 768,
            "temperature": 0.2,
            "stream": False,
        }
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "zhiying/0.1 (+local-first video assistant)",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint, data=payload, headers=headers, method="POST"
        )
        last_error = ""
        from backend.analysis.cloud import MANAGED, http_error
        retries = 0 if MANAGED.get() else self.max_retries
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    raw = response.read()
                data = json.loads(raw.decode("utf-8"))
                text = _extract_content(data)
                if not text:
                    raise VlError("云端视频段返回空内容")
                result = VisionText(text, data)
                result.actual_start_ms = start_ms
                result.actual_end_ms = start_ms + encoded_duration
                result.encoded_duration_ms = encoded_duration
                return result
            except urllib.error.HTTPError as exc:
                detail = _read_error_body(exc)
                last_error = f"HTTP {exc.code}: {detail}"
                if MANAGED.get() and exc.code not in {401, 403, 404}:
                    raise http_error(VlError(last_error), exc) from exc
                if 400 <= exc.code < 500 and exc.code != 429:
                    if exc.code in {401, 403}:
                        self._auth_error = f"云端视频段认证失败 HTTP {exc.code}，请重新填写完整 API Key 并重启服务"
                        raise VlAuthenticationError(self._auth_error) from exc
                    break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(4.0, 0.8 * (2 ** attempt)))
        raise VlError(f"云端视频段调用失败({self.model}): {last_error}")


# =============================================================================
# 降级实现
# =============================================================================




# =============================================================================
# 服务
# =============================================================================




class VlService:
    """视觉理解门面：重试、拆分描述/OCR、降级。

    Parameters
    ----------
    runner:
        云端视觉请求适配器。
    max_retries:
        单帧重试次数（需求文档要求 2 次）。
    """

    def __init__(
        self,
        runner: Any | None,
        *,
        segment_runner: "QwenVideoRunner | None" = None,
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
        # P1 视频段通道：云端 qwen-vl 的 video_url 扩展。仅云端构造时注入；
        self._segment_runner = segment_runner
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
        self._terminal_error = ""
        self._terminal_error_lock = threading.Lock()

    def _record_auth_failure(self, exc: Exception) -> bool:
        if not isinstance(exc, (VlAuthenticationError, VlModelNotFoundError)):
            return False
        with self._terminal_error_lock:
            if not self._terminal_error:
                self._terminal_error = str(exc)
                self._degraded_reason = self._terminal_error
                logger.warning("视觉通道暂停：%s", self._terminal_error)
        return True

    # -- 构造 --

    @classmethod
    def _try_cloud(
        cls,
        cfg: AppConfig,
        vl_cfg: Any,
        *,
        prompts: PromptRegistry | None = None,
    ) -> VlService | None:
        """尝试装配云端后端；配不全时返回 ``None`` 明确报告配置错误。"""
        base_url = (vl_cfg.cloud_base_url or "").strip()
        model = (vl_cfg.cloud_model or "").strip()
        if not base_url or not model:
            logger.warning(
                "已开启云端视觉但缺少参数（base_url=%r model=%r），无法装配云端视觉",
                base_url, model,
            )
            return None

        api_key = (vl_cfg.cloud_api_key or "").strip()
        if not api_key and vl_cfg.cloud_api_key_env:
            api_key = (os.environ.get(vl_cfg.cloud_api_key_env) or "").strip()
        key_error = secret_input_error(api_key)
        if key_error:
            logger.warning("云端视觉密钥不可用：%s", key_error)
            return cls(None, model_name=f"cloud:{model}", degraded_reason=key_error)
        if not api_key:
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
            logger.warning("云端视觉后端构造失败，无法装配云端视觉: %s", exc)
            return None

        frames = vl_cfg.cloud_sequence_frames
        # P1 视频段通道：video_mode != frames 时同时装配 QwenVideoRunner，
        # 能力探测放到首次调用（video_url 是厂商扩展，不假设端点支持）。
        segment_runner: QwenVideoRunner | None = None
        if str(getattr(vl_cfg, "video_mode", "auto") or "auto") != "frames":
            try:
                segment_runner = QwenVideoRunner(
                    base_url,
                    model,
                    api_key,
                    timeout_s=max(120.0, float(vl_cfg.cloud_timeout_s) * 2),
                    max_retries=int(vl_cfg.cloud_max_retries),
                    segment_max_s=float(getattr(vl_cfg, "video_segment_max_s", 15)),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("视频段 Runner 构造失败，仅用逐帧模式: %s", exc)
        logger.info(
            "视觉理解云端配置已加载（尚未验证调用）: %s (%s, 每轮最多 %d 帧%s)", model, base_url, frames,
            ", 视频段通道就绪" if segment_runner is not None else "",
        )
        return cls(
            runner,
            segment_runner=segment_runner,
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
    def from_config(cls, config=None, *, prompts=None):
        cfg = config or get_app_config()
        v = cfg.vl
        key = v.cloud_api_key or os.environ.get(v.cloud_api_key_env, "")
        if not key or secret_input_error(key):
            raise VlAuthenticationError("请配置完整云端视觉模型 API Key")
        cloud = cls._try_cloud(cfg, v, prompts=prompts)
        if cloud is None or not cloud.available:
            raise VlError("云端视觉参数不完整，无法装配")
        return cloud

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

    @property
    def is_cloud(self) -> bool:
        """云端后端（网络型）：并发安全、无需本地推理锁（D5 用）。"""
        return str(getattr(self._runner, "name", "")).startswith("cloud")

    @property
    def supports_video_segments(self) -> bool:
        """视频段通道是否就绪（仅云端 qwen-vl 系）。"""
        runner = self._segment_runner
        if runner is None:
            return False
        try:
            return bool(runner.available())
        except Exception:  # noqa: BLE001
            return False

    def info(self) -> dict[str, Any]:
        return {
            "model": self._model_name,
            "available": self.available,
            "prompt_key": self._prompt_key,
            "max_retries": self._max_retries,
            "degraded_reason": self._degraded_reason,
            "is_cloud": self.is_cloud,
            "video_segments": self.supports_video_segments,
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

        if self._runner is None or self._terminal_error:
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
                if self._record_auth_failure(exc):
                    break
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

        if self._runner is None or self._terminal_error:
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
                if self._record_auth_failure(exc):
                    break
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

    def describe_video_segment(
        self,
        video_path: str | Path,
        start_ms: int,
        end_ms: int,
        *,
        prompt: str | None = None,
        prompt_key: str = SEGMENT_PROMPT_KEY,
        extra_vars: dict[str, Any] | None = None,
    ) -> VlResult:
        """P1：把 ``[start_ms, end_ms)`` 的短视频段整段发给云端视频模型。

        **永远返回 VlResult**（失败 ``degraded=True``），由编排器决定该窗
        是否回退抽帧描述。这是段通道的唯一入口，本地 GGUF 没有此能力。
        """
        started = time.monotonic()
        path = Path(video_path)
        runner = self._segment_runner
        if runner is None or not self.supports_video_segments or self._terminal_error:
            return VlResult(
                image_path=path, model=self._model_name, prompt_key=prompt_key,
                degraded=True, error=self._terminal_error or "视频段通道未装配（需云端 qwen-vl 系）",
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
        if not path.is_file():
            return VlResult(
                image_path=path, model=self._model_name, prompt_key=prompt_key,
                degraded=True, error=f"视频文件不存在: {path}",
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )

        lo, hi = max(0, int(start_ms)), max(int(start_ms) + 500, int(end_ms))
        if prompt is None:
            variables = {"start_time": _ms_label(lo), "end_time": _ms_label(hi)}
            variables.update(extra_vars or {})
            prompt = self._render(prompt_key, variables)

        last_error = ""
        attempts = 0
        for attempt in range(1, self._max_retries + 2):
            attempts = attempt
            try:
                raw = runner.describe_segment(path, lo, hi, prompt)
            except Exception as exc:  # noqa: BLE001 - 单段失败只回退这一段
                last_error = str(exc)
                if self._record_auth_failure(exc):
                    break
                logger.warning(
                    "视频段理解第 %d 次失败(%s-%s): %s", attempt,
                    _ms_label(lo), _ms_label(hi), exc,
                )
                if attempt <= self._max_retries:
                    time.sleep(min(1.5, 0.3 * attempt))
                continue
            description, ocr = split_description(raw)
            if not description and not ocr:
                last_error = "模型返回空内容"
                continue
            return VlResult(
                image_path=path,
                description=description,
                ocr_text=ocr,
                raw=raw,
                model=self._model_name,
                prompt_key=prompt_key,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                attempts=attempts,
                extra={"segment_ms": [lo, hi]},
            )
        return VlResult(
            image_path=path, model=self._model_name, prompt_key=prompt_key,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            attempts=attempts, degraded=True,
            error=last_error or "视频段理解失败",
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
