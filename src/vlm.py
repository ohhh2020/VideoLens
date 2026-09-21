from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from .models import SceneCaption


class VLMError(RuntimeError):
    pass


def _image_data_url(path: Path) -> str:
    data = Path(path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    suffix = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        suffix, "image/jpeg"
    )
    return f"data:{mime};base64,{encoded}"


def _extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None


class VLMClient:
    def __init__(self, settings) -> None:
        self.settings = settings
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.settings.openai_api_key)

    def _get_client(self):
        if self._client is None:
            if not self.available:
                raise VLMError("未配置 OPENAI_API_KEY，无法调用大模型。")
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
        return self._client

    def caption_keyframes(self, keyframes: list) -> list[SceneCaption]:
        if not self.available or not keyframes:
            return []

        client = self._get_client()
        captions: list[SceneCaption] = []
        batch_size = 4

        for index in range(0, len(keyframes), batch_size):
            batch = keyframes[index : index + batch_size]
            content: list[dict] = [
                {
                    "type": "text",
                    "text": (
                        "你是视频画面分析助手。请对下面按时间顺序排列的视频关键帧分别进行分析，"
                        "只输出 JSON 对象，格式为："
                        '{"frames":[{"timestamp":时间秒,"description":"画面描述",'
                        '"objects":["主要对象"],"action":"动作或事件"}]}。'
                        "不要输出多余文字。各帧时间戳依次为："
                        + json.dumps([round(item.timestamp, 2) for item in batch], ensure_ascii=False)
                    ),
                }
            ]
            for item in batch:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(Path(item.path))},
                    }
                )

            try:
                response = client.chat.completions.create(
                    model=self.settings.vlm_model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.2,
                )
                raw = response.choices[0].message.content or ""
                parsed = _extract_json(raw)
            except Exception:  # noqa: BLE001 - 单个批次失败不影响整体
                continue

            if not parsed:
                continue

            frames = parsed.get("frames") if isinstance(parsed, dict) else parsed
            if not isinstance(frames, list):
                continue

            for frame in frames[: len(batch)]:
                if not isinstance(frame, dict):
                    continue
                captions.append(
                    SceneCaption(
                        timestamp=float(frame.get("timestamp", 0) or 0),
                        description=str(frame.get("description") or ""),
                        objects=frame.get("objects") or [],
                        action=str(frame.get("action") or ""),
                    )
                )

        return captions

    def summarize(self, prompt: str) -> dict | None:
        if not self.available:
            return None

        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            raw = response.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            return None

        return _extract_json(raw)

