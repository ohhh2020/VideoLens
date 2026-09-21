from __future__ import annotations

import json

from .media import format_ts
from .vlm import VLMClient


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "……（已截断）"


def _transcript_text(transcript: list) -> str:
    lines = []
    for item in transcript:
        lines.append(f"[{format_ts(item.start)}-{format_ts(item.end)}] {item.text}")
    return "\n".join(lines)


def _captions_text(captions: list) -> str:
    lines = []
    for item in captions:
        objects = "、".join(item.objects or [])
        action = item.action or ""
        extra = f"，对象：{objects}" if objects else ""
        extra += f"，动作：{action}" if action else ""
        lines.append(f"[{format_ts(item.timestamp)}] {item.description}{extra}")
    return "\n".join(lines)


def _llm_summary(metadata: dict, transcript: list, captions: list, settings) -> dict | None:
    client = VLMClient(settings)
    transcript_text = _truncate(_transcript_text(transcript), 12000)
    captions_text = _truncate(_captions_text(captions), 4000)
    if not transcript_text and not captions_text:
        return None

    prompt = f"""
你是一名专业视频内容分析师。请根据以下信息总结视频，并只输出一个 JSON 对象。

视频标题：{metadata.get("title", "未命名视频")}
视频时长：{format_ts(metadata.get("duration") or 0)}

语音转写：
{transcript_text or "（无语音转写）"}

关键帧画面描述：
{captions_text or "（无画面描述）"}

JSON 字段要求：
{{
  "one_sentence": "一句话总结",
  "core_points": ["核心要点1", "核心要点2"],
  "timeline": [{{"start": 0, "end": 60, "title": "章节标题", "content": "该段内容"}}],
  "keywords": ["关键词1", "关键词2"],
  "tags": ["标签1", "标签2"],
  "audience": "适合人群",
  "markdown": "完整 Markdown 总结，包含：一句话总结、核心内容、时间线、关键结论、关键词与标签"
}}

请严格输出 JSON，不要输出任何多余说明。
""".strip()

    return client.summarize(prompt)


def _rule_summary(metadata: dict, transcript: list, captions: list) -> dict:
    transcript_lines = _transcript_text(transcript).splitlines()
    meaningful = [line for line in transcript_lines if len(line.strip()) > 4]
    one_sentence = meaningful[0] if meaningful else metadata.get("title", "未命名视频")

    sorted_segments = sorted(transcript, key=lambda item: len(item.text), reverse=True)
    seen = set()
    core_points = []
    for item in sorted_segments:
        text = item.text.strip()
        if text not in seen:
            seen.add(text)
            core_points.append(f"[{format_ts(item.start)}] {text}")
        if len(core_points) >= 5:
            break

    timeline = []
    group_seconds = 60
    grouped: dict[int, list] = {}
    for item in transcript:
        bucket = int(item.start // group_seconds) * group_seconds
        grouped.setdefault(bucket, []).append(item)
    for bucket in sorted(grouped.keys())[:16]:
        first = grouped[bucket][0]
        content = "；".join(item.text for item in grouped[bucket][:3])
        timeline.append(
            {
                "start": bucket,
                "end": min(bucket + group_seconds, round(metadata.get("duration") or 0, 2)),
                "title": first.text[:30],
                "content": content[:120],
            }
        )

    keywords = []
    try:
        import jieba.analyse

        keywords = jieba.analyse.extract_tags(
            " ".join(item.text for item in transcript), topK=8
        )
    except Exception:  # noqa: BLE001
        keywords = []

    tags = ["视频总结"]
    if metadata.get("duration", 0) and metadata["duration"] > 1800:
        tags.append("长视频")
    else:
        tags.append("短视频")
    if captions:
        tags.append("含画面分析")

    markdown = _build_markdown(
        metadata=metadata,
        one_sentence=one_sentence,
        core_points=core_points,
        timeline=timeline,
        keywords=keywords,
        tags=tags,
        audience="对该主题感兴趣的用户",
    )

    return {
        "one_sentence": one_sentence,
        "core_points": core_points,
        "timeline": timeline,
        "keywords": keywords,
        "tags": tags,
        "audience": "对该主题感兴趣的用户",
        "markdown": markdown,
    }


def _build_markdown(
    metadata: dict,
    one_sentence: str,
    core_points: list[str],
    timeline: list[dict],
    keywords: list[str],
    tags: list[str],
    audience: str,
) -> str:
    title = metadata.get("title", "未命名视频")
    lines = [
        f"# {title} 视频总结",
        "",
        "## 一句话总结",
        one_sentence,
        "",
        "## 核心内容",
    ]
    lines.extend(f"- {point}" for point in core_points)
    lines.append("")
    lines.append("## 时间线")
    for item in timeline:
        start = format_ts(item.get("start", 0))
        end = format_ts(item.get("end", 0))
        lines.append(f"- **{start}-{end}｜{item.get('title', '')}**：{item.get('content', '')}")
    lines.extend(["", "## 关键词与标签", ""])
    if keywords:
        lines.append("关键词：" + "、".join(keywords))
    if tags:
        lines.append("标签：" + "、".join(tags))
    lines.extend(["", "## 适合人群", audience, ""])
    return "\n".join(lines)


def build_summary(metadata: dict, transcript: list, captions: list, settings) -> dict:
    result = _llm_summary(metadata, transcript, captions, settings)
    if result is None or not isinstance(result, dict):
        result = _rule_summary(metadata, transcript, captions)

    result.setdefault("title", metadata.get("title", "未命名视频"))
    result.setdefault("one_sentence", "")
    result.setdefault("core_points", [])
    result.setdefault("timeline", [])
    result.setdefault("keywords", [])
    result.setdefault("tags", [])
    result.setdefault("audience", "")

    if not result.get("markdown"):
        result["markdown"] = _build_markdown(
            metadata=metadata,
            one_sentence=result["one_sentence"],
            core_points=result["core_points"],
            timeline=result["timeline"],
            keywords=result["keywords"],
            tags=result["tags"],
            audience=result["audience"],
        )
    return result


def transcript_to_markdown(transcript: list) -> str:
    lines = ["# 视频转写", ""]
    for item in transcript:
        lines.append(f"- **{format_ts(item.start)}-{format_ts(item.end)}** {item.text}")
    return "\n".join(lines)


def captions_to_markdown(captions: list) -> str:
    if not captions:
        return "未配置视觉理解 API，暂无画面描述。"
    lines = ["# 画面描述", ""]
    for item in captions:
        lines.append(
            f"- **{format_ts(item.timestamp)}** {item.description}"
            + (f"（对象：{'、'.join(item.objects or [])}）" if item.objects else "")
            + (f"（动作：{item.action}）" if item.action else "")
        )
    return "\n".join(lines)

