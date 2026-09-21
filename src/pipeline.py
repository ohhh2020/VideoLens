"""

实现流程：解析元信息 → 下载视频 → 提取音频 → 语音转写 → 抽取关键帧→ 画面理解 → 生成总结 → 保存结果

"""

from __future__ import annotations

import uuid                              # 生成任务ID
from datetime import datetime            # 生成时间戳
from pathlib import Path
from typing import Callable

from .config import settings as default_settings
from .downloader import download_video, extract_metadata
from .keyframes import extract_keyframes
from .media import extract_audio
from .models import to_serializable
from .storage import save_result
from .summarizer import build_summary, captions_to_markdown, transcript_to_markdown
from .transcriber import Transcriber
from .vlm import VLMClient

#一个汇报进度函数，被反复调用
def _safe_callback(callback: Callable[[float, str], None] | None, percent: float, message: str):
    if callback is None:
        return
    try:
        callback(min(max(percent, 0.0), 1.0), message)
    except Exception:  # noqa: BLE001 - 进度回调失败不影响主流程
        pass

#主函数，分析视频链接
def analyze_url(
    url: str,
    settings=default_settings,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict:
    settings.ensure_dirs()

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]  #生成唯一任务ID：时间戳+6位随机十六进制
    job_dir = settings.output_dir / job_id
    keyframe_dir = job_dir / "keyframes"
    job_dir.mkdir(parents=True, exist_ok=True)

    _safe_callback(progress_callback, 0.02, "正在解析视频信息")
    metadata = extract_metadata(url, settings)

    _safe_callback(progress_callback, 0.08, "正在下载视频")
    if metadata.get("is_local"):                                                   #本地或者下载
        video_path = Path(url)
    else:
        def download_hook(percent: float, message: str) -> None:
            _safe_callback(progress_callback, 0.08 + percent * 0.22, message)

        video_path = download_video(url, settings, progress_callback=download_hook)

    _safe_callback(progress_callback, 0.32, "正在提取音频")
    audio_path = None
    try:
        audio_path = extract_audio(video_path, job_dir)
    except Exception as exc:  # noqa: BLE001
        _safe_callback(progress_callback, 0.34, f"音频提取失败，继续分析：{exc}")

    _safe_callback(progress_callback, 0.38, "正在语音转写")
    transcript = []
    try:
        transcript = Transcriber(settings).transcribe(audio_path)
    except Exception as exc:  # noqa: BLE001
        _safe_callback(progress_callback, 0.52, f"语音转写失败，继续分析：{exc}")

    _safe_callback(progress_callback, 0.56, "正在抽取关键帧")
    keyframes = []
    try:
        keyframes = extract_keyframes(
            video_path,
            keyframe_dir,
            interval=settings.frame_interval,
            max_frames=settings.max_frames,
        )
    except Exception as exc:  # noqa: BLE001
        _safe_callback(progress_callback, 0.68, f"关键帧抽取失败，继续分析：{exc}")

    _safe_callback(progress_callback, 0.72, "正在进行画面理解")
    captions = []
    try:
        captions = VLMClient(settings).caption_keyframes(keyframes)
    except Exception as exc:  # noqa: BLE001
        _safe_callback(progress_callback, 0.84, f"画面理解失败，继续分析：{exc}")

    _safe_callback(progress_callback, 0.86, "正在生成总结")
    try:
        summary = build_summary(metadata, transcript, captions, settings)
    except Exception as exc:  # noqa: BLE001
        _safe_callback(progress_callback, 0.92, f"智能总结失败，使用基础总结：{exc}")
        summary = {
            "title": metadata.get("title", "未命名视频"),
            "one_sentence": metadata.get("title", ""),
            "core_points": [],
            "timeline": [],
            "keywords": [],
            "tags": ["视频总结"],
            "audience": "",
            "markdown": f"# {metadata.get('title', '未命名视频')}\n\n总结生成失败，请查看转写文本。",
        }

    _safe_callback(progress_callback, 0.94, "正在保存结果")
    result = {
        "job_id": job_id,
        "metadata": metadata,
        "video_path": str(video_path),
        "audio_path": str(audio_path) if audio_path else "",
        "transcript": transcript,
        "keyframes": keyframes,
        "captions": captions,
        "summary": summary,
        "transcript_markdown": transcript_to_markdown(transcript),
        "captions_markdown": captions_to_markdown(captions),
        "output_dir": str(job_dir),
    }
    save_result(job_dir, result)

    _safe_callback(progress_callback, 1.0, "分析完成")
    return to_serializable(result)

