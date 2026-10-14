"""完整真实流水线验收：FFmpeg、BGE-M3、SigLIP2、llama.cpp、Chroma、SQLite。

本脚本的字幕/画面是已知内容的合成样例，用于验证端到端功能；ASR 准确率
由公开真实人声样本的独立测试给出。模型和媒体处理均不使用 Mock。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from videolens.bootstrap import start_local, stop_local
from videolens.config import atomic_json, load_settings
from videolens.service import VideoService


def create_fixture(folder: Path, ffmpeg: str, seconds: int = 40) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    folder.mkdir(parents=True, exist_ok=True)
    font_file = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
    font = ImageFont.truetype(str(font_file), 48) if font_file.is_file() else ImageFont.load_default()
    panels = [("#122e51", "视频主题：语音识别模型", "第一部分：声音可以转成文字"),
              ("#194d3f", "语音识别的应用", "第二部分：自动字幕和会议记录")]
    for number, (color, heading, detail) in enumerate(panels):
        image = Image.new("RGB", (1280, 720), color)
        draw = ImageDraw.Draw(image)
        draw.text((70, 180), heading, fill="white", font=font)
        draw.text((70, 300), detail, fill="#e8e8e8", font=font)
        draw.rectangle((80, 440, 330, 580), fill="red" if number == 0 else "green")
        image.save(folder / f"panel-{number}.png")
    target = folder / "中文 & 语音识别示例.mp4"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-framerate", "1", "-t", str(seconds / 2),
               "-i", str(folder / "panel-0.png"), "-loop", "1", "-framerate", "1", "-t", str(seconds / 2), "-i", str(folder / "panel-1.png"),
               "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
               "-map", "[v]", "-map", "2:a", "-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(target)]
    subprocess.run(command, check=True, timeout=240, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    from videolens.retrieval import timestamp
    lines = []
    for index, start in enumerate(range(0, seconds, 20), 1):
        text = "欢迎体验语音识别模型，它能将声音转为文字。" if start < seconds / 2 else "语音识别可以用于自动字幕和会议记录。"
        lines.append(f"{index}\n{timestamp(start * 1000).replace('.', ',')} --> {timestamp(min(seconds, start + 20) * 1000).replace('.', ',')}\n{text}\n")
    target.with_suffix(".srt").write_text("\n".join(lines), encoding="utf-8")
    return target


def main() -> int:
    import psutil

    parser = argparse.ArgumentParser(description="验证真实本地端到端分析与引用")
    parser.add_argument("--output", type=Path, default=Path("data/verification/pipeline.json"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--seconds", type=int, default=40, choices=[40, 3600], help="40 秒功能样例，或真实生成一小时低帧率合成媒体专项")
    parser.add_argument("--work-name", default="pipeline", help="区分独立验收数据库，避免复用上次任务")
    args = parser.parse_args()
    original = load_settings()
    settings = original.model_copy(deep=True)
    if not args.work_name.replace("-", "").isalnum():
        parser.error("work-name 只能包含字母、数字和连字符")
    settings.data_dir = str(Path(f"data/verification/{args.work_name}-data").resolve())
    settings.cloud_enabled = False
    settings.allow_model_download = False
    settings.device = "cpu"
    settings.frames_per_clip = 1
    settings.output_tokens = 512
    settings.local_context_tokens = 8192
    fixture = create_fixture(Path("data/fixtures") / args.work_name, settings.ffmpeg, args.seconds)
    report = {"kind": "real_local_pipeline", "fixture": f"{args.seconds} 秒 720p、1fps、静音合成字幕视频", "cloud": "disabled", "cases": [], "model_note": "实际本地权重与llama.cpp，没有Mock；合成字幕专项不代表一小时真实人声ASR或复杂画面性能"}
    report["embedding_device"] = settings.device
    report["llama_gpu_layers"] = settings.llama_gpu_layers
    args.output.parent.mkdir(parents=True, exist_ok=True)
    peak = {"rss_bytes": 0}
    stopped = threading.Event()

    def monitor():
        while not stopped.wait(0.2):
            processes = [psutil.Process(), *psutil.Process().children(recursive=True)]
            total = 0
            for process in processes:
                try:
                    total += process.memory_info().rss
                except psutil.Error:
                    pass
            peak["rss_bytes"] = max(peak["rss_bytes"], total)

    threading.Thread(target=monitor, daemon=True).start()
    startup = time.monotonic()
    local_state = start_local(settings, owner="verify-pipeline", wait_seconds=240)
    report["llama_startup_seconds"] = round(time.monotonic() - startup, 3)
    service = VideoService(settings)
    started = time.monotonic()
    try:
        service.start()
        video = service.import_file(fixture)
        job = service.create_task(video["id"])
        last_stage = None
        while time.monotonic() - started < args.timeout:
            state = service.get_task(job["id"])
            if state["stage"] != last_stage:
                print(json.dumps({"stage": state["stage"], "status": state["status"]}, ensure_ascii=False), flush=True)
                last_stage = state["stage"]
            if state["status"] in {"succeeded", "partial", "failed", "cancelled"}:
                break
            time.sleep(0.5)
        else:
            service.control_task(job["id"], "cancel")
            raise TimeoutError("真实流水线超时，已请求取消")
        assert state["status"] in {"succeeded", "partial"}, state
        details = service.get_video(video["id"])
        index = service.store.get_meta("index:" + video["id"], {})
        assert set(index.get("collections", {})) == {"text", "image"}, "两种真实向量索引均须成功"
        assert details["report"]["mode"] == "generated", "本轮验收必须有真实本地生成，不能用摘录代替"
        report["cases"].append({"name": "analysis", "status": "passed", "job": state, "clip_count": len(details["clips"]), "frame_count": len(details["frames"]), "chapter_count": len(details["report"]["chapters"]), "frames": details["frames"], "summary": details["report"]["summary"]})
        hits = service.search(video["id"], "自动字幕和会议记录")
        assert hits and any("会议记录" in r["text"] for h in hits for r in h["refs"])
        image_hits = service.search(video["id"], image_path=service.media_path(video["id"], details["frames"][0]["id"]))
        assert image_hits
        report["cases"].append({"name": "text_and_image_search", "status": "passed", "text_hits": len(hits), "image_hits": len(image_hits)})
        session = service.create_session(video["id"])
        question_started = time.monotonic()
        first_event_seconds = None
        events = []
        for event in service.ask_stream(session["id"], "语音识别有什么应用？"):
            if first_event_seconds is None:
                first_event_seconds = time.monotonic() - question_started
            events.append(event)
        answer = events[-1]["data"]
        assert events[-1]["event"] == "final" and answer["mode"] == "generated" and answer["refs"]
        assert any(e["event"] == "delta" for e in events), "必须收到真实模型流式增量"
        report["cases"].append({"name": "streaming_question", "status": "passed", "answer": answer, "first_event_seconds": first_event_seconds, "wall_seconds": round(time.monotonic() - question_started, 3)})
        export = service.export(video["id"], "json")
        assert json.loads(export)["id"] == video["id"] and str(Path.cwd()) not in export
        args.output.with_suffix(".md").write_text(service.export(video["id"], "markdown"), encoding="utf-8")
        service.close()
        service = VideoService(settings)
        assert len(service.session_messages(session["id"])) == 2
        assert service.get_video(video["id"])["report"]
        report["cases"].append({"name": "export_and_restart", "status": "passed"})
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        service.close()
        stopped.set()
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["peak_process_tree_rss_bytes"] = peak["rss_bytes"]
        atomic_json(args.output, report)
        if local_state.get("started"):
            stop_local(settings, owner="verify-pipeline")
        print(f"真实流水线报告：{args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
