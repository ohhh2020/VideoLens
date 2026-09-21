from __future__ import annotations

from pathlib import Path

import gradio as gr

from src.pipeline import analyze_url


def run_analysis(url: str, progress=gr.Progress()):
    if not url or not url.strip():
        return "请输入视频链接", None, "无结果", None, "无结果", "无结果", None, None

    def callback(percent: float, message: str) -> None:
        progress(percent, desc=message)

    progress(0.01, desc="准备分析")
    result = analyze_url(url, progress_callback=callback)

    summary = result.get("summary", {})
    keyframe_paths = [item["path"] for item in result.get("keyframes", []) if Path(item["path"]).exists()]
    timeline = summary.get("timeline", [])
    video_path = result.get("video_path") or None

    return (
        "✅ 分析完成",
        video_path,
        summary.get("markdown", "无总结"),
        timeline,
        result.get("transcript_markdown", "无转写"),
        result.get("captions_markdown", "无画面描述"),
        keyframe_paths,
        result,
    )


def build_app():
    with gr.Blocks(title="VideoLens 视频链接智能总结系统") as demo:
        gr.Markdown(
            "# VideoLens 视频链接智能总结系统\n"
            "输入视频链接，自动完成下载、语音转写、关键帧理解和结构化总结。"
        )

        with gr.Row():
            url_input = gr.Textbox(
                label="视频链接",
                placeholder="粘贴 Bilibili / YouTube / MP4 直链，或本地 MP4 绝对路径",
                scale=5,
            )
            analyze_button = gr.Button("开始分析", variant="primary", scale=1)

        status = gr.Markdown("等待输入")

        with gr.Tabs():
            with gr.Tab("总结报告"):
                video_output = gr.Video(label="原始视频")
                summary_output = gr.Markdown("暂无结果")
            with gr.Tab("时间线"):
                timeline_output = gr.JSON(label="结构化时间线")
            with gr.Tab("语音转写"):
                transcript_output = gr.Markdown("暂无结果")
            with gr.Tab("画面理解"):
                gallery_output = gr.Gallery(label="关键帧", columns=4, height="auto")
                captions_output = gr.Markdown("暂无结果")
            with gr.Tab("完整数据"):
                result_output = gr.JSON(label="result.json")

        outputs = [
            status,
            video_output,
            summary_output,
            timeline_output,
            transcript_output,
            captions_output,
            gallery_output,
            result_output,
        ]
        analyze_button.click(
            fn=run_analysis,
            inputs=[url_input],
            outputs=outputs,
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860)

