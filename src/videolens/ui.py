"""Gradio 网页：所有操作通过同一套 HTTP API，令牌只停留在 Python 后端。

页面中的 HTML 仅使用程序生成的结构，并对标题、证据、ID 逐项转义。
模型输出放入启用 sanitize_html 的 Markdown/Chatbot，不能直接变成网页脚本。
"""

from __future__ import annotations

import html
import inspect
import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

import httpx


def iter_sse(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """按空行组装 SSE，支持跨网络块、多行 data 和服务器心跳。"""
    event, data = "message", []
    for line in lines:
        if not line:
            if data:
                text = "\n".join(data)
                try:
                    value = json.loads(text)
                except json.JSONDecodeError:
                    value = {"text": text}
                yield {"event": event, "data": value}
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        # 注释心跳与 id 不参与回答正文；未完成的最后一个事件不会冒充完整结果。


def _timestamp(milliseconds: int | float) -> str:
    value = max(0, int(milliseconds))
    seconds, ms = divmod(value, 1000)
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02}:{minute:02}:{sec:02}.{ms:03}"


def _media_html(video_id: str, start_ms: int = 0) -> str:
    if not video_id:
        return "<p>请选择一个视频。</p>"
    safe_id = quote(str(video_id), safe="")
    return (f'<video id="videolens-player" controls preload="metadata" style="width:100%;max-height:440px;border-radius:12px" '
            f'src="/app/videolens-media/{safe_id}#t={max(0, start_ms) / 1000:.3f}"></video>'
            '<p style="font-size:12px;opacity:.7">点击下方时间线中的行可定位播放；已到期清理的视频需重新导入后播放。</p>')


def _refs_html(video_id: str, refs: list[dict]) -> str:
    """证据 ID 与内容都来自数据，必须转义后才能写入可点击的 HTML。"""
    safe_video = quote(str(video_id), safe="")
    cards = []
    for ref in refs[:30]:
        start = max(0, int(ref.get("start_ms", 0)))
        stamp = _timestamp(start)
        frame = ref.get("frame_id")
        image = (f'<img loading="lazy" style="max-width:240px;max-height:160px;border-radius:6px" '
                 f'src="/app/videolens-frames/{safe_video}/{quote(str(frame), safe="")}"><br>') if frame else ""
        text = html.escape(str(ref.get("text", "")))
        identifier = html.escape(str(ref.get("id", "")))
        cards.append(f'<div style="padding:12px;border:1px solid #8885;border-radius:8px;margin:8px 0">{image}'
                     f'<a href="/app/videolens-media/{safe_video}#t={start / 1000:.3f}" target="_blank" rel="noopener">'
                     f'{stamp} · {identifier}</a><p>{text}</p></div>')
    return "".join(cards) or "<p>暂时没有可展示的证据。</p>"


def _frames_html(video_id: str, frames: list[dict], limit: int = 60) -> str:
    """即使视觉模型未配置，也能查看实际提取出的关键帧。"""
    cards = []
    safe_video = quote(str(video_id), safe="")
    for frame in frames[:limit]:
        frame_id = quote(str(frame["id"]), safe="")
        timestamp = int(frame.get("timestamp_ms", 0))
        caption = html.escape(str(frame.get("description") or frame.get("ocr") or "尚无视觉模型描述"))
        cards.append(f'<figure style="width:240px;margin:8px"><a href="/app/videolens-media/{safe_video}#t={timestamp / 1000:.3f}" target="_blank" rel="noopener">'
                     f'<img loading="lazy" style="width:100%;border-radius:6px" src="/app/videolens-frames/{safe_video}/{frame_id}" alt="{_timestamp(timestamp)} 关键帧"></a>'
                     f'<figcaption>{_timestamp(timestamp)} · {int(frame.get("width", 0))}×{int(frame.get("height", 0))}<br>{caption}</figcaption></figure>')
    note = f"<p>共 {len(frames)} 帧，默认预览前 {limit} 帧；点击时间线行可查看对应切片的全部帧。</p>" if len(frames) > limit else ""
    return note + '<div style="display:flex;flex-wrap:wrap">' + "".join(cards) + "</div>" if cards else "<p>暂时没有关键帧。</p>"


class ApiClient:
    """短连接客户端，避免把 httpx 对象或访问令牌写入 Gradio 的浏览器状态。"""

    def __init__(self, settings: Any):
        host = "[::1]" if settings.host == "::1" else "127.0.0.1"
        self.base_url = f"http://{host}:{settings.port}/api/v1"
        self.headers = {"Authorization": f"Bearer {settings.api_token}"}

    def request(self, method: str, path: str, **kwargs):
        import gradio as gr
        try:
            with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=1800, trust_env=False) as client:
                response = client.request(method, path, **kwargs)
            if not response.is_success:
                try:
                    message = response.json().get("error", {}).get("message", "操作失败")
                except ValueError:
                    message = "服务返回了无法解析的响应"
                raise gr.Error(message)
            return response.json()
        except httpx.HTTPError as exc:
            raise gr.Error("无法连接本地服务，请检查启动终端和环境诊断。") from exc


def mount_ui(app: Any, settings: Any, service: Any):
    """将 Gradio 挂到 /app，并将媒体代理放进它的登录会话保护范围。"""
    # Gradio 上传缓存也归项目管理；关闭使用统计，启动离线界面不发送分析数据。
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("GRADIO_TEMP_DIR", str(Path(settings.data_dir) / "tmp" / "gradio"))
    import gradio as gr
    from fastapi import Depends, HTTPException, Request
    from fastapi.responses import FileResponse, Response

    api = ApiClient(settings)

    def settings_text(result):
        """更新响应含 restart_required；编辑器只展示真正的配置字段。"""
        return json.dumps(result.get("settings", result), ensure_ascii=False, indent=2)

    def require_video(value):
        if not value:
            raise gr.Error("请先导入或选择一个视频。")
        return value

    def videos_dropdown(selected=None):
        videos = api.request("GET", "/videos")
        choices = [(f"{item.get('title', item['id'])} · {item.get('status', '')}", item["id"]) for item in videos]
        ids = {item["id"] for item in videos}
        chosen = selected if selected in ids else (videos[0]["id"] if videos else None)
        return gr.update(choices=choices, value=chosen)

    def startup():
        return videos_dropdown(), json.dumps(api.request("GET", "/settings"), ensure_ascii=False, indent=2)

    def upload(file_path, subtitle_path):
        if not file_path:
            raise gr.Error("请选择视频文件。")
        with ExitStack() as stack:
            files = {"file": (Path(file_path).name, stack.enter_context(open(file_path, "rb")))}
            if subtitle_path:
                files["subtitle"] = (Path(subtitle_path).name, stack.enter_context(open(subtitle_path, "rb")))
            video = api.request("POST", "/videos/upload", files=files)
        task = api.request("POST", f"/videos/{video['id']}/tasks", json={"kind": "analyze"})
        return videos_dropdown(video["id"]), task["id"], task, "已导入并提交后台分析。"

    def import_url(url):
        video = api.request("POST", "/videos/url", json={"url": url.strip()})
        task = api.request("POST", f"/videos/{video['id']}/tasks", json={"kind": "analyze"})
        return videos_dropdown(video["id"]), task["id"], task, "链接已登记，后台任务将获取视频。"

    def submit(video_id, kind):
        task = api.request("POST", f"/videos/{require_video(video_id)}/tasks", json={"kind": kind})
        return task["id"], task

    def task_control(task_id, action):
        if not task_id:
            raise gr.Error("请先选择或创建任务。")
        return api.request("POST", f"/tasks/{task_id}/{action}")

    def poll_task(task_id):
        return api.request("GET", f"/tasks/{task_id}") if task_id else gr.skip()

    def task_choices():
        tasks = api.request("GET", "/tasks")
        return gr.update(choices=[(f"{item.get('kind', 'analyze')} · {item.get('status', '')} · {item['id']}", item["id"]) for item in tasks])

    def load_video(video_id):
        if not video_id:
            return "<p>尚未导入视频。</p>", {}, "暂无报告。", [], "", gr.update(choices=[], value=None), {}, "", ""
        video = api.request("GET", f"/videos/{video_id}")
        report = video.get("report") or {}
        clips = video.get("clips") or []
        rows = [[clip.get("start_time", _timestamp(clip.get("start_ms", 0))),
                 clip.get("end_time", _timestamp(clip.get("end_ms", 0))),
                 clip.get("text") or clip.get("description") or clip.get("ocr", ""), clip["id"]] for clip in clips]
        sessions = api.request("GET", f"/videos/{video_id}/sessions")
        session_update = gr.update(choices=[(f"{item.get('created_at', '')} · {item['id']}", item["id"]) for item in sessions], value=None)
        safe_id = quote(str(video_id), safe="")
        downloads = (f'<a href="/app/videolens-export/{safe_id}/markdown" target="_blank">下载 Markdown 报告</a>　'
                     f'<a href="/app/videolens-export/{safe_id}/json" target="_blank">下载 JSON 数据</a>')
        metadata = {key: value for key, value in video.items() if key not in {"clips", "frames", "report"}}
        markdown = report.get("markdown") or report.get("summary") or "分析尚未完成，请在任务结束后点击「刷新视频详情」。"
        warnings = video.get("warnings") or []
        if warnings:
            markdown = "**本次分析提示：**\n\n" + "\n".join(f"- {str(item)}" for item in warnings) + "\n\n" + markdown
        return (_media_html(video_id), metadata, markdown, rows, _refs_html(video_id, report.get("refs", [])),
                session_update, video, downloads, _frames_html(video_id, video.get("frames", [])))

    def seek(video_id, details, evt: gr.SelectData):
        index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        clips = details.get("clips", [])
        if not 0 <= index < len(clips):
            return gr.skip(), gr.skip()
        selected = clips[index]
        frames = [frame for frame in details.get("frames", []) if frame.get("clip_id") == selected["id"]]
        return _media_html(video_id, int(selected.get("start_ms", 0))), _frames_html(video_id, frames)

    # gr 在函数内导入；把延迟注解转成实际类型，Gradio 才能正确注入点击事件。
    seek.__annotations__["evt"] = gr.SelectData

    def new_session(video_id):
        session = api.request("POST", f"/videos/{require_video(video_id)}/sessions")
        sessions = api.request("GET", f"/videos/{video_id}/sessions")
        return gr.update(choices=[(item["id"], item["id"]) for item in sessions], value=session["id"]), []

    def load_session(session_id):
        if not session_id:
            return []
        messages = api.request("GET", f"/sessions/{session_id}/messages")
        return [{"role": item.get("role", "assistant"), "content": item.get("content", item.get("text", ""))}
                for item in messages if item.get("role") in {"user", "assistant"}]

    def chat(video_id, session_id, question, history):
        require_video(video_id)
        if not question.strip():
            raise gr.Error("请填写问题。")
        if not session_id:
            session = api.request("POST", f"/videos/{video_id}/sessions")
            session_id = session["id"]
        sessions = api.request("GET", f"/videos/{video_id}/sessions")
        session_update = gr.update(choices=[(item["id"], item["id"]) for item in sessions], value=session_id)
        messages = list(history or []) + [{"role": "user", "content": question}, {"role": "assistant", "content": "正在检索视频证据……"}]
        yield messages, session_update, "", "", "正在检索与生成。"
        accumulated, final_received = "", False
        try:
            with httpx.Client(base_url=api.base_url, headers=api.headers, timeout=1800, trust_env=False) as client:
                with client.stream("POST", f"/sessions/{session_id}/ask", json={"question": question, "stream": True}) as response:
                    if not response.is_success:
                        response.read()
                        raise gr.Error(response.json().get("error", {}).get("message", "问答失败"))
                    for event in iter_sse(response.iter_lines()):
                        payload = event["data"]
                        if event["event"] == "delta":
                            accumulated += payload.get("text", "")
                            messages[-1] = {"role": "assistant", "content": accumulated}
                            yield messages, gr.skip(), "", gr.skip(), "正在生成；证据以最终校验结果为准。"
                        elif event["event"] == "final":
                            final_received = True
                            messages[-1] = {"role": "assistant", "content": payload.get("answer", accumulated)}
                            yield messages, gr.skip(), "", _refs_html(video_id, payload.get("refs", [])), f"回答完成；模式：{payload.get('mode', '未知')}。"
                        elif event["event"] == "error":
                            raise gr.Error(payload.get("message", "回答中断"))
            if not final_received:
                raise gr.Error("回答连接提前结束，没有收到最终校验结果。")
        except (httpx.HTTPError, gr.Error) as exc:
            message = str(exc) if isinstance(exc, gr.Error) else "本地服务连接中断。"
            messages[-1] = {"role": "assistant", "content": accumulated + "\n\n⚠ 回答未完成：" + message}
            yield messages, gr.skip(), "", gr.skip(), "回答失败，未作为成功结果显示。"

    def search(video_id, query, image_path):
        require_video(video_id)
        if image_path:
            with open(image_path, "rb") as stream:
                results = api.request("POST", f"/videos/{video_id}/search-image", files={"image": (Path(image_path).name, stream)}, data={"query": query or ""})
        else:
            results = api.request("POST", f"/videos/{video_id}/search", json={"query": query})
        refs = [ref for result in results for ref in result.get("refs", [])]
        return results, _refs_html(video_id, refs)

    def delete(video_id):
        result = api.request("DELETE", f"/videos/{require_video(video_id)}")
        return videos_dropdown(), result

    def read_provider(kind):
        current = api.request("GET", "/settings")
        provider = current.get(f"{kind}_provider", {})
        return (provider.get("enabled", False), provider.get("name", "自定义服务"), provider.get("base_url", ""),
                provider.get("model", ""), provider.get("key_ref", ""), provider.get("context_tokens", 8192),
                provider.get("timeout_seconds", 60), current.get("cloud_enabled", False), "")

    def save_provider(kind, enabled, name, url, model, key_ref, context, timeout, cloud, key):
        if key:
            if not key_ref.strip():
                raise gr.Error("保存密钥前请填写凭据名称，例如 my-text。")
            api.request("PUT", f"/keys/{quote(key_ref.strip(), safe='')}", json={"key": key})
        result = api.request("PATCH", "/settings", json={"cloud_enabled": cloud, f"{kind}_provider": {
            "enabled": enabled, "name": name, "base_url": url.strip(), "model": model.strip(),
            "key_ref": key_ref.strip(), "context_tokens": int(context), "timeout_seconds": float(timeout)}})
        return "模型设置已保存。", settings_text(result), ""

    def save_json(text):
        try:
            patch = json.loads(text)
        except ValueError as exc:
            raise gr.Error("设置必须是有效 JSON；请检查双引号、逗号和括号。") from exc
        if not isinstance(patch, dict):
            raise gr.Error("设置最外层必须是 JSON 对象。")
        # 公共响应会隐藏外部绝对路径。仅发送用户真正修改的项，避免把脱敏占位符
        # 或相对展示路径写回并覆盖后端仍然有效的本机路径。
        baseline = api.request("GET", "/settings")
        patch = {key: value for key, value in patch.items() if baseline.get(key) != value}
        result = api.request("PATCH", "/settings", json=patch)
        suffix = "请重启服务以应用目录、端口或并发变更。" if result.get("restart_required") else "已经生效。"
        return settings_text(result), "设置已保存。" + suffix

    def save_processing(minimum, target, maximum, frames, media_days, failed_hours, ttl_days):
        patch = {"clip_min_seconds": minimum, "clip_target_seconds": target, "clip_max_seconds": maximum,
                 "frames_per_clip": int(frames), "managed_media_days": media_days,
                 "failed_temp_hours": failed_hours, "result_ttl_days": ttl_days}
        current = api.request("PATCH", "/settings", json=patch)
        return settings_text(current), "分析与清理设置已保存。"

    def save_prompts(summary, answer):
        current = api.request("PATCH", "/settings", json={"summary_prompt": summary, "answer_prompt": answer})
        return settings_text(current), "提示词已保存。"

    def default_prompts():
        from .config import Settings
        defaults = Settings()
        return defaults.summary_prompt, defaults.answer_prompt

    # 不给浏览器使用 API Token；Gradio State 仅保存公开的视频详情和会话标识。
    with gr.Blocks(title="知影 · VideoLens") as demo:
        gr.Markdown("# 知影 · VideoLens\n把视频变成可检索、可追问的知识，每个结论都能回到原始证据。")
        with gr.Row():
            video_choice = gr.Dropdown(label="当前视频", choices=[], scale=5)
            refresh_library = gr.Button("刷新视频库", scale=1)
            theme = gr.Radio(["跟随系统", "浅色", "深色"], value="跟随系统", label="外观", scale=1)
        details_state = gr.State({})
        with gr.Tabs():
            with gr.Tab("① 导入与视频库"):
                gr.Markdown("支持 MP4 / MKV / AVI / MOV。字幕优先使用，缺少字幕时尝试本地转写。")
                with gr.Row():
                    video_file = gr.File(label="选择或拖入视频", type="filepath", file_types=[".mp4", ".mkv", ".avi", ".mov"])
                    subtitle_file = gr.File(label="可选字幕（SRT / VTT / ASS）", type="filepath", file_types=[".srt", ".vtt", ".ass"])
                upload_button = gr.Button("上传并分析", variant="primary")
                with gr.Row():
                    url_input = gr.Textbox(label="在线视频链接", placeholder="https://…", scale=5)
                    url_button = gr.Button("获取链接并分析", scale=1)
                import_status = gr.Textbox(label="导入状态", interactive=False)
                metadata = gr.JSON(label="视频信息")
                with gr.Row():
                    analyze_button = gr.Button("重新分析当前视频")
                    reindex_button = gr.Button("重建当前视频索引")
                    delete_button = gr.Button("删除当前视频及其分析数据", variant="stop")
                deletion_status = gr.JSON(label="删除结果")
            with gr.Tab("② 分析任务"):
                gr.Markdown("暂停在安全检查点生效。恢复会复用已完成产物。任务失败或降级原因会保留在详情中。")
                with gr.Row():
                    tasks_select = gr.Dropdown(label="历史任务", choices=[], scale=5)
                    refresh_tasks = gr.Button("刷新任务列表", scale=1)
                task_id = gr.Textbox(label="当前任务 ID", interactive=False)
                with gr.Row():
                    pause_button = gr.Button("暂停")
                    resume_button = gr.Button("恢复")
                    cancel_button = gr.Button("取消", variant="stop")
                task_json = gr.JSON(label="任务进度与阶段详情")
                timer = gr.Timer(value=2)
            with gr.Tab("③ 报告、检索与对话"):
                refresh_details = gr.Button("刷新视频详情（分析完成后点击）")
                with gr.Row():
                    with gr.Column():
                        player = gr.HTML("<p>请选择或导入视频。</p>")
                        timeline = gr.Dataframe(headers=["开始", "结束", "内容", "切片 ID"], datatype=["str", "str", "str", "str"], interactive=False, label="点击任意时间线行定位播放", wrap=True)
                    with gr.Column():
                        report_md = gr.Markdown("暂无报告。", sanitize_html=True)
                        download_links = gr.HTML()
                with gr.Accordion("报告中的原始证据", open=False):
                    report_evidence = gr.HTML()
                with gr.Accordion("关键帧（点击时间线可按切片查看）", open=False):
                    frame_gallery = gr.HTML()
                with gr.Accordion("文本 / 图片检索", open=False):
                    search_query = gr.Textbox(label="检索文字", placeholder="例如：什么是梯度下降？")
                    search_image = gr.Image(label="可选查询图片", type="filepath")
                    search_button = gr.Button("检索")
                    search_results = gr.JSON(label="检索结果与得分")
                    search_evidence = gr.HTML()
                with gr.Row():
                    session_choice = gr.Dropdown(label="独立会话", choices=[], scale=5)
                    session_button = gr.Button("新建会话", scale=1)
                chat_kwargs = {"type": "messages"} if "type" in inspect.signature(gr.Chatbot).parameters else {}
                chatbot = gr.Chatbot(label="围绕当前视频继续追问", sanitize_html=True, height=380, **chat_kwargs)
                question = gr.Textbox(label="你的问题", placeholder="例如：这段视频有哪些核心结论？", lines=2)
                ask_button = gr.Button("发送问题", variant="primary")
                chat_status = gr.Textbox(label="回答状态", interactive=False)
                answer_evidence = gr.HTML()
            with gr.Tab("④ 模型与系统设置"):
                gr.Markdown("**云端数据说明：**启用后会向你指定的接口发送转录、问题和必要关键帧；原始视频不直接上传给模型。密钥写入系统凭据库，页面不会读回。")
                with gr.Accordion("云端文本 / 视觉模型", open=True):
                    provider_kind = gr.Radio([("总结与问答", "text"), ("画面理解与 OCR", "vision")], value="text", label="设置用途")
                    with gr.Row():
                        provider_enabled = gr.Checkbox(label="启用此模型", value=False)
                        cloud_enabled = gr.Checkbox(label="允许将分析内容发送给配置的云端模型", value=False)
                    provider_name = gr.Textbox(label="供应商名称", value="自定义服务")
                    provider_url = gr.Textbox(label="OpenAI 兼容接口地址（含 /v1）", placeholder="https://example.com/v1")
                    provider_model = gr.Textbox(label="模型名称")
                    with gr.Row():
                        provider_ref = gr.Textbox(label="凭据名称", placeholder="例如 my-text")
                        provider_key = gr.Textbox(label="新的 API Key（留空保留原密钥）", type="password")
                    with gr.Row():
                        provider_context = gr.Number(label="上下文容量（Token）", value=8192, precision=0)
                        provider_timeout = gr.Number(label="超时秒数", value=60)
                    with gr.Row():
                        save_provider_button = gr.Button("保存模型设置")
                        check_provider_button = gr.Button("测试此模型连接")
                    connection_result = gr.JSON(label="连接测试结果")
                with gr.Accordion("切片与自动清理", open=False):
                    with gr.Row():
                        clip_min = gr.Number(label="最短切片（秒）", value=settings.clip_min_seconds)
                        clip_target = gr.Number(label="目标切片（秒）", value=settings.clip_target_seconds)
                        clip_max = gr.Number(label="最长切片（秒）", value=settings.clip_max_seconds)
                        frame_count = gr.Number(label="每切片最多帧数", value=settings.frames_per_clip, precision=0)
                    with gr.Row():
                        media_days = gr.Number(label="媒体保留天数", value=settings.managed_media_days)
                        temp_hours = gr.Number(label="失败临时文件保留小时", value=settings.failed_temp_hours)
                        ttl_days = gr.Number(label="分析结果保留天数（0=长期）", value=settings.result_ttl_days)
                    processing_button = gr.Button("保存切片与清理设置")
                with gr.Accordion("提示词", open=False):
                    summary_prompt = gr.Textbox(label="总结提示词", value=settings.summary_prompt, lines=3)
                    answer_prompt = gr.Textbox(label="问答提示词", value=settings.answer_prompt, lines=3)
                    with gr.Row():
                        save_prompts_button = gr.Button("保存提示词")
                        reset_prompts_button = gr.Button("填写默认提示词（再保存生效）")
                with gr.Accordion("完整配置（目录、模型、并发、预算、检索权重等）", open=False):
                    gr.Markdown("JSON 编辑器包含全部可编辑参数。目录与端口变更可能需要重启。字段含义见开发文档；API Key 不得写在这里。")
                    config_editor = gr.Code(language="json", label="配置 JSON", lines=22)
                    save_json_button = gr.Button("保存完整配置")
                settings_status = gr.Textbox(label="设置状态", interactive=False)
                with gr.Row():
                    doctor_button = gr.Button("检查环境与模型")
                    cleanup_button = gr.Button("立即执行到期清理")
                maintenance = gr.JSON(label="维护结果")

        refresh_library.click(videos_dropdown, inputs=video_choice, outputs=video_choice)
        demo.load(startup, outputs=[video_choice, config_editor])
        demo.load(read_provider, inputs=provider_kind, outputs=[provider_enabled, provider_name, provider_url, provider_model, provider_ref, provider_context, provider_timeout, cloud_enabled, provider_key])
        video_outputs = [player, metadata, report_md, timeline, report_evidence, session_choice, details_state, download_links, frame_gallery]
        video_choice.change(load_video, inputs=video_choice, outputs=video_outputs)
        video_choice.change(lambda: ([], "", ""), outputs=[chatbot, answer_evidence, chat_status])
        refresh_details.click(load_video, inputs=video_choice, outputs=video_outputs)
        refresh_details.click(lambda: ([], "", ""), outputs=[chatbot, answer_evidence, chat_status])
        upload_button.click(upload, inputs=[video_file, subtitle_file], outputs=[video_choice, task_id, task_json, import_status])
        url_button.click(import_url, inputs=url_input, outputs=[video_choice, task_id, task_json, import_status])
        analyze_button.click(lambda value: submit(value, "analyze"), inputs=video_choice, outputs=[task_id, task_json])
        reindex_button.click(lambda value: submit(value, "reindex"), inputs=video_choice, outputs=[task_id, task_json])
        delete_button.click(delete, inputs=video_choice, outputs=[video_choice, deletion_status])
        refresh_tasks.click(task_choices, outputs=tasks_select)
        tasks_select.change(lambda value: (value, poll_task(value)), inputs=tasks_select, outputs=[task_id, task_json])
        timer.tick(poll_task, inputs=task_id, outputs=task_json)
        pause_button.click(lambda value: task_control(value, "pause"), inputs=task_id, outputs=task_json)
        resume_button.click(lambda value: task_control(value, "resume"), inputs=task_id, outputs=task_json)
        cancel_button.click(lambda value: task_control(value, "cancel"), inputs=task_id, outputs=task_json)
        timeline.select(seek, inputs=[video_choice, details_state], outputs=[player, frame_gallery])
        session_button.click(new_session, inputs=video_choice, outputs=[session_choice, chatbot])
        session_choice.input(load_session, inputs=session_choice, outputs=chatbot)
        ask_button.click(chat, inputs=[video_choice, session_choice, question, chatbot], outputs=[chatbot, session_choice, question, answer_evidence, chat_status], concurrency_limit=1)
        search_button.click(search, inputs=[video_choice, search_query, search_image], outputs=[search_results, search_evidence])
        provider_kind.change(read_provider, inputs=provider_kind, outputs=[provider_enabled, provider_name, provider_url, provider_model, provider_ref, provider_context, provider_timeout, cloud_enabled, provider_key])
        save_provider_button.click(save_provider, inputs=[provider_kind, provider_enabled, provider_name, provider_url, provider_model, provider_ref, provider_context, provider_timeout, cloud_enabled, provider_key], outputs=[settings_status, config_editor, provider_key])
        check_provider_button.click(lambda kind: api.request("POST", f"/providers/{kind}/check"), inputs=provider_kind, outputs=connection_result)
        processing_button.click(save_processing, inputs=[clip_min, clip_target, clip_max, frame_count, media_days, temp_hours, ttl_days], outputs=[config_editor, settings_status])
        save_prompts_button.click(save_prompts, inputs=[summary_prompt, answer_prompt], outputs=[config_editor, settings_status])
        reset_prompts_button.click(default_prompts, outputs=[summary_prompt, answer_prompt])
        save_json_button.click(save_json, inputs=config_editor, outputs=[config_editor, settings_status])
        doctor_button.click(lambda: api.request("GET", "/doctor"), outputs=maintenance)
        cleanup_button.click(lambda: api.request("POST", "/cleanup"), outputs=maintenance)
        theme.change(fn=None, inputs=theme, js="(value) => { const dark = value === '深色' || (value === '跟随系统' && window.matchMedia('(prefers-color-scheme: dark)').matches); document.documentElement.classList.toggle('dark', dark); document.body.classList.toggle('dark', dark); }")

    demo.queue(default_concurrency_limit=4)
    app = gr.mount_gradio_app(app, demo, path="/app", auth=(settings.ui_username, settings.ui_password),
                              auth_message="知影本地登录：用户名默认为 videolens，密码见本机 data/.access-token 文件。")
    mounted = next(route.app for route in app.routes if getattr(route, "path", None) == "/app")

    def logged_in(request):
        # Gradio 6 把 get_current_user 定义为局部函数；这里读取它实际使用的
        # 两个 HttpOnly cookie，并对照同一个服务端会话表，绝不只凭 cookie 存在放行。
        cookie_id = getattr(mounted, "cookie_id", "")
        token = request.cookies.get(f"access-token-{cookie_id}") or request.cookies.get(f"access-token-unsecure-{cookie_id}")
        user = getattr(mounted, "tokens", {}).get(token)
        if not user:
            raise HTTPException(401, "请先登录知影网页")
        return user

    logged_in.__annotations__["request"] = Request

    # 这些路由属于 Gradio 子应用，与它的登录/退出共享会话令牌表。
    @mounted.get("/videolens-media/{video_id}", dependencies=[Depends(logged_in)])
    def ui_media(video_id: str):
        return FileResponse(service.media_path(video_id))

    @mounted.get("/videolens-frames/{video_id}/{frame_id}", dependencies=[Depends(logged_in)])
    def ui_frame(video_id: str, frame_id: str):
        return FileResponse(service.media_path(video_id, frame_id))

    @mounted.get("/videolens-export/{video_id}/{format}", dependencies=[Depends(logged_in)])
    def ui_export(video_id: str, format: str):
        if format not in {"markdown", "json"}:
            raise HTTPException(422, "格式仅支持 markdown 或 json")
        extension = "json" if format == "json" else "md"
        return Response(service.export(video_id, format), media_type="application/json" if format == "json" else "text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="videolens-report.{extension}"'})

    return app
