"""Gradio 网页：所有操作通过同一套 HTTP API，令牌只停留在 Python 后端。

页面中的 HTML 仅使用程序生成的结构，并对标题、证据、ID 逐项转义。
模型输出放入启用 sanitize_html 的 Markdown/Chatbot，不能直接变成网页脚本。
"""

from __future__ import annotations

import json
import os
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
    """无需账号密码。输入和输出始终可见，其余操作集中在折叠设置中。"""
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    import gradio as gr
    from fastapi import HTTPException
    from fastapi.responses import FileResponse, Response
    from .bootstrap import MODEL_SPECS

    api = ApiClient(settings)
    states = {'queued': '排队中', 'running': '分析中', 'succeeded': '完成', 'partial': '部分完成（存在降级）',
              'failed': '失败', 'paused': '已暂停', 'cancelled': '已取消'}
    stages = {'acquire': '获取视频', 'probe': '读取媒体', 'transcript': '字幕与转写', 'clips': '划分切片',
              'frames': '提取关键帧', 'vision': '理解画面', 'index': '建立检索索引', 'summary': '生成报告',
              'completed': '处理完成', 'queued': '等待处理'}

    def library(selected=None):
        videos = api.request('GET', '/videos')
        choices = [(item.get('title', item['id']), item['id']) for item in videos]
        chosen = selected if selected in {v['id'] for v in videos} else (videos[0]['id'] if videos else None)
        return gr.update(choices=choices, value=chosen)

    def report_link(video_id):
        if not video_id:
            return ''
        video = api.request('GET', f'/videos/{video_id}')
        if not video.get('report'):
            return '分析完成后，这里会出现 Markdown 报告下载链接。'
        return f'[下载分析报告（.md）](/app/videolens-export/{quote(video_id, safe="")}/markdown)'

    def select_video(video_id):
        # 一个视频一个页面会话；切换视频不串用另一视频的历史。
        return report_link(video_id), '', '', ''

    def analyze(file, url, selected):
        if file and url.strip():
            raise gr.Error('本地视频和 URL 选择其中一种即可。')
        if file:
            with open(file, 'rb') as stream:
                video = api.request('POST', '/videos/upload', files={'file': (Path(file).name, stream)})
            selected = video['id']
        elif url.strip():
            selected = api.request('POST', '/videos/url', json={'url': url.strip()})['id']
        if not selected:
            raise gr.Error('请拖入视频、输入 URL，或从视频库选择一个视频。')
        task = api.request('POST', f'/videos/{selected}/tasks', json={'kind': 'analyze'})
        return library(selected), task['id'], '已提交分析，进度会自动更新。', '', '', ''

    def progress(task_id, video_id):
        if not task_id:
            return gr.skip(), gr.skip()
        task = api.request('GET', f'/tasks/{task_id}')
        if task.get('video_id') and task['video_id'] != video_id:
            return gr.skip(), gr.skip()
        status = task['status']
        detail = task.get('error') or '；'.join(task.get('warnings', []))
        text = f"{states.get(status, status)} · {stages.get(task.get('stage'), '')} · {task.get('progress', 0):.0f}%"
        if detail:
            text += '\n' + str(detail)
        return text, report_link(video_id) if status in {'succeeded', 'partial'} else gr.skip()

    def ask(video_id, session, question):
        if not video_id or not question.strip():
            raise gr.Error('请选择已分析的视频，并输入问题。')
        if not session:
            session = api.request('POST', f'/videos/{video_id}/sessions')['id']
        yield '正在检索视频证据……', session
        answer, complete = '', False
        try:
            with httpx.Client(base_url=api.base_url, headers=api.headers, timeout=1800, trust_env=False) as client:
                with client.stream('POST', f'/sessions/{session}/ask', json={'question': question, 'stream': True}) as response:
                    if not response.is_success:
                        response.read()
                        raise gr.Error(response.json().get('error', {}).get('message', '问答失败'))
                    for event in iter_sse(response.iter_lines()):
                        data = event['data']
                        if event['event'] == 'delta':
                            answer += data.get('text', '')
                            yield answer, session
                        elif event['event'] == 'final':
                            complete = True
                            yield data.get('answer', answer), session
                        elif event['event'] == 'error':
                            raise gr.Error(data.get('message', '回答中断'))
            if not complete:
                raise gr.Error('连接提前结束，未收到最终校验结果。')
        except (httpx.HTTPError, gr.Error) as exc:
            text = str(exc) if isinstance(exc, gr.Error) else '连接中断，请检查本地服务。'
            yield answer + '\n\n**回答未完成：**' + text, session

    def read_provider(kind):
        current = api.request('GET', '/settings')
        p = current.get(kind + '_provider', {})
        return (p.get('enabled', False), current.get('cloud_enabled', False), p.get('base_url', ''),
                p.get('model', ''), p.get('context_tokens', 8192), p.get('timeout_seconds', 120), '')

    def save_provider(kind, enabled, cloud, url, model, context, timeout, key):
        old = api.request('GET', '/settings').get(kind + '_provider', {})
        ref = old.get('key_ref') or 'web-' + kind
        if key.strip():
            api.request('PUT', '/keys/' + quote(ref, safe=''), json={'key': key.strip()})
        api.request('PATCH', '/settings', json={'cloud_enabled': cloud, kind + '_provider': {
            'enabled': enabled, 'name': '网页配置', 'base_url': url.strip().rstrip('/'), 'model': model.strip(),
            'key_ref': ref, 'context_tokens': int(context), 'timeout_seconds': timeout}})
        return '模型设置已保存，API Key 不会读回页面。', ''

    def save_prompts(summary, answer):
        api.request('PATCH', '/settings', json={'summary_prompt': summary, 'answer_prompt': answer})
        return '系统提示词已保存。'

    def models(selected=None):
        data = api.request('GET', '/local-models')
        choices = [(f"{r['name']}（{r['bytes'] / 1024 ** 3:.2f} GiB）", r['id']) for r in data['models']]
        ids = {r['id'] for r in data['models']}
        chosen = selected if selected in ids else next((r['id'] for r in data['models'] if r['selected']), choices[0][1] if choices else None)
        text = 'llama.cpp：' + ('运行中' if data['running'] else '未运行')
        if data['jobs']:
            job = data['jobs'][-1]
            text += f"\n最近操作：{job['message']}（{states.get(job['status'], job['status'])}）"
            if job['total']:
                text += f" · {job['bytes'] / 1024 ** 2:.0f} / {job['total'] / 1024 ** 2:.0f} MiB"
        return gr.update(choices=choices, value=chosen), text

    def transfer(kind, path, url, checksum):
        api.request('POST', '/local-models/' + kind, json={'path': (path or '').strip().strip('"'), 'url': url or '', 'sha256': checksum or ''})
        return '已提交，处理进度会自动更新。'

    def start(model_id, layers, context):
        if not model_id:
            raise gr.Error('先下载或导入模型，再选择模型。')
        api.request('POST', f'/local-models/{model_id}/start', json={'gpu_layers': int(layers), 'context': int(context)})
        return '模型已通过健康检查，现在可以分析和问答。'

    def delete(model_id, confirmed):
        if not model_id or not confirmed:
            raise gr.Error('请选择模型并勾选删除确认。')
        result = api.request('DELETE', '/local-models/' + model_id)
        return result['message'], False

    spec = MODEL_SPECS['local']
    with gr.Blocks(title='知影 · VideoLens', delete_cache=(3600, 86400)) as demo:
        gr.Markdown('# 知影 · VideoLens\n导入视频，下载分析报告，围绕视频继续提问。')
        task = gr.State('')
        session = gr.State('')
        with gr.Accordion('设置', open=False):
            refresh = gr.Button('刷新视频库')
            with gr.Tabs():
                with gr.Tab('云端模型'):
                    gr.Markdown('启用云端会发送转录、问题和必要关键帧。密钥加密保存在本项目 data/secrets 中。')
                    kind = gr.Radio([('总结与问答', 'text'), ('画面理解与 OCR', 'vision')], value='text', label='用途')
                    with gr.Row():
                        enabled = gr.Checkbox(label='启用此用途的云端模型')
                        cloud = gr.Checkbox(label='允许云端分析（总开关）')
                    base = gr.Textbox(label='接口 Base URL', placeholder='例如 https://dashscope.aliyuncs.com/compatible-mode/v1')
                    model = gr.Textbox(label='模型 ID', placeholder='例如 qwen-plus')
                    key = gr.Textbox(label='API Key（留空保留原密钥）', type='password')
                    with gr.Row():
                        context = gr.Number(label='上下文预算（Token）', value=8192, precision=0)
                        timeout = gr.Number(label='请求超时（秒）', value=120)
                    with gr.Row():
                        save_cloud = gr.Button('保存云端设置')
                        check = gr.Button('测试连接')
                with gr.Tab('本地模型 · llama.cpp'):
                    gr.Markdown('单文件 GGUF。导入会复制到项目，原文件保留。选择模型后点击启动即可生效，无需重启网页。')
                    local = gr.Dropdown(label='本地模型', choices=[])
                    with gr.Row():
                        layers = gr.Number(label='GPU 卸载层数（0=CPU，99=尽量全卸载）', value=settings.llama_gpu_layers, precision=0)
                        local_context = gr.Number(label='本地上下文（Token）', value=settings.local_context_tokens, precision=0)
                    with gr.Row():
                        start_button = gr.Button('应用并启动所选模型')
                        stop_button = gr.Button('停止本地模型')
                    local_status = gr.Textbox(label='模型运行与下载进度', interactive=False)
                    with gr.Accordion('下载 / 导入模型', open=False):
                        path = gr.Textbox(label='已有 GGUF 的本机完整路径', placeholder=r'D:\模型\example.gguf')
                        import_button = gr.Button('导入模型副本')
                        url = gr.Textbox(label='GGUF 的 HTTPS 下载直链', value=f"https://huggingface.co/{spec['repo_id']}/resolve/{spec['revision']}/{spec['filename']}")
                        checksum = gr.Textbox(label='发布者提供的 SHA-256', value=spec['sha256'])
                        download_button = gr.Button('下载模型（默认 Qwen3-4B Q4_K_M）')
                    with gr.Accordion('删除模型 / 部署运行时', open=False):
                        confirm = gr.Checkbox(label='确认删除所选模型的项目内副本')
                        delete_button = gr.Button('删除所选模型', variant='stop')
                        runtime = gr.Radio(['cpu', 'cuda'], value='cpu', label='llama.cpp 运行时类型')
                        deploy = gr.Button('下载并部署 llama.cpp')
                with gr.Tab('系统提示词'):
                    summary_prompt = gr.Textbox(label='分析报告系统提示词', value=settings.summary_prompt, lines=4)
                    answer_prompt = gr.Textbox(label='问答系统提示词', value=settings.answer_prompt, lines=4)
                    save_prompt = gr.Button('保存系统提示词')
            setting_status = gr.Textbox(label='设置操作结果', interactive=False)
        with gr.Row():
            with gr.Column():
                gr.Markdown('## 输入')
                video = gr.Dropdown(label='视频库 / 当前视频', choices=[])
                file = gr.File(label='本地视频（拖入或选择）', type='filepath', file_types=['.mp4', '.mkv', '.avi', '.mov'])
                video_url = gr.Textbox(label='或输入视频 URL', placeholder='https://…')
                analyze_button = gr.Button('分析视频', variant='primary')
                question = gr.Textbox(label='问答提示词', placeholder='例如：这段视频的核心结论是什么？', lines=3)
                ask_button = gr.Button('发送问题', variant='primary')
            with gr.Column():
                gr.Markdown('## 输出')
                status = gr.Textbox(label='分析进度', interactive=False)
                report = gr.Markdown('分析完成后可下载 .md 报告。', sanitize_html=True)
                answer = gr.Markdown('回答将在这里直接显示。', sanitize_html=True)
        poll = gr.Timer(3)
        demo.load(library, outputs=video).then(select_video, inputs=video, outputs=[report, session, answer, task])
        provider_outputs = [enabled, cloud, base, model, context, timeout, key]
        demo.load(read_provider, inputs=kind, outputs=provider_outputs)
        demo.load(models, inputs=local, outputs=[local, local_status])
        refresh.click(library, inputs=video, outputs=video).then(report_link, inputs=video, outputs=report)
        video.input(select_video, inputs=video, outputs=[report, session, answer, task])
        analyze_button.click(analyze, inputs=[file, video_url, video], outputs=[video, task, status, report, session, answer])
        poll.tick(progress, inputs=[task, video], outputs=[status, report], show_progress='hidden')
        poll.tick(models, inputs=local, outputs=[local, local_status], show_progress='hidden')
        ask_button.click(ask, inputs=[video, session, question], outputs=[answer, session], concurrency_limit=1)
        kind.change(read_provider, inputs=kind, outputs=provider_outputs)
        save_cloud.click(save_provider, inputs=[kind, enabled, cloud, base, model, context, timeout, key], outputs=[setting_status, key])
        check.click(lambda k: api.request('POST', f'/providers/{k}/check')['message'], inputs=kind, outputs=setting_status)
        save_prompt.click(save_prompts, inputs=[summary_prompt, answer_prompt], outputs=setting_status)
        import_button.click(lambda p: transfer('import', p, '', ''), inputs=path, outputs=setting_status)
        download_button.click(lambda u, h: transfer('download', '', u, h), inputs=[url, checksum], outputs=setting_status)
        start_button.click(start, inputs=[local, layers, local_context], outputs=setting_status)
        def stop_message():
            result = api.request('POST', '/local-models/stop')
            return '本地模型已停止。' if result.get('stopped') else '当前没有正在运行的受管模型。'
        stop_button.click(stop_message, outputs=setting_status)
        delete_button.click(delete, inputs=[local, confirm], outputs=[setting_status, confirm])
        deploy.click(lambda v: api.request('POST', '/local-models/runtime/' + v)['message'], inputs=runtime, outputs=setting_status)

    demo.queue(default_concurrency_limit=4)
    app = gr.mount_gradio_app(app, demo, path='/app', show_error=False)
    mounted = next(route.app for route in app.routes if getattr(route, 'path', None) == '/app')

    @mounted.get('/videolens-media/{video_id}')
    def ui_media(video_id: str):
        return FileResponse(service.media_path(video_id))

    @mounted.get('/videolens-frames/{video_id}/{frame_id}')
    def ui_frame(video_id: str, frame_id: str):
        return FileResponse(service.media_path(video_id, frame_id))

    @mounted.get('/videolens-export/{video_id}/{format}')
    def ui_export(video_id: str, format: str):
        if format not in {'markdown', 'json'}:
            raise HTTPException(422, '格式仅支持 markdown 或 json')
        extension = 'json' if format == 'json' else 'md'
        content = service.export(video_id, format)
        # 报告原件始终保存到项目内；浏览器下载副本的位置由浏览器自身决定。
        folder = Path(settings.data_dir) / 'exports'
        folder.mkdir(parents=True, exist_ok=True)
        safe_id = ''.join(c for c in video_id if c.isalnum() or c in '-_')
        (folder / f'{safe_id}.{extension}').write_text(content, encoding='utf-8')
        return Response(content, media_type='application/json' if format == 'json' else 'text/markdown; charset=utf-8',
                        headers={'Content-Disposition': f'attachment; filename="videolens-report.{extension}"'})
    return app
