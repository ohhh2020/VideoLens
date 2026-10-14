"""中文命令行入口。除 serve 与模型安装外，命令都调用正在运行的同一套 API。

示例：python -m videolens analyze "课程.mp4" --subtitle "课程.srt" --wait
这样 CLI 和网页共享任务、视频和会话，而不是维护两套各自独立的数据。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import httpx

from .config import load_settings

TERMINAL_STATES = {"succeeded", "completed", "success", "partial", "partial_success", "failed", "cancelled", "canceled", "paused"}


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def read_sse(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """在完整空行边界解析事件，HTTP 分包不会把 JSON 截成两段。"""
    event, data = "message", []
    for line in lines:
        if not line:
            if data:
                try:
                    payload = json.loads("\n".join(data))
                except json.JSONDecodeError as exc:
                    raise RuntimeError("服务返回了无效的流式事件") from exc
                yield {"event": event, "data": payload}
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))


def _request(client: httpx.Client, method: str, route: str, **kwargs) -> httpx.Response:
    response = client.request(method, route, **kwargs)
    if not response.is_success:
        try:
            message = response.json().get("error", {}).get("message", "操作失败")
        except ValueError:
            message = "服务返回了非预期响应"
        raise RuntimeError(f"HTTP {response.status_code}：{message}")
    return response


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="videolens", description="知影：本地视频分析、检索与可追溯问答")
    parser.add_argument("--config", help="指定 UTF-8 JSON 配置文件")
    commands = parser.add_subparsers(dest="command", required=True, title="可用命令")
    serve = commands.add_parser("serve", help="启动后端与 Gradio 网页")
    serve.add_argument("--host", help="监听地址，首版仅支持本机回环地址")
    serve.add_argument("--port", type=int, help="监听端口，默认读取配置")
    serve.add_argument("--no-ui", action="store_true", help="只启动 API")
    serve.add_argument("--open-browser", action="store_true", help="健康检查通过后自动打开网页")
    commands.add_parser("doctor", help="检查正在运行的服务、环境与模型")
    commands.add_parser("videos", help="列出视频库")
    analyze = commands.add_parser("analyze", help="导入本地视频或 HTTP(S) 链接并分析")
    analyze.add_argument("source", help="本地视频路径或在线视频 URL")
    analyze.add_argument("--subtitle", help="可选 SRT/VTT/ASS 字幕路径（仅本地视频）")
    analyze.add_argument("--wait", action="store_true", help="等待任务进入终态并持续显示进度")
    ask = commands.add_parser("ask", help="围绕视频提问，可复用会话")
    ask.add_argument("video_id", help="视频 ID，可用 videos 命令查询")
    ask.add_argument("question", help="问题，包含空格时请用双引号包围")
    ask.add_argument("--session", help="已有会话 ID；不指定则新建会话")
    ask.add_argument("--stream", action="store_true", help="实时接收回答与最终证据")
    search = commands.add_parser("search", help="按文字或图片检索证据")
    search.add_argument("video_id", help="视频 ID")
    search.add_argument("query", nargs="?", default="", help="检索文字")
    search.add_argument("--image", help="查询图片路径")
    export = commands.add_parser("export", help="导出报告，全部采用 UTF-8")
    export.add_argument("video_id", help="视频 ID")
    export.add_argument("--format", choices=["markdown", "json"], default="markdown", help="导出格式")
    export.add_argument("--output", help="输出路径；不指定时写到标准输出")
    tasks = commands.add_parser("tasks", help="查询或控制后台任务")
    tasks.add_argument("task_id", nargs="?", help="任务 ID；不指定时列出全部任务")
    tasks.add_argument("--action", choices=["pause", "resume", "cancel"], help="暂停、恢复或取消指定任务")
    tasks.add_argument("--follow", action="store_true", help="通过 SSE 持续查看指定任务")
    models = commands.add_parser("models", help="查看、下载、运行和验证本地模型")
    models.add_argument("action", nargs="?", choices=["list", "download", "runtime", "start-local", "stop-local", "verify"], default="list")
    models.add_argument("--names", default="asr,text,image,local", help="逗号分隔：asr,text,image,local（用于 download）")
    models.add_argument("extra", nargs=argparse.REMAINDER, help="传给模型工具的其他参数")
    reindex = commands.add_parser("reindex", help="重建指定视频的向量索引")
    reindex.add_argument("video_id", help="视频 ID")
    commands.add_parser("cleanup", help="立即执行配置中的到期清理")
    return parser


def _follow(client: httpx.Client, task_id: str) -> int:
    """SSE 断开后用最后一个事件 ID 重新订阅，避免重复打印已收到的进度。"""
    cursor = 0
    for attempt in range(3):
        try:
            with client.stream("GET", f"tasks/{task_id}/events", headers={"Last-Event-ID": str(cursor)}) as response:
                response.raise_for_status()
                event, data = "message", []
                for line in response.iter_lines():
                    if line.startswith("id:"):
                        cursor = max(cursor, int(line[3:].strip()))
                    elif line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        data.append(line[5:].lstrip())
                    elif not line and data:
                        payload = json.loads("\n".join(data))
                        _print_json({"event": event, "data": payload})
                        if event == "done":
                            status = payload.get("status", payload.get("state"))
                            return 1 if status in {"failed", "cancelled", "canceled"} else 0
                        event, data = "message", []
            # 连接结束但没有 done，需要重新确认，不把网络断开当成任务成功。
            task = _request(client, "GET", f"tasks/{task_id}").json()
            if task.get("status", task.get("state")) in TERMINAL_STATES:
                _print_json(task)
                return 1 if task.get("status", task.get("state")) in {"failed", "cancelled", "canceled"} else 0
        except (httpx.TransportError, json.JSONDecodeError):
            if attempt == 2:
                raise RuntimeError("任务事件连接中断，请用 tasks 命令查询实际状态")
        time.sleep(attempt + 1)
    raise RuntimeError("未收到任务终态，请用 tasks 命令继续查询")


def _serve(settings, args) -> int:
    import socket
    import uvicorn
    from .api import create_app

    # model_copy(update=...) 不执行校验；显式重建可拒绝意外监听公网地址。
    if args.host or args.port:
        values = settings.model_dump()
        values.update({"api_token": settings.api_token})
        if args.host:
            values["host"] = args.host
        if args.port:
            values["port"] = args.port
        settings = type(settings)(**values)
    # 在创建业务队列之前检查端口，重复双击不会先恢复一遍任务才发现监听冲突。
    # 最终绑定仍交给 Uvicorn；若检查后有其他程序抢占，Uvicorn 会明确报错退出。
    try:
        with socket.socket(socket.AF_INET6 if settings.host == "::1" else socket.AF_INET) as probe:
            probe.bind((settings.host, settings.port))
    except OSError as exc:
        raise RuntimeError(f"端口 {settings.port} 已被占用或不可用；请关闭重复的知影窗口，或在配置中更换端口。") from exc
    host = "[::1]" if settings.host == "::1" else settings.host
    address = f"http://{host}:{settings.port}"
    logging.getLogger(__name__).info("知影正在启动：%s/app（无需登录）", address)
    if args.open_browser:
        def open_when_ready():
            for _ in range(120):
                try:
                    with httpx.Client(timeout=1, trust_env=False) as client:
                        if client.get(address + "/health").is_success:
                            from .paths import open_project_browser
                            try:
                                open_project_browser(address + ("/health" if args.no_ui else "/app"))
                            except RuntimeError as exc:
                                logging.getLogger(__name__).warning("%s", exc)
                            return
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
        threading.Thread(target=open_when_ready, daemon=True, name="videolens-browser").start()
    uvicorn.run(create_app(settings, with_ui=not args.no_ui), host=settings.host, port=settings.port, access_log=False, log_config=None)
    return 0


def main(argv: list[str] | None = None) -> int:
    """返回退出码；脚本用它区分成功与失败，不能只看有没有打印文字。"""
    # Windows 管道和重定向也保持 UTF-8，不依赖当前控制台代码页。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    from .logging_setup import configure_logging
    from .paths import configure_paths
    configure_logging(configure_paths())
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        if args.command == "serve":
            return _serve(settings, args)
        if args.command == "models" and args.action != "list":
            action = {"download": "models", "verify": "verify-models"}.get(args.action, args.action)
            command = [sys.executable, "-m", "videolens.bootstrap", action]
            if args.action in {"download", "verify"}:
                command += ["--names", args.names]
            command += args.extra
            environment = os.environ.copy()
            if args.config:
                environment["VIDEOLENS_CONFIG"] = str(Path(args.config).resolve())
            return subprocess.run(command, env=environment, check=False).returncode
        host = "[::1]" if settings.host == "::1" else "127.0.0.1"
        with httpx.Client(base_url=f"http://{host}:{settings.port}/api/v1/", timeout=1800,
                          headers={"Authorization": f"Bearer {settings.api_token}"}, trust_env=False) as client:
            if args.command in {"doctor", "videos", "models"}:
                route = "videos" if args.command == "videos" else "doctor"
                _print_json(_request(client, "GET", route).json())
            elif args.command == "analyze":
                if args.source.lower().startswith(("https://", "http://")):
                    if args.subtitle:
                        parser.error("URL 导入暂不支持单独指定字幕；请下载后用本地路径导入")
                    video = _request(client, "POST", "videos/url", json={"url": args.source}).json()
                else:
                    video = _request(client, "POST", "videos/import", json={"path": str(Path(args.source).resolve()),
                                     "subtitle": str(Path(args.subtitle).resolve()) if args.subtitle else None}).json()
                task = _request(client, "POST", f"videos/{video['id']}/tasks", json={"kind": "analyze"}).json()
                _print_json({"video": video, "task": task})
                if args.wait:
                    return _follow(client, task["id"])
            elif args.command == "tasks":
                if (args.action or args.follow) and not args.task_id:
                    parser.error("--action 和 --follow 需要提供 task_id")
                if args.action:
                    _print_json(_request(client, "POST", f"tasks/{args.task_id}/{args.action}").json())
                elif args.task_id:
                    _print_json(_request(client, "GET", f"tasks/{args.task_id}").json())
                else:
                    _print_json(_request(client, "GET", "tasks").json())
                if args.follow:
                    return _follow(client, args.task_id)
            elif args.command == "search":
                if args.image:
                    image = Path(args.image)
                    with image.open("rb") as handle:
                        result = _request(client, "POST", f"videos/{args.video_id}/search-image", files={"image": (image.name, handle)}, data={"query": args.query})
                else:
                    if not args.query.strip():
                        parser.error("请填写检索文字或使用 --image 指定图片")
                    result = _request(client, "POST", f"videos/{args.video_id}/search", json={"query": args.query})
                _print_json(result.json())
            elif args.command == "ask":
                if args.session:
                    sessions = _request(client, "GET", f"videos/{args.video_id}/sessions").json()
                    if args.session not in {item["id"] for item in sessions}:
                        raise RuntimeError("该会话不属于指定视频，不能混用会话证据")
                    session_id = args.session
                else:
                    session_id = _request(client, "POST", f"videos/{args.video_id}/sessions").json()["id"]
                print(f"会话 ID：{session_id}", file=sys.stderr)
                if not args.stream:
                    _print_json(_request(client, "POST", f"sessions/{session_id}/ask", json={"question": args.question}).json())
                else:
                    final_received = False
                    with client.stream("POST", f"sessions/{session_id}/ask", json={"question": args.question, "stream": True}) as response:
                        response.raise_for_status()
                        for event in read_sse(response.iter_lines()):
                            payload = event["data"]
                            if event["event"] == "delta":
                                print(payload.get("text", ""), end="", flush=True)
                            elif event["event"] == "final":
                                final_received = True
                                print("\n最终校验结果：")
                                _print_json(payload)
                            elif event["event"] == "error":
                                raise RuntimeError(payload.get("message", "回答中断"))
                    if not final_received:
                        raise RuntimeError("回答流提前结束，未收到最终结果")
            elif args.command == "export":
                result = _request(client, "GET", f"videos/{args.video_id}/export", params={"format": args.format})
                if args.output:
                    target = Path(args.output)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(result.text, encoding="utf-8")
                    print(f"已导出到：{target.resolve()}")
                else:
                    print(result.text)
            elif args.command == "reindex":
                _print_json(_request(client, "POST", f"videos/{args.video_id}/tasks", json={"kind": "reindex"}).json())
            elif args.command == "cleanup":
                _print_json(_request(client, "POST", "cleanup").json())
        return 0
    except httpx.HTTPError:
        print("无法完成本地 API 请求，请先运行「一键启动.cmd」，再检查服务与网络连接。", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"操作失败：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n命令已停止；后台任务状态请用 tasks 查询。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
