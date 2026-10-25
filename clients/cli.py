"""知影命令行：``python -m clients.cli <子命令>``。

设计取向
--------
1. **本地优先、零服务依赖**：默认直接装配 ``Orchestrator`` 跑，不要求先起
   uvicorn。要连远端服务时加 ``--api http://127.0.0.1:8000 --token xxx``，
   走同一套子命令（内部换成 HTTP 后端）。
2. **可脚本化**：所有读操作支持 ``--json``，输出为纯 JSON（不含进度条/ANSI），
   方便接 shell 管道与其它程序；进度条一律写 stderr，不污染 stdout。
3. **失败要能看懂**：不给裸 traceback。``PipelineError``/``RepositoryError``/
   ``MediaError`` 都翻译成一句人话 + 退出码（见 ``EXIT_*``）。

子命令
------
``analyze``  分析本地文件或 URL（转写 → 切片 → 抽帧 → 向量 → 总结）
``ask``      对已索引视频提问，输出回答 + 带毫秒时间戳的引用
``list``     列出库里的视频
``show``     视频详情（切片 + 关键帧）
``export``   导出 Markdown 报告
``stats``    存储用量（对照需求文档表3 容量上限）
``delete``   删除视频及其向量
``doctor``   环境自检（配置 / ffmpeg / 可选依赖 / 模型清单 / 目录可写）
``serve``    起 HTTP 服务（uvicorn）
``mcp``      起 MCP 服务（stdio 或 HTTP）

用法::

    python -m clients.cli analyze demo.mp4
    python -m clients.cli analyze "https://www.bilibili.com/video/BVxxxx" --language zh
    python -m clients.cli ask v1a2b3c4d5e6f "切片长度是怎么定的"
    python -m clients.cli list --json | jq '.items[].video_id'
    python -m clients.cli doctor

退出码：0 成功 / 1 业务失败 / 2 用法或环境问题 / 130 用户中断。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

__all__ = ["main", "EXIT_OK", "EXIT_FAIL", "EXIT_USAGE", "EXIT_INTERRUPT"]


def _runtime_version() -> str:
    """版本号以配置为准（app.version），读不到时退回硬编码占位。"""
    try:
        from core.config import get_app_config  # noqa: PLC0415

        return str(get_app_config().app.version or "0.1.0")
    except Exception:  # noqa: BLE001 - --version 不该因为配置问题失败
        return "0.1.0"

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_INTERRUPT = 130

#: 进度条宽度（字符）。太窄看不清，太宽在 80 列终端折行。
_BAR_WIDTH = 28


# =============================================================================
# 终端输出
# =============================================================================


def _enable_ansi() -> bool:
    """Windows 老控制台默认不认 ANSI 转义，这里尽力打开；失败就退化成纯文本。"""
    if os.name != "nt":
        return True
    try:  # pragma: no cover - 仅在 Windows 上走
        import ctypes  # noqa: PLC0415

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return False
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:  # noqa: BLE001
        return False


class Printer:
    """stdout / stderr 分离 + 可选着色。

    进度条写 stderr：``cli.py analyze x.mp4 | tee log`` 时日志里不会混进
    ``\r[####    ] 42%`` 这种脏东西。
    """

    def __init__(self, *, color: bool = True, quiet: bool = False) -> None:
        self.color_enabled = bool(color) and _enable_ansi()
        self.quiet = quiet
        self._last_len = 0

    # -- 基础 --------------------------------------------------------------
    def _paint(self, text: str, code: str) -> str:
        if not self.color_enabled:
            return text
        return f"\x1b[{code}m{text}\x1b[0m"

    def info(self, text: str = "") -> None:
        if not self.quiet:
            print(text, flush=True)

    def warn(self, text: str) -> None:
        print(self._paint(f"警告: {text}", "33"), file=sys.stderr, flush=True)

    def error(self, text: str) -> None:
        print(self._paint(f"错误: {text}", "31"), file=sys.stderr, flush=True)

    def ok(self, text: str) -> None:
        if not self.quiet:
            print(f"{self._paint('✓', '32')} {text}", flush=True)

    def head(self, text: str) -> None:
        if not self.quiet:
            print(self._paint(text, "1;36"), flush=True)

    def dim(self, text: str) -> str:
        return self._paint(text, "2")

    # -- 进度条 ------------------------------------------------------------
    def progress(self, stage: str, value: float, message: str = "") -> None:
        """``value`` ∈ [0,1]；同一行覆盖刷新。非 TTY 时降级成逐行日志。"""
        if self.quiet:
            return
        ratio = max(0.0, min(1.0, float(value or 0.0)))
        if not sys.stderr.isatty():
            # 重定向到文件时不要 \r，改成每 10% 一行，便于事后看日志
            if int(ratio * 100) % 10 == 0:
                print(f"[{int(ratio * 100):3d}%] {stage} {message}".rstrip(),
                      file=sys.stderr, flush=True)
            return
        filled = int(_BAR_WIDTH * ratio)
        bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
        line = f"\r{stage:<10} [{bar}] {int(ratio * 100):3d}% {message}"
        # 上一行更长时先擦掉尾巴，否则会留下残影
        pad = max(0, self._last_len - len(line))
        sys.stderr.write(line + " " * pad)
        sys.stderr.flush()
        self._last_len = len(line)

    def progress_done(self) -> None:
        if not self.quiet and sys.stderr.isatty() and self._last_len:
            sys.stderr.write("\r" + " " * self._last_len + "\r")
            sys.stderr.flush()
            self._last_len = 0


# =============================================================================
# 序列化工具
# =============================================================================


def _plain(obj: Any) -> Any:
    """把 dataclass / 枚举 / Path / 自定义 ``to_dict`` 统一转成 JSON 友好的结构。"""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_plain(v) for v in obj]
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return _plain(to_dict())
        except TypeError:
            pass
    if is_dataclass(obj) and not isinstance(obj, type):
        return _plain(asdict(obj))
    value = getattr(obj, "value", None)
    if isinstance(value, (str, int, float)):
        return value
    return str(obj)


def _dump(obj: Any) -> str:
    return json.dumps(_plain(obj), ensure_ascii=False, indent=2)


def _fmt_ms(ms: int | float) -> str:
    """毫秒 → ``HH:MM:SS.mmm``（对外统一时间格式）。"""
    total = max(0, int(ms or 0))
    hours, rest = divmod(total, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _human_bytes(size: int | float) -> str:
    value = float(size or 0)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """等宽表格。中文按 2 列宽计算 —— 否则 GBK/CJK 混排会错行。"""

    def width(text: str) -> int:
        return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)

    cells = [[str(c) for c in row] for row in rows]
    widths = [width(h) for h in headers]
    for row in cells:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], width(cell))

    def pad(text: str, target: int) -> str:
        return text + " " * max(0, target - width(text))

    lines = ["  ".join(pad(h, widths[i]) for i, h in enumerate(headers)).rstrip()]
    lines.append("  ".join("-" * w for w in widths))
    for row in cells:
        lines.append(
            "  ".join(pad(cell, widths[i]) for i, cell in enumerate(row)).rstrip()
        )
    return "\n".join(lines)


# =============================================================================
# 后端：本地直连 / 远程 HTTP
# =============================================================================


class BackendError(RuntimeError):
    """后端调用的统一错误（CLI 只处理这一层，不关心是本地还是 HTTP）。"""


class LocalBackend:
    """直接装配 ``Orchestrator``：不起服务、不占端口，适合脚本与批处理。"""

    remote = False

    def __init__(self, *, config_path: str | None = None, load_models: bool = True,
                 printer: Printer | None = None) -> None:
        from core.config import load_app_config  # noqa: PLC0415
        from core.pipeline.orchestrator import Orchestrator  # noqa: PLC0415

        self._printer = printer
        try:
            self.config = load_app_config(config_path)
        except Exception as exc:  # noqa: BLE001
            raise BackendError(f"配置加载失败: {exc}") from exc
        try:
            self.orch = Orchestrator.from_config(self.config, load_models=load_models)
        except Exception as exc:  # noqa: BLE001
            raise BackendError(f"编排器装配失败: {exc}") from exc

    # -- 分析 --------------------------------------------------------------
    def analyze(self, source: str, **opts: Any) -> dict[str, Any]:
        from core.pipeline.orchestrator import AnalyzeOptions  # noqa: PLC0415

        options = AnalyzeOptions(
            language=opts.get("language"),
            skip_vision=bool(opts.get("skip_vision")),
            skip_summary=bool(opts.get("skip_summary")),
            skip_vectors=bool(opts.get("skip_vectors")),
            force=bool(opts.get("force")),
            max_keyframes=opts.get("max_keyframes"),
            title=opts.get("title") or "",
            progress=self._progress,
        )
        report = self.orch.analyze(source, options)
        return _plain(report)

    def _progress(self, stage: str, value: float, message: str) -> None:
        if self._printer is not None:
            self._printer.progress(stage, value, message)

    # -- 读取 --------------------------------------------------------------
    def ask(self, video_id: str, question: str, *, top_k: int | None = None) -> dict[str, Any]:
        result = self.orch.ask(video_id, question, top_k=top_k)
        return _plain(result)

    def list_videos(self, *, limit: int = 20, offset: int = 0,
                    status: str | None = None) -> dict[str, Any]:
        repo = self.orch.repository
        records = repo.list_videos(limit=limit, offset=offset, status=status)
        return {
            "items": [_plain(r) for r in records],
            "total": repo.count_videos(),
        }

    def show(self, video_id: str) -> dict[str, Any]:
        repo = self.orch.repository
        record = repo.find_video(video_id)
        if record is None:
            raise BackendError(f"视频不存在: {video_id}")
        return {
            "video": _plain(record),
            "clips": [_plain(c) for c in repo.get_clips(video_id)],
            "keyframes": [_plain(k) for k in repo.get_keyframes(video_id)],
        }

    def stats(self) -> dict[str, Any]:
        return _plain(self.orch.repository.stats())

    def delete(self, video_id: str) -> dict[str, Any]:
        return _plain(self.orch.delete(video_id))

    def summarize(self, video_id: str) -> str:
        return self.orch.summarize(video_id)

    def chapters(self, video_id: str, *, chapter_seconds: int = 300,
                 min_seconds: int = 300, summarize: bool = True) -> dict[str, Any]:
        rows = self.orch.chapters(
            video_id,
            chapter_seconds=chapter_seconds,
            min_seconds=min_seconds,
            summarize=summarize,
        )
        return {
            "video_id": video_id,
            "needs_chapters": bool(rows),
            "count": len(rows),
            "chapters": [c.to_dict() for c in rows],
        }

    def chapter_report(self, video_id: str, *, chapter_seconds: int = 300,
                       min_seconds: int = 300) -> str:
        return self.orch.chapter_report(
            video_id, chapter_seconds=chapter_seconds, min_seconds=min_seconds
        )

    def close(self) -> None:
        try:
            self.orch.close()
        except Exception:  # noqa: BLE001
            pass


class RemoteBackend:
    """走 ``/api/*``。Obsidian 插件与多端场景下与 CLI 共享同一套语义。"""

    remote = True

    def __init__(self, base_url: str, *, token: str | None = None,
                 timeout: float = 60.0, printer: Printer | None = None,
                 transport: Any | None = None) -> None:
        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise BackendError("远程模式需要 httpx（pip install httpx）") from exc

        self._httpx = httpx
        self.base = base_url.rstrip("/")
        self.timeout = float(timeout)
        self._printer = printer
        self._headers = {"Accept": "application/json"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(base_url=self.base, timeout=self.timeout,
                                    headers=self._headers, transport=transport)

    # -- 辅助 --------------------------------------------------------------
    def _request(self, method: str, path: str, *, raw: bool = False,
                 **kwargs: Any) -> Any:
        httpx = self._httpx
        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise BackendError(f"请求 {method} {self.base}{path} 失败: {exc}") from exc
        if resp.status_code >= 400:
            detail = ""
            try:
                payload = resp.json()
                detail = str(payload.get("detail") or payload)[:300]
            except Exception:  # noqa: BLE001
                detail = resp.text[:300]
            raise BackendError(f"{method} {path} → HTTP {resp.status_code}: {detail}")
        if not resp.content:
            return {}
        if raw:
            return resp.text
        try:
            return resp.json()
        except ValueError as exc:
            raise BackendError(f"{method} {path} 返回非 JSON: {resp.text[:200]}") from exc

    # -- 分析 --------------------------------------------------------------
    def analyze(self, source: str, **opts: Any) -> dict[str, Any]:
        payload = {
            "source": source,
            "language": opts.get("language"),
            "skip_vision": bool(opts.get("skip_vision")),
            "skip_summary": bool(opts.get("skip_summary")),
            "skip_vectors": bool(opts.get("skip_vectors")),
            "force": bool(opts.get("force")),
            "max_keyframes": opts.get("max_keyframes"),
            "title": opts.get("title") or "",
        }
        started = self._request("POST", "/api/videos/analyze",
                               json={k: v for k, v in payload.items() if v is not None})
        task_id = started.get("task_id") or ""
        video_id = started.get("video_id") or ""
        if not task_id:
            return started
        # 轮询（不依赖 SSE：命令行被重定向时 SSE 的 \r 心跳反而添乱）。
        # 轮询时长按配置的性能目标放大：1 小时视频 CPU 上转写就要 40 分钟，
        # deadline 若低于它，任务还在服务端跑、命令行已经报超时了。
        try:
            from core.config import get_app_config  # noqa: PLC0415

            budget = float(get_app_config().targets.asr_1h_cpu_ms) / 1000.0 + 600.0
        except Exception:  # noqa: BLE001 - 配置读不到时退回保守值
            budget = 3600.0
        deadline = time.monotonic() + max(60.0, self.timeout * 20, budget)
        last_progress = -1.0
        while True:
            if time.monotonic() > deadline:
                raise BackendError(f"任务 {task_id} 超时未完成")
            task = self._request("GET", f"/api/tasks/{task_id}")
            status = str(task.get("status") or "")
            stage = str(task.get("stage") or "")
            progress = float(task.get("progress") or 0.0)
            if self._printer is not None and progress - last_progress >= 0.01:
                self._printer.progress(stage or "running", progress,
                                       str(task.get("message") or ""))
                last_progress = progress
            if status in {"done", "failed", "cancelled"}:
                if self._printer is not None:
                    self._printer.progress_done()
                if status != "done":
                    raise BackendError(
                        f"任务 {task_id} {status}: {task.get('error') or '未知原因'}"
                    )
                report = task.get("report")
                if isinstance(report, dict):
                    return report
                # 老版本服务只回 video_id，再去拉一次详情兜底
                if video_id:
                    return {"video_id": video_id, "status": "ready"}
                return task
            time.sleep(0.4)

    # -- 读取 --------------------------------------------------------------
    def ask(self, video_id: str, question: str, *, top_k: int | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"question": question}
        if top_k:
            body["top_k"] = int(top_k)
        return self._request("POST", f"/api/videos/{video_id}/ask", json=body)

    def list_videos(self, *, limit: int = 20, offset: int = 0,
                    status: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return self._request("GET", "/api/videos", params=params)

    def show(self, video_id: str) -> dict[str, Any]:
        video = self._request("GET", f"/api/videos/{video_id}")
        clips = self._request("GET", f"/api/videos/{video_id}/clips")
        frames = self._request("GET", f"/api/videos/{video_id}/keyframes")
        return {
            "video": video.get("video", video) if isinstance(video, dict) else video,
            "clips": clips,
            "keyframes": frames,
        }

    def stats(self) -> dict[str, Any]:
        return self._request("GET", "/api/stats")

    def delete(self, video_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/videos/{video_id}")

    def summarize(self, video_id: str) -> str:
        payload = self._request("POST", f"/api/videos/{video_id}/summary")
        return str(payload.get("summary") or "")

    def chapters(self, video_id: str, *, chapter_seconds: int = 300,
                 min_seconds: int = 300, summarize: bool = True) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/videos/{video_id}/chapters",
            params={
                "chapter_seconds": chapter_seconds,
                "min_seconds": min_seconds,
                "summarize": "true" if summarize else "false",
            },
        )

    def chapter_report(self, video_id: str, *, chapter_seconds: int = 300,
                       min_seconds: int = 300) -> str:
        return self._request(
            "GET",
            f"/api/videos/{video_id}/chapters/report",
            params={"chapter_seconds": chapter_seconds, "min_seconds": min_seconds},
            raw=True,
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass


# =============================================================================
# 子命令实现
# =============================================================================


def cmd_analyze(args: argparse.Namespace, backend: Any, out: Printer) -> int:
    if not args.json:
        out.head(f"分析: {args.source}")
    report = backend.analyze(
        args.source,
        language=args.language,
        skip_vision=args.no_vision,
        skip_summary=args.no_summary,
        skip_vectors=args.no_vectors,
        force=args.force,
        max_keyframes=args.max_keyframes,
        title=args.title,
    )
    out.progress_done()

    if args.json:
        print(_dump(report), flush=True)
        if args.report and report.get("status") == "ready":
            # --json 和 --report 一起用时，报告照写，但只把路径打到 stderr，
            # 保证 stdout 仍是纯 JSON（下游 jq / 管道不会被污染）。
            ns = argparse.Namespace(video_id=str(report.get("video_id") or ""),
                                    out=args.report_out)
            try:
                cmd_export(ns, backend, out)
            except Exception as exc:  # noqa: BLE001 - 导出失败不该让分析结果算失败
                print(f"报告导出失败: {exc}", file=sys.stderr)
        return EXIT_OK if report.get("status") == "ready" else EXIT_FAIL

    status = str(report.get("status") or "")
    video_id = str(report.get("video_id") or "")
    if status != "ready":
        out.error(f"分析未完成（status={status}）")
        for stage in report.get("stages") or []:
            if isinstance(stage, dict) and stage.get("error"):
                out.error(f"  {stage.get('stage')}: {stage.get('error')}")
        return EXIT_FAIL

    out.ok(f"完成 video_id={video_id}")
    out.info(f"  标题      {report.get('title') or '-'}")
    out.info(f"  时长      {_fmt_ms(report.get('duration_ms'))}")
    out.info(f"  语言      {report.get('language') or '-'}"
             f" / 转写来源 {report.get('transcript_source') or '-'}")
    out.info(f"  切片      {report.get('clip_count')} 个"
             f" / 关键帧 {report.get('keyframe_count')} 张"
             f" / 向量 {report.get('vector_count')} 条")
    stages = report.get("stages") or []
    if stages:
        out.info("")
        out.info(_table(
            ["阶段", "状态", "耗时", "说明"],
            [
                [
                    str(s.get("stage")),
                    "跳过" if s.get("skipped") else ("降级" if s.get("degraded")
                                                     else ("ok" if s.get("ok") else "失败")),
                    f"{int(s.get('elapsed_ms') or 0)}ms",
                    str(s.get("message") or s.get("error") or "")[:40],
                ]
                for s in stages if isinstance(s, dict)
            ],
        ))
    degraded = report.get("degraded") or []
    if degraded:
        out.info("")
        out.warn("降级项：")
        for note in degraded:
            out.info(f"  - {note}")
    summary = str(report.get("summary") or "").strip()
    if summary:
        out.info("")
        out.head("总结")
        out.info(summary[:2000])
    out.info("")
    if args.report:
        ns = argparse.Namespace(video_id=video_id, out=args.report_out)
        try:
            cmd_export(ns, backend, out)
        except Exception as exc:  # noqa: BLE001 - 报告导出失败不影响分析结论
            out.warn(f"报告导出失败: {exc}")
    out.info(out.dim(f"下一步: python -m clients.cli ask {video_id} \"你的问题\""))
    return EXIT_OK


def cmd_ask(args: argparse.Namespace, backend: Any, out: Printer) -> int:
    result = backend.ask(args.video_id, args.question, top_k=args.top_k)
    if args.json:
        print(_dump(result), flush=True)
        return EXIT_OK if result.get("answer") else EXIT_FAIL

    answer = str(result.get("answer") or "").strip()
    if not answer:
        out.error("没有得到回答")
        for note in result.get("notes") or []:
            out.info(f"  - {note}")
        return EXIT_FAIL

    out.head(f"问: {result.get('question') or args.question}")
    out.info("")
    out.info(answer)
    refs = result.get("refs") or []
    if refs:
        out.info("")
        out.head("引用证据")
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            out.info(
                f"[{ref.get('index')}] {ref.get('time_range') or ''} "
                f"clip={ref.get('clip_id')} "
                f"({int(ref.get('start_ms') or 0)}–{int(ref.get('end_ms') or 0)}ms)"
            )
            text = str(ref.get("text") or "").strip()
            if text:
                out.info(f"    {text[:160]}")
            for path in ref.get("keyframe_paths") or []:
                out.info(f"    关键帧: {path}")
    meta = [
        f"backend={result.get('backend') or '-'}",
        f"{int(result.get('elapsed_ms') or 0)}ms",
        f"上下文 {int(result.get('context_chars') or 0)} 字",
    ]
    if result.get("degraded"):
        meta.append("降级")
    out.info("")
    out.info(out.dim(" · ".join(meta)))
    return EXIT_OK


def cmd_chapters(args: argparse.Namespace, backend: Any, out: Printer) -> int:
    """长视频按章节切开逐章总结；短视频自动退回单段。"""
    if args.report:
        text = backend.chapter_report(
            args.video_id,
            chapter_seconds=args.chapter_seconds,
            min_seconds=args.min_seconds,
        )
        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            out.info(f"已写入 {path}")
        else:
            out.info(text)
        return EXIT_OK

    data = backend.chapters(
        args.video_id,
        chapter_seconds=args.chapter_seconds,
        min_seconds=args.min_seconds,
        summarize=not args.no_summarize,
    )
    if args.json:
        print(_dump(data), flush=True)
        return EXIT_OK

    chapters = data.get("chapters") or []
    if not chapters:
        out.warn(
            f"视频不足 {args.min_seconds} 秒，没有拆章的必要 —— "
            f"用 ask {args.video_id} \"总结一下\" 或 analyze 的总结区即可"
        )
        return EXIT_OK
    out.head(f"{args.video_id} 共 {len(chapters)} 章（每章约 {args.chapter_seconds} 秒）")
    for ch in chapters:
        out.info("")
        out.head(f"第 {ch.get('index')} 章  {ch.get('time_range') or ''}")
        out.info(str(ch.get("summary") or "").strip())
        meta = [f"{ch.get('backend') or '-'}", f"{ch.get('clip_count')} 段"]
        if ch.get("degraded"):
            meta.append("降级")
        out.info(out.dim(" · ".join(meta)))
    return EXIT_OK


def cmd_list(args: argparse.Namespace, backend: Any, out: Printer) -> int:
    data = backend.list_videos(limit=args.limit, offset=args.offset, status=args.status)
    if args.json:
        print(_dump(data), flush=True)
        return EXIT_OK
    items = data.get("items") or []
    if not items:
        out.info("库里还没有视频。先跑：python -m clients.cli analyze <文件或URL>")
        return EXIT_OK
    out.info(_table(
        ["video_id", "状态", "时长", "切片", "标题"],
        [
            [
                str(item.get("video_id") or ""),
                str(item.get("status") or ""),
                _fmt_ms(item.get("duration_ms")),
                str(item.get("clip_count") if item.get("clip_count") is not None else "-"),
                str(item.get("title") or item.get("display_name") or "")[:36],
            ]
            for item in items if isinstance(item, dict)
        ],
    ))
    out.info("")
    out.info(out.dim(f"共 {data.get('total', len(items))} 条，显示 {len(items)} 条"))
    return EXIT_OK


def cmd_show(args: argparse.Namespace, backend: Any, out: Printer) -> int:
    data = backend.show(args.video_id)
    if args.json:
        print(_dump(data), flush=True)
        return EXIT_OK
    video = data.get("video") or {}
    out.head(f"{video.get('title') or video.get('display_name') or args.video_id}")
    out.info(f"  video_id  {video.get('video_id')}")
    out.info(f"  状态      {video.get('status')}"
             + (f"（{video.get('stage')}）" if video.get("stage") else ""))
    out.info(f"  时长      {_fmt_ms(video.get('duration_ms'))}")
    out.info(f"  分辨率    {video.get('width')}x{video.get('height')} @ {video.get('fps')}fps")
    if video.get("local_path"):
        out.info(f"  文件      {video.get('local_path')}")
    if video.get("source_url"):
        out.info(f"  来源      {video.get('source_url')}")

    clips = data.get("clips") or []
    if clips:
        out.info("")
        out.head(f"切片（{len(clips)}）")
        out.info(_table(
            ["clip_id", "区间", "字符", "关键帧", "正文"],
            [
                [
                    str(c.get("clip_id") or ""),
                    f"{_fmt_ms(c.get('start_ms'))}–{_fmt_ms(c.get('end_ms'))}",
                    str(len(str(c.get("core_text") or c.get("text") or ""))),
                    str(len(c.get("keyframe_ids") or [])),
                    str(c.get("core_text") or c.get("text") or "")[:32],
                ]
                for c in clips if isinstance(c, dict)
            ],
        ))
    frames = data.get("keyframes") or []
    if frames:
        out.info("")
        out.head(f"关键帧（{len(frames)}）")
        out.info(_table(
            ["keyframe_id", "时间", "尺寸", "大小", "清晰度"],
            [
                [
                    str(f.get("keyframe_id") or ""),
                    _fmt_ms(f.get("timestamp_ms")),
                    f"{f.get('width')}x{f.get('height')}",
                    _human_bytes(f.get("size_bytes") or 0),
                    f"{float(f.get('sharpness') or 0):.1f}",
                ]
                for f in frames if isinstance(f, dict)
            ],
        ))
    return EXIT_OK


def cmd_export(args: argparse.Namespace, backend: Any, out: Printer) -> int:
    data = backend.show(args.video_id)
    video = data.get("video") or {}
    clips = [c for c in (data.get("clips") or []) if isinstance(c, dict)]
    frames = [f for f in (data.get("keyframes") or []) if isinstance(f, dict)]

    lines: list[str] = [
        f"# {video.get('title') or video.get('display_name') or args.video_id}",
        "",
        f"- video_id: `{video.get('video_id')}`",
        f"- 时长: {_fmt_ms(video.get('duration_ms'))}",
        f"- 分辨率: {video.get('width')}x{video.get('height')}",
    ]
    if video.get("source_url"):
        lines.append(f"- 来源: {video.get('source_url')}")
    lines += ["", "## 切片时间线", ""]
    for clip in clips:
        start = _fmt_ms(clip.get("start_ms"))
        end = _fmt_ms(clip.get("end_ms"))
        text = str(clip.get("core_text") or clip.get("text") or "").strip()
        lines.append(f"### [{start} – {end}] `{clip.get('clip_id')}`")
        if text:
            lines += ["", text]
        ids = list(clip.get("keyframe_ids") or [])
        if ids:
            lines += ["", "关键帧：" + "、".join(f"`{i}`" for i in ids)]
        lines.append("")
    if frames:
        lines += ["## 关键帧清单", "", "| keyframe_id | 时间 | 尺寸 | 文件 |", "| --- | --- | --- | --- |"]
        for frame in frames:
            lines.append(
                f"| `{frame.get('keyframe_id')}` | {_fmt_ms(frame.get('timestamp_ms'))} "
                f"| {frame.get('width')}x{frame.get('height')} | `{frame.get('path')}` |"
            )
        lines.append("")

    # 需求文档要求报告 ≤5000 字符：超了就在这里截断，别让下游再猜
    limit = 5000
    body = "\n".join(lines)
    truncated = len(body) > limit
    if truncated:
        body = body[:limit] + "\n\n……（报告超 5000 字，已截断）"

    target = Path(args.out) if args.out else (
        ROOT / "data" / "exports" / f"{args.video_id}.md"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    out.ok(f"已导出 {target}（{len(body)} 字{'，已截断' if truncated else ''}）")
    return EXIT_OK


def cmd_stats(args: argparse.Namespace, backend: Any, out: Printer) -> int:
    data = backend.stats()
    if args.json:
        print(_dump(data), flush=True)
        return EXIT_OK

    counts = {
        "videos": data.get("videos"),
        "clips": data.get("clips"),
        "transcript_segments": data.get("transcript_segments"),
        "keyframes": data.get("keyframes"),
        "tasks_pending": data.get("tasks_pending"),
        "tasks_running": data.get("tasks_running"),
    }
    labels = {
        "videos": "视频",
        "clips": "切片",
        "transcript_segments": "转录片段",
        "keyframes": "关键帧",
        "tasks_pending": "排队任务",
        "tasks_running": "运行任务",
    }
    limits = data.get("limits") or {}
    usage = data.get("usage") or {}
    rows = []
    for key, label in labels.items():
        limit = limits.get(f"max_{key}")
        rows.append([
            label,
            str(counts.get(key) if counts.get(key) is not None else 0),
            str(limit) if limit else "-",
            f"{float(usage.get(key) or 0) * 100:.1f}%" if limit else "-",
        ])
    out.head("存储用量")
    out.info(_table(["表", "当前", "上限", "占用"], rows))
    out.info("")
    out.info(f"  关键帧占用  {_human_bytes(data.get('keyframe_bytes') or 0)}"
             f"（上限 {_human_bytes(limits.get('max_keyframe_bytes') or 0)}）")
    out.info(f"  数据库大小  {_human_bytes(data.get('db_bytes') or 0)}")
    return EXIT_OK


def cmd_delete(args: argparse.Namespace, backend: Any, out: Printer) -> int:
    if not args.yes:
        try:
            answer = input(f"确认删除 {args.video_id}（含切片/关键帧/向量）？[y/N] ").strip().lower()
        except EOFError:
            # 管道/非交互环境没有 stdin：按"未确认"处理，而不是抛栈。
            # 要在脚本里真删，请带 -y。
            out.info("非交互环境无法确认，已取消（删除请加 -y）")
            return EXIT_OK
        if answer not in {"y", "yes"}:
            out.info("已取消")
            return EXIT_OK
    result = backend.delete(args.video_id)
    if args.json:
        print(_dump(result), flush=True)
        return EXIT_OK
    out.ok(f"已删除 {args.video_id}")
    vectors = result.get("vectors")
    if isinstance(vectors, dict):
        out.info(f"  向量清理: clip={vectors.get('clip', 0)} frame={vectors.get('frame', 0)}")
    if result.get("vectors_error"):
        out.warn(f"向量清理失败: {result['vectors_error']}")
    return EXIT_OK


# -- doctor -------------------------------------------------------------------

#: 可选依赖 → 缺了会损失什么。doctor 用它把"缺依赖"翻译成人话。
_OPTIONAL_DEPS: tuple[tuple[str, str], ...] = (
    ("cv2", "清晰度评估（Laplacian）；缺失时退化为 numpy/PIL"),
    ("scenedetect", "镜头切分；缺失时退化到 ffmpeg / 等分"),
    ("faster_whisper", "语音转写；缺失时只能用字幕或关键词降级"),
    ("chromadb", "向量检索；缺失时退化为 BM25 字面检索"),
    ("jieba", "中文分词（BM25）；缺失时用单字切分"),
    ("yt_dlp", "在线视频下载"),
    ("transformers", "BGE-M3 / SigLIP2 编码"),
    ("torch", "深度模型底座"),
    ("llama_cpp", "本地 GGUF 推理"),
    ("fastapi", "HTTP 服务"),
    ("gradio", "浏览器 UI"),
    ("httpx", "远程模式"),
)


def _module_available(name: str) -> tuple[bool, str]:
    import importlib.util  # noqa: PLC0415

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False, ""
    if spec is None:
        return False, ""
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "")
        return True, str(version)
    except Exception:  # noqa: BLE001 - 装是装了，import 时炸了也要如实报告
        return True, "（import 失败）"


def cmd_doctor(args: argparse.Namespace, out: Printer) -> int:
    problems = 0
    out.head("知影环境自检")

    out.info("")
    out.info(f"  Python      {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 11):
        out.error("需要 Python 3.11+")
        problems += 1

    # 配置
    try:
        from core.config import load_app_config  # noqa: PLC0415

        cfg = load_app_config(args.config)
        out.ok(f"配置        {cfg.source_file or '（使用内置默认值）'}")
        out.info(f"  项目根    {cfg.root_dir}")
        out.info(f"  数据目录  {cfg.paths.data}")
        out.info(f"  模型目录  {cfg.paths.models}")
        out.info(f"  设备      {cfg.models.device}（CPU 降级 {'开' if cfg.models.allow_cpu_fallback else '关'}）")
        diff = cfg.diff_from_defaults()
        if diff:
            out.info(f"  自定义项  {len(diff)} 处")
            for key, (default, actual) in list(diff.items())[:6]:
                out.info(f"    {key}: {default} → {actual}")
        for name in ("data", "models", "frames", "temp"):
            path = Path(cfg.path(name))
            try:
                path.mkdir(parents=True, exist_ok=True)
                probe = path / ".zhiying-write-test"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
            except OSError as exc:
                out.error(f"目录不可写 {path}: {exc}")
                problems += 1
    except Exception as exc:  # noqa: BLE001
        out.error(f"配置加载失败: {exc}")
        problems += 1
        cfg = None

    # ffmpeg
    out.info("")
    for binary in ("ffmpeg", "ffprobe"):
        found = shutil.which(binary)
        if found:
            out.ok(f"{binary:<11} {found}")
        else:
            out.error(f"{binary:<11} 不在 PATH 上（可设 ZHIYING_FFMPEG / ZHIYING_FFMPEG_DIR）")
            problems += 1
    try:
        from core.media.ffmpeg_utils import probe_media, resolve_binary  # noqa: PLC0415

        out.info(f"  ffmpeg 解析  {resolve_binary('ffmpeg')}")
        out.info(f"  ffprobe 解析 {resolve_binary('ffprobe')}")
        del probe_media
    except Exception as exc:  # noqa: BLE001
        out.warn(f"ffmpeg 封装层不可用: {exc}")

    # 可选依赖
    out.info("")
    out.head("依赖")
    for name, purpose in _OPTIONAL_DEPS:
        ok, version = _module_available(name)
        if ok:
            out.info(f"  ✓ {name:<14} {version[:24]:<24} {purpose}")
        else:
            out.info(f"  · {name:<14} {'（未安装）':<24} {purpose}")

    # 模型清单（只读，不下载）
    out.info("")
    out.head("模型权重（不下载，只看状态）")
    try:
        from core.models.manager import get_model_manager  # noqa: PLC0415

        manager = get_model_manager(args.config_models)
        statuses = manager.status_report()
        for status in statuses:
            state = getattr(status.state, "value", str(status.state))
            mark = "✓" if state == "present" else "·"
            out.info(f"  {mark} {status.name:<24} {state}")
        missing = [s.name for s in statuses if getattr(s.state, "value", "") != "present"]
        if missing:
            out.info("")
            out.info(out.dim("  下载：python scripts/download_models.py --all"))
    except Exception as exc:  # noqa: BLE001
        out.warn(f"模型管理器不可用: {exc}")

    out.info("")
    if problems:
        out.error(f"发现 {problems} 个阻塞问题")
        return EXIT_FAIL
    out.ok("环境看起来是好的")
    return EXIT_OK


# -- serve / mcp --------------------------------------------------------------


def cmd_serve(args: argparse.Namespace, out: Printer) -> int:
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        out.error("缺少 uvicorn（pip install 'uvicorn[standard]'）")
        return EXIT_USAGE
    from core.config import load_app_config  # noqa: PLC0415

    cfg = load_app_config(args.config)
    host = args.host or cfg.api.host
    port = args.port or cfg.api.port
    out.ok(f"启动 HTTP 服务 http://{host}:{port}（文档 /docs）")
    target = "api.main:app" if args.reload else None
    if target is None:
        from api.main import create_app  # noqa: PLC0415

        uvicorn.run(create_app(cfg), host=host, port=port, log_config=None)
    else:
        uvicorn.run(target, host=host, port=port, reload=True, log_config=None)
    return EXIT_OK


def cmd_mcp(args: argparse.Namespace, out: Printer) -> int:
    from api.mcp_server import create_http_router, main as mcp_main  # noqa: PLC0415

    del create_http_router
    argv: list[str] = []
    if args.http:
        argv.append("--http")
    if args.host:
        argv += ["--host", args.host]
    if args.port:
        argv += ["--port", str(args.port)]
    if args.list_tools:
        argv.append("--list-tools")
    if args.no_models:
        argv.append("--no-models")
    if not args.http:
        out.warn("stdio 模式：stdout 被 JSON-RPC 占用，本提示写 stderr")
    return int(mcp_main(argv))


# =============================================================================
# 参数解析
# =============================================================================


def _sub_common() -> argparse.ArgumentParser:
    """子命令也能写全局选项（``list --json`` 而不是 ``--json list``）。

    默认值用 ``SUPPRESS``：否则子解析器的默认值会盖掉顶层已解析的真值，
    形成 ``--json`` 写在子命令前就失效的怪现象（argparse 的经典坑）。
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS)
    common.add_argument("--api", default=argparse.SUPPRESS, metavar="URL")
    common.add_argument("--token", default=argparse.SUPPRESS)
    common.add_argument("--timeout", type=float, default=argparse.SUPPRESS)
    common.add_argument("--no-color", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("--quiet", "-q", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    # 默认装配本地模型（只定位已下载权重，不联网）；--no-models 强制全降级，
    # 用于"只想验证管道能不能跑通"或无权重环境的冒烟测试。
    common.add_argument("--no-models", action="store_true", default=argparse.SUPPRESS)
    return common


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zhiying",
        description="知影 —— 本地优先的视频智能分析助手（CLI）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  zhiying analyze demo.mp4\n"
            "  zhiying analyze \"https://www.bilibili.com/video/BVxxxx\" --language zh\n"
            "  zhiying ask v1a2b3c4d5e6f \"结论的引用时间戳是什么\"\n"
            "  zhiying list --json | jq '.items[].video_id'\n"
            "  zhiying doctor\n"
        ),
    )
    parser.add_argument("--config", default=None, help="配置文件路径（默认 configs/app_config.yaml）")
    parser.add_argument("--api", default=None, metavar="URL",
                        help="走远程服务（如 http://127.0.0.1:8000）；不填则本地直连")
    parser.add_argument("--token", default=None, help="远程模式的 API Token")
    parser.add_argument("--timeout", type=float, default=60.0, help="远程请求超时（秒）")
    parser.add_argument("--no-color", action="store_true", help="关闭 ANSI 颜色")
    parser.add_argument("--quiet", "-q", action="store_true", help="只输出结果，不要进度与提示")
    parser.add_argument("--json", action="store_true", help="输出 JSON（适合脚本）")
    parser.add_argument("--version", action="version",
                        version=f"知影 {_runtime_version()}")

    sub = parser.add_subparsers(dest="command", metavar="<子命令>")
    common = _sub_common()

    p = sub.add_parser("analyze", help="分析本地文件或 URL", parents=[common])
    p.add_argument("source", help="本地路径或视频 URL")
    p.add_argument("--language", default=None, help="强制语言（如 zh / en）")
    p.add_argument("--title", default="", help="自定义标题")
    p.add_argument("--max-keyframes", type=int, default=None, help="关键帧上限（调试用）")
    p.add_argument("--no-vision", action="store_true", help="跳过视觉理解")
    p.add_argument("--no-summary", action="store_true", help="跳过总结")
    p.add_argument("--no-vectors", action="store_true", help="跳过向量入库")
    p.add_argument("--force", action="store_true", help="忽略已有结果重跑")
    p.add_argument(
        "--report",
        action="store_true",
        help="分析完成后自动导出 Markdown 报告（等价于紧接着跑 export）",
    )
    p.add_argument(
        "--report-out",
        default=None,
        metavar="PATH",
        help="配合 --report：指定报告落盘路径（默认 data/exports/<video_id>.md）",
    )
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("ask", help="对已索引视频提问", parents=[common])
    p.add_argument("video_id")
    p.add_argument("question")
    p.add_argument("--top-k", type=int, default=None, help="检索条数")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chapters", help="章节化摘要（长视频逐章总结）", parents=[common])
    p.add_argument("video_id")
    p.add_argument("--chapter-seconds", dest="chapter_seconds", type=int, default=300,
                   help="每章大致时长（秒，默认 300）")
    p.add_argument("--min-seconds", dest="min_seconds", type=int, default=600,
                   help="短于该时长不拆章（秒，默认 600）")
    p.add_argument("--no-summarize", dest="no_summarize", action="store_true",
                   help="只分组不调模型（秒出结果，用于先看骨架）")
    p.add_argument("--report", action="store_true", help="输出完整 markdown 报告")
    p.add_argument("--out", "-o", default=None, help="配合 --report：报告落盘路径")
    p.set_defaults(func=cmd_chapters)

    p = sub.add_parser("list", help="列出库里的视频", parents=[common])
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--status", default=None, help="按状态过滤（ready/failed/…）")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="视频详情", parents=[common])
    p.add_argument("video_id")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("export", help="导出 Markdown 报告", parents=[common])
    p.add_argument("video_id")
    p.add_argument("--out", "-o", default=None, help="输出路径（默认 data/exports/<id>.md）")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("stats", help="存储用量", parents=[common])
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("delete", help="删除视频及其向量", parents=[common])
    p.add_argument("video_id")
    p.add_argument("--yes", "-y", action="store_true", help="不询问")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("doctor", help="环境自检", parents=[common])
    p.add_argument("--models-config", dest="config_models", default=None,
                   help="模型清单配置（默认 configs/models.yaml）")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("serve", help="启动 HTTP 服务", parents=[common])
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--reload", action="store_true", help="开发模式自动重载")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("mcp", help="启动 MCP 服务", parents=[common])
    p.add_argument("--http", action="store_true", help="用 HTTP 而不是 stdio")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--list-tools", action="store_true")
    p.set_defaults(func=cmd_mcp)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:  # Windows 控制台默认 GBK，中文会炸
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    out = Printer(color=not args.no_color, quiet=args.quiet)

    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_USAGE

    # 不需要后端/编排器的子命令先跑掉
    if args.command == "doctor":
        return cmd_doctor(args, out)
    if args.command in {"serve", "mcp"}:
        return int(getattr(args, "func")(args, out))

    backend: Any = None
    try:
        if args.api:
            token = args.token or os.environ.get("ZHIYING_API_TOKEN", "")
            backend = RemoteBackend(args.api, token=token, timeout=args.timeout, printer=out)
        else:
            backend = LocalBackend(
                config_path=args.config,
                load_models=not bool(getattr(args, "no_models", False)),
                printer=out,
            )
        return int(getattr(args, "func")(args, backend, out))
    except BackendError as exc:
        out.progress_done()
        out.error(str(exc))
        return EXIT_FAIL
    except KeyboardInterrupt:
        out.progress_done()
        out.warn("已中断")
        return EXIT_INTERRUPT
    except Exception as exc:  # noqa: BLE001 - 兜底：CLI 不该把 traceback 摔给用户
        out.progress_done()
        out.error(f"{type(exc).__name__}: {exc}")
        if os.environ.get("ZHIYING_DEBUG"):
            raise
        return EXIT_FAIL
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    raise SystemExit(main())
