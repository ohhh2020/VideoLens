"""clients 层自检（CLI 主体 + 远程后端），零第三方测试框架。

不去真跑一次 ``analyze``（那要模型权重、要几分钟），而是把 CLI 拆成两层来测：

1. **纯函数层**：``_fmt_ms`` / ``_human_bytes`` / ``_table`` / ``_plain`` / ``_dump``
   / ``describe_stage`` —— 表格对齐、时间格式、JSON 序列化是"输出契约"，
   坏了用户立刻能看见，必须有断言锁住。
2. **命令层**：用 ``FakeBackend`` 替换真实后端，逐个 ``cmd_*`` 跑一遍，断言
   stdout 内容与退出码；``main()`` 的分发用 ``patch`` 注入。
3. **RemoteBackend**：用 ``httpx.MockTransport`` 造 200/404/500/非 JSON 响应，
   确认错误被翻译成 ``BackendError``（CLI 只在 ``except BackendError`` 里收口）。

跑法::

    & "C:\\Program Files\\Python311\\python.exe" tests/test_cli.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clients import cli as Z  # noqa: E402
from clients.gradio_ui import _parse_video_choice, describe_stage  # noqa: E402

WORK = ROOT / ".tmp_test" / "cli"
PASS = 0
FAIL = 0


def check(label: str, cond: bool, extra: Any = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label} {extra}")


@contextlib.contextmanager
def captured() -> Iterator[tuple[str, str]]:
    """同时抓 stdout/stderr（cmd_* 的进度与警告写 stderr）。"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield out, err  # type: ignore[misc]


def run_cli(argv: Sequence[str]) -> tuple[int, str, str]:
    """解析参数 + 执行，返回 (退出码, stdout, stderr)。"""
    args = Z.build_parser().parse_args(list(argv))
    out = Z.Printer(color=False, quiet=args.quiet)
    with captured() as (stdout, stderr):
        code = int(getattr(args, "func")(args, FAKE, out))
    return code, stdout.getvalue(), stderr.getvalue()


# ---------------------------------------------------------------------------
# FakeBackend: 完全照着 LocalBackend 的公开方法签
# ---------------------------------------------------------------------------


REPORT = {
    "video_id": "vTEST0001",
    "title": "示例视频",
    "status": "ready",
    "stages": [
        {"stage": "parse", "ok": True, "skipped": False, "degraded": False,
         "elapsed_ms": 5, "message": "本地文件", "error": ""},
        {"stage": "transcribe", "ok": True, "skipped": False, "degraded": True,
         "elapsed_ms": 120, "message": "字幕优先", "error": ""},
        {"stage": "vision", "ok": True, "skipped": True, "degraded": False,
         "elapsed_ms": 0, "message": "已跳过", "error": ""},
    ],
    "clip_count": 2,
    "keyframe_count": 7,
    "segment_count": 4,
    "vector_count": 9,
    "transcript_source": "subtitle",
    "language": "zh",
    "duration_ms": 40_000,
    "summary": "这段视频讲了三件事：切片、抽帧、检索。",
    "degraded": ["未提供 ModelManager，视觉理解降级为纯文本"],
    "errors": [],
}

ASK = {
    "question": "切片长度是怎么定的",
    "answer": "根据 [1]，切片默认 10–30 秒。[2] 补充了重叠。",
    "refs": [
        {"index": 1, "clip_id": "vTEST0001-c0000", "video_id": "vTEST0001",
         "start_ms": 0, "end_ms": 20_100, "time_range": "00:00:00.000–00:00:20.100",
         "keyframe_paths": ["D:/frames/kf1.jpg"], "text": "切片默认十到三十秒。"},
        {"index": 2, "clip_id": "vTEST0001-c0001", "video_id": "vTEST0001",
         "start_ms": 20_100, "end_ms": 40_000, "time_range": "00:00:20.100–00:00:40.000",
         "keyframe_paths": [], "text": "重叠 1.5 秒。"},
    ],
    "video_id": "vTEST0001",
    "elapsed_ms": 42,
    "backend": "keywords",
    "degraded": True,
    "notes": [],
    "context_chars": 321,
    "candidates": 5,
}

VIDEOS = {
    "items": [
        {"video_id": "vTEST0001", "status": "ready", "duration_ms": 40_000,
         "clip_count": 2, "title": "示例视频"},
        {"video_id": "vTEST0002", "status": "failed", "duration_ms": 0,
         "clip_count": None, "title": ""},
    ],
    "total": 2,
}

SHOW = {
    "video": {"video_id": "vTEST0001", "title": "示例视频", "status": "ready",
              "stage": "", "duration_ms": 40_000, "width": 1280, "height": 720,
              "fps": 25.0, "local_path": "D:/videos/a.mp4", "source_url": "",
              "display_name": "a.mp4"},
    "clips": [
        {"clip_id": "vTEST0001-c0000", "start_ms": 0, "end_ms": 20_100,
         "core_text": "切片默认十到三十秒。", "keyframe_ids": ["kf-1", "kf-2"]},
        {"clip_id": "vTEST0001-c0001", "start_ms": 20_100, "end_ms": 40_000,
         "core_text": "重叠 1.5 秒。", "keyframe_ids": ["kf-3"]},
    ],
    "keyframes": [
        {"keyframe_id": "kf-1", "timestamp_ms": 5_000, "width": 1280, "height": 720,
         "size_bytes": 105_082, "sharpness": 87.2, "path": "D:/frames/kf-1.jpg"},
        {"keyframe_id": "kf-2", "timestamp_ms": 12_000, "width": 1280, "height": 720,
         "size_bytes": 98_000, "sharpness": 91.0, "path": "D:/frames/kf-2.jpg"},
    ],
}

STATS = {
    "videos": 3, "clips": 12, "transcript_segments": 40, "keyframes": 21,
    "tasks_pending": 1, "tasks_running": 0,
    "keyframe_bytes": 21 * 100_000, "db_bytes": 1_048_576,
    "limits": {"max_videos": 200, "max_clips": 50_000,
               "max_transcript_segments": 500_000, "max_keyframes": 100_000,
               "max_keyframe_bytes": 21_474_836_480},
    "usage": {"videos": 0.015, "clips": 0.00024, "transcript_segments": 0.00008,
              "keyframes": 0.00021},
}


class FakeBackend:
    """记录调用参数，返回固定数据。"""

    remote = False

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.closed = False

    def analyze(self, source: str, **opts: Any) -> dict[str, Any]:
        self.calls.append(("analyze", {"source": source, **opts}))
        return dict(REPORT)

    def ask(self, video_id: str, question: str, *, top_k: int | None = None) -> dict[str, Any]:
        self.calls.append(("ask", {"video_id": video_id, "question": question, "top_k": top_k}))
        return dict(ASK)

    def list_videos(self, *, limit: int = 20, offset: int = 0,
                    status: str | None = None) -> dict[str, Any]:
        self.calls.append(("list_videos", {"limit": limit, "offset": offset, "status": status}))
        return dict(VIDEOS)

    def show(self, video_id: str) -> dict[str, Any]:
        self.calls.append(("show", video_id))
        return dict(SHOW)

    def stats(self) -> dict[str, Any]:
        self.calls.append(("stats", None))
        return dict(STATS)

    def delete(self, video_id: str) -> dict[str, Any]:
        self.calls.append(("delete", video_id))
        return {"video_id": video_id, "vectors": {"clip": 6, "frame": 7}, "db": ["a.jpg"]}

    def summarize(self, video_id: str) -> str:
        return "总结"

    def close(self) -> None:
        self.closed = True


FAKE = FakeBackend()


# ---------------------------------------------------------------------------
# 1. 纯函数
# ---------------------------------------------------------------------------


def test_format_helpers() -> None:
    print("\n[1] 格式与序列化")
    check("0ms", Z._fmt_ms(0) == "00:00:00.000", Z._fmt_ms(0))
    check("毫秒补零", Z._fmt_ms(7) == "00:00:00.007", Z._fmt_ms(7))
    check("跨越分钟", Z._fmt_ms(61_500) == "00:01:01.500", Z._fmt_ms(61_500))
    check("跨越小时", Z._fmt_ms(3_723_456) == "01:02:03.456", Z._fmt_ms(3_723_456))
    check("负数钳到 0", Z._fmt_ms(-5) == "00:00:00.000", Z._fmt_ms(-5))
    check("None 当 0", Z._fmt_ms(None) == "00:00:00.000")  # type: ignore[arg-type]

    check("字节", Z._human_bytes(512) == "512 B", Z._human_bytes(512))
    check("KiB", Z._human_bytes(2048) == "2.0 KiB", Z._human_bytes(2048))
    check("MiB", Z._human_bytes(5 * 1024 * 1024) == "5.0 MiB")
    check("GiB", Z._human_bytes(21_474_836_480) == "20.0 GiB", Z._human_bytes(21_474_836_480))

    table = Z._table(["名", "值"], [["a", "1"], ["很长的名字", "22"]])
    lines = table.splitlines()
    check("表格三行", len(lines) == 4, lines)
    check("表头对齐含中文", lines[0].startswith("名"), lines[0])
    # 中文按 2 列宽：'很长的名字' = 10 列，'名' = 2 列 → 第一列宽 10
    check("中文宽度按 2 计", lines[3].startswith("很长的名字"), lines[3])
    check("分隔线存在", set(lines[1].replace(" ", "")) == {"-"}, lines[1])

    payload = Z._plain({"path": Path("a/b"), "items": (1, 2), "obj": REPORT})
    check("Path→str", payload["path"] == str(Path("a/b")), payload["path"])
    check("tuple→list", payload["items"] == [1, 2])
    check("嵌套 dict 保留", payload["obj"]["video_id"] == "vTEST0001")
    text = Z._dump(payload)
    check("JSON 可解析", isinstance(json.loads(text), dict))
    check("中文不转义", "示例视频" in text, text[:80])
    check("枚举→value", Z._plain({"s": __import__("enum").Enum("E", "A").A})["s"] == 1)

    check("阶段翻译 asr", describe_stage("transcribing") == "语音转写")
    check("阶段未知原样", describe_stage("whatever") == "whatever")
    check("阶段空", describe_stage("") == "运行中")
    check("video 选择解析", _parse_video_choice("v1 · 标题") == "v1")
    check("video 选择无标题", _parse_video_choice("v1") == "v1")


def test_printer() -> None:
    print("\n[2] Printer")
    printer = Z.Printer(color=False)
    with captured() as (out, err):
        printer.info("普通")
        printer.ok("好了")
        printer.head("标题")
        printer.warn("注意")
        printer.error("坏了")
        printer.dim("暗")
        printer.progress("asr", 0.5, "转写中")
        printer.progress("asr", 1.0, "完成")
        printer.progress_done()
    check("info 走 stdout", "普通" in out.getvalue(), out.getvalue())
    check("ok 带勾", "✓ 好了" in out.getvalue(), out.getvalue())
    check("warn 走 stderr", "警告: 注意" in err.getvalue(), err.getvalue())
    check("error 走 stderr", "错误: 坏了" in err.getvalue(), err.getvalue())
    # 非 TTY：progress 降级成按 10% 逐行
    check("非 TTY 进度按 10% 打点", "50%" in err.getvalue() and "[100%]" in err.getvalue(),
          err.getvalue()[-120:])
    check("非 TTY 不写 \\r", "\r" not in err.getvalue())

    quiet = Z.Printer(color=False, quiet=True)
    with captured() as (out, err):
        quiet.info("不该出现")
        quiet.ok("也不该")
        quiet.progress("asr", 0.3, "x")
    check("quiet 静默", out.getvalue() == "" and err.getvalue() == "",
          (out.getvalue(), err.getvalue()))


# ---------------------------------------------------------------------------
# 2. 参数解析
# ---------------------------------------------------------------------------


def test_parser() -> None:
    print("\n[3] 参数解析")
    p = Z.build_parser()
    args = p.parse_args(["analyze", "a.mp4", "--language", "zh", "--no-vision", "--force"])
    check("analyze 位置参数", args.source == "a.mp4", args.source)
    check("analyze language", args.language == "zh")
    check("analyze 开关", args.no_vision and args.force)
    check("analyze func", args.func is Z.cmd_analyze)

    args = p.parse_args(["ask", "v1", "问题在这", "--top-k", "3"])
    check("ask 两位置参数", args.video_id == "v1" and args.question == "问题在这")
    check("ask top-k", args.top_k == 3)

    args = p.parse_args(["list", "--limit", "5", "--status", "ready", "--json"])
    check("list 参数", (args.limit, args.status, args.json) == (5, "ready", True))

    args = p.parse_args(["--api", "http://x:1", "--token", "t", "stats"])
    check("全局 --api", args.api == "http://x:1")
    check("全局 --token", args.token == "t")

    args = p.parse_args(["mcp", "--http", "--port", "9000"])
    check("mcp 参数", args.http and args.port == 9000)

    args = p.parse_args(["export", "v1", "-o", "out.md"])
    check("export -o", args.out == "out.md")

    try:
        p.parse_args(["ask"])
        check("缺参数应报错", False)
    except SystemExit as exc:
        check("缺参数退出码 2", exc.code == 2, exc.code)


# ---------------------------------------------------------------------------
# 3. 子命令
# ---------------------------------------------------------------------------


def test_analyze_cmd() -> None:
    print("\n[4] analyze")
    code, out, err = run_cli(["analyze", "demo.mp4", "--language", "zh", "--no-summary"])
    check("退出码 0", code == 0, code)
    check("打印 video_id", "vTEST0001" in out, out[:200])
    check("打印时长", "00:00:40.000" in out, out[:400])
    check("阶段表含转写", "转写" in out or "transcribe" in out)
    check("降级提示", "降级" in out, out[-300:])
    check("总结段落", "这段视频讲了三件事" in out, out[-400:])
    call = FAKE.calls[-1]
    check("透传 language", call[1]["language"] == "zh", call)
    check("透传 skip_summary", call[1]["skip_summary"] is True)
    check("下一步提示", "clients.cli ask" in out, out[-120:])

    code, out, err = run_cli(["analyze", "demo.mp4", "--json"])
    check("json 模式退出码", code == 0, code)
    payload = json.loads(out)
    check("json 含 video_id", payload["video_id"] == "vTEST0001")
    check("json 含 stages", len(payload["stages"]) == 3)

    # 失败报告 → 退出码 1
    saved = dict(REPORT)
    try:
        REPORT["status"] = "failed"
        REPORT["stages"] = [{"stage": "probe", "ok": False, "skipped": False,
                             "degraded": False, "elapsed_ms": 3,
                             "message": "", "error": "ffprobe 打不开文件"}]
        code, out, err = run_cli(["analyze", "bad.mp4"])
        check("失败退出码 1", code == 1, code)
        check("失败打印阶段错误", "ffprobe 打不开文件" in err, err)
    finally:
        REPORT.clear()
        REPORT.update(saved)


def test_ask_cmd() -> None:
    print("\n[5] ask")
    code, out, err = run_cli(["ask", "vTEST0001", "切片长度是怎么定的"])
    check("退出码 0", code == 0, code)
    check("打印回答", "根据 [1]" in out, out[:200])
    check("打印引用区间", "00:00:00.000–00:00:20.100" in out, out[-500:])
    check("打印 clip_id", "vTEST0001-c0000" in out)
    check("打印毫秒", "0–20100ms" in out, out[-400:])
    check("打印关键帧路径", "D:/frames/kf1.jpg" in out)
    check("打印后端", "backend=keywords" in out, out[-200:])
    check("打印降级标记", "降级" in out)
    call = FAKE.calls[-1]
    check("top_k 默认 None", call[1]["top_k"] is None)

    code, out, err = run_cli(["ask", "v1", "q", "--top-k", "4", "--json"])
    payload = json.loads(out)
    check("json refs 两条", len(payload["refs"]) == 2)
    check("透传 top_k", FAKE.calls[-1][1]["top_k"] == 4)

    saved = dict(ASK)
    try:
        ASK["answer"] = ""
        ASK["notes"] = ["没有可用的上下文"]
        code, out, err = run_cli(["ask", "v1", "q"])
        check("空回答退出码 1", code == 1, code)
        check("打印 notes", "没有可用的上下文" in out, out)
    finally:
        ASK.clear()
        ASK.update(saved)


def test_list_show_stats() -> None:
    print("\n[6] list / show / stats")
    code, out, err = run_cli(["list"])
    check("list 退出码", code == 0, code)
    check("list 表头", "video_id" in out and "状态" in out, out[:200])
    check("list 含两条", out.count("vTEST000") == 2, out)
    check("list 总数", "共 2 条" in out, out[-80:])
    check("list 空值显示 -", " - " in out or "| -" in out or "\t-" in out or " -" in out,
          out)
    code, out, _ = run_cli(["list", "--json"])
    check("list json", json.loads(out)["total"] == 2)

    code, out, err = run_cli(["show", "vTEST0001"])
    check("show 退出码", code == 0, code)
    check("show 标题", "示例视频" in out, out[:200])
    check("show 分辨率", "1280x720" in out, out[:400])
    check("show 切片表", "vTEST0001-c0000" in out)
    check("show 关键帧表", "kf-1" in out)
    check("show 关键帧大小", "102.6 KiB" in out, out[-400:])

    code, out, _ = run_cli(["stats"])
    check("stats 退出码", code == 0, code)
    check("stats 视频数", "视频" in out and "3" in out, out[:300])
    check("stats 占用百分比", "%" in out, out)
    check("stats 关键帧占用 GiB", "GiB" in out, out[-200:])
    code, out, _ = run_cli(["stats", "--json"])
    check("stats json limits", json.loads(out)["limits"]["max_videos"] == 200)


def test_export_delete() -> None:
    print("\n[7] export / delete")
    target = WORK / "report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    code, out, err = run_cli(["export", "vTEST0001", "--out", str(target)])
    check("export 退出码", code == 0, code)
    check("文件已写", target.is_file())
    body = target.read_text(encoding="utf-8")
    check("frontmatter 式头", body.startswith("# 示例视频"), body[:40])
    check("含 video_id", "vTEST0001" in body)
    check("含时间线", "00:00:00.000" in body, body[:400])
    check("含关键帧清单", "kf-2" in body, body[-400:])
    check("报告未超 5000", len(body) <= 5000 + 40, len(body))

    code, out, err = run_cli(["delete", "vTEST0001", "--yes"])
    check("delete 退出码", code == 0, code)
    check("delete 提示向量", "clip=6" in out, out)
    check("delete 调用", FAKE.calls[-1] == ("delete", "vTEST0001"), FAKE.calls[-1])
    code, out, _ = run_cli(["delete", "vTEST0001", "--yes", "--json"])
    check("delete json", "vectors" in json.loads(out))


# ---------------------------------------------------------------------------
# 4. main() 分发
# ---------------------------------------------------------------------------


def test_main_dispatch() -> None:
    print("\n[8] main 分发")
    with captured() as (out, err):
        code = Z.main([])
    check("无子命令 → 2", code == Z.EXIT_USAGE, code)
    check("无子命令打帮助", "子命令" in out.getvalue() or "usage" in out.getvalue(),
          out.getvalue()[:200])

    created: list[str] = []

    class FakeLocal:
        def __init__(self, *, config_path: str | None = None,
                     load_models: bool = False, printer: Any = None) -> None:
            created.append("local")
            self.printer = printer

        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"不该调用 {name}")

        def list_videos(self, **kwargs: Any) -> dict[str, Any]:
            return dict(VIDEOS)

        def close(self) -> None:
            created.append("closed")

    original = Z.LocalBackend
    Z.LocalBackend = FakeLocal  # type: ignore[assignment,misc]
    try:
        with captured() as (out, err):
            code = Z.main(["list"])
    finally:
        Z.LocalBackend = original
    check("本地后端被装配", created == ["local", "closed"], created)
    check("list 走通", code == 0 and "vTEST0001" in out.getvalue(), code)

    # BackendError → 退出码 1 且不吐 traceback
    class Boom:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise Z.BackendError("服务不可用")

    Z.LocalBackend = Boom  # type: ignore[assignment,misc]
    try:
        with captured() as (out, err):
            code = Z.main(["stats"])
    finally:
        Z.LocalBackend = original
    check("后端错误 → 1", code == Z.EXIT_FAIL, code)
    check("错误走 stderr", "服务不可用" in err.getvalue(), err.getvalue())
    check("无 traceback", "Traceback" not in err.getvalue())

    # KeyboardInterrupt → 130
    class Interrupt:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise KeyboardInterrupt

    Z.LocalBackend = Interrupt  # type: ignore[assignment,misc]
    try:
        with captured() as (out, err):
            code = Z.main(["stats"])
    finally:
        Z.LocalBackend = original
    check("中断 → 130", code == Z.EXIT_INTERRUPT, code)
    check("中断有提示", "已中断" in err.getvalue(), err.getvalue())


# ---------------------------------------------------------------------------
# 5. RemoteBackend（httpx.MockTransport）
# ---------------------------------------------------------------------------


def test_remote_backend() -> None:
    print("\n[9] RemoteBackend")
    import httpx

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path == "/api/health":
            return httpx.Response(200, json={"status": "ok", "orchestrator": True})
        if path == "/api/videos/analyze":
            return httpx.Response(202, json={"task_id": "t1", "video_id": "vT"})
        if path == "/api/tasks/t1":
            return httpx.Response(200, json={"status": "done", "progress": 1.0,
                                             "stage": "done", "report": REPORT})
        if path == "/api/videos/vT/ask":
            return httpx.Response(200, json=ASK)
        if path == "/api/videos":
            return httpx.Response(200, json=VIDEOS)
        if path == "/api/stats":
            return httpx.Response(200, json=STATS)
        if path.startswith("/api/videos/missing"):
            return httpx.Response(404, json={"detail": "视频不存在"})
        if path.startswith("/api/videos/boom"):
            return httpx.Response(500, json={"detail": "内部错误"})
        if path == "/api/videos/notjson":
            return httpx.Response(200, text="<html>oops</html>",
                                  headers={"content-type": "text/html"})
        return httpx.Response(404, json={"detail": f"no route {path}"})

    transport = httpx.MockTransport(handler)
    printer_holder = Z.Printer(color=False, quiet=True)
    backend = Z.RemoteBackend("http://test.local", token="secret",
                              transport=transport, printer=printer_holder)
    try:
        check("health 正常", backend._request("GET", "/api/health")["status"] == "ok")
        check("带 Authorization", seen[-1].headers.get("authorization") == "Bearer secret",
              seen[-1].headers)

        report = backend.analyze("demo.mp4", language="zh", skip_vision=True)
        check("analyze 轮询到 report", report["video_id"] == "vTEST0001", report)
        posts = [r for r in seen if r.url.path == "/api/videos/analyze"]
        body = json.loads(posts[-1].content.decode("utf-8"))
        check("分析请求体带 language", body["language"] == "zh", body)
        check("None 字段被剔除", "max_keyframes" not in body, body)

        ask = backend.ask("vT", "问题", top_k=3)
        check("ask 走通", ask["answer"].startswith("根据 [1]"))
        body = json.loads([r for r in seen if r.url.path.endswith("/ask")][-1].content.decode())
        check("ask 带 top_k", body["top_k"] == 3, body)

        listing = backend.list_videos(limit=5)
        check("list 参数", "limit=5" in str(seen[-1].url), str(seen[-1].url))

        try:
            backend.show("missing")
            check("404 应抛错", False)
        except Z.BackendError as exc:
            check("404 翻译成 BackendError", "404" in str(exc) and "视频不存在" in str(exc),
                  str(exc))

        try:
            backend.ask("boom", "q")
            check("500 应抛错", False)
        except Z.BackendError as exc:
            check("500 翻译成 BackendError", "500" in str(exc), str(exc))

        try:
            backend._request("GET", "/api/videos/notjson")
            check("非 JSON 应抛错", False)
        except Z.BackendError as exc:
            check("非 JSON 翻译成 BackendError", "非 JSON" in str(exc), str(exc))
    finally:
        backend.close()

    # 连不上 → BackendError
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    broken = Z.RemoteBackend("http://127.0.0.1:1", transport=httpx.MockTransport(refuse))
    try:
        try:
            broken.stats()
            check("连不上应抛错", False)
        except Z.BackendError as exc:
            check("连接错误翻译", "请求 GET" in str(exc) and "失败" in str(exc), str(exc))
    finally:
        broken.close()

    # 任务失败 → BackendError
    def failing(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/videos/analyze":
            return httpx.Response(202, json={"task_id": "t2"})
        return httpx.Response(200, json={"status": "failed", "progress": 0.3,
                                        "error": "ffprobe 失败"})

    flaky = Z.RemoteBackend("http://test.local", transport=httpx.MockTransport(failing))
    try:
        try:
            flaky.analyze("x.mp4")
            check("任务失败应抛错", False)
        except Z.BackendError as exc:
            check("任务失败翻译", "failed" in str(exc) and "ffprobe" in str(exc), str(exc))
    finally:
        flaky.close()


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    test_format_helpers()
    test_printer()
    test_parser()
    test_analyze_cmd()
    test_ask_cmd()
    test_list_show_stats()
    test_export_delete()
    test_main_dispatch()
    test_remote_backend()
    print(f"\nPASS={PASS} FAIL={FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
