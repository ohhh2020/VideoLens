"""api 层自检（REST + SSE + MCP），零第三方测试框架。

用 ``fastapi.testclient.TestClient``（httpx 驱动）跑真实 ASGI，不启端口。
被测的 ``Orchestrator`` 用与 ``test_pipeline.py`` 相同的假组件装配，
因此本测试**不依赖模型权重**，只依赖 ffmpeg（抽帧）与 chromadb（向量）。

覆盖：
  1. /api/health 与根路径探活
  2. 分析任务：提交 → 轮询 → 完成；SSE 进度帧
  3. 视频列表 / 详情 / 切片 / 关键帧 / 关键帧图片字节
  4. 问答（带引用与毫秒时间戳）
  5. 参数校验（422）、404、鉴权（401/403/200）
  6. MCP：initialize / tools/list / tools/call / 错误码 / HTTP 端点
  7. 删除与容量统计

跑法::

    & "C:\\Program Files\\Python311\\python.exe" tests/test_api.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import ensure as ensure_fixtures  # noqa: E402

WORK = ROOT / ".tmp_test" / "api"
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


# ---------------------------------------------------------------------------
# 假组件（与 test_pipeline.py 同形）
# ---------------------------------------------------------------------------

DIM = 256


def _embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for token in (text or "").lower().split() or [""]:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)
        vec[h % DIM] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


class FakeAsr:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio_path: Any, video_id: str, subtitles: Sequence[Any] = (),
                   duration_ms: int | None = None, allow_subtitle: bool | None = None) -> Any:
        self.calls += 1
        segments = [
            {"index": 0, "start_ms": 0, "end_ms": 8200, "text": "第一段讲的是视频解析与音频抽取。"},
            {"index": 1, "start_ms": 8200, "end_ms": 20100, "text": "第二段说明切片长度默认十到三十秒。"},
            {"index": 2, "start_ms": 20100, "end_ms": 30500, "text": "第三段介绍关键帧抽取与清晰度过滤。"},
            {"index": 3, "start_ms": 30500, "end_ms": 40000, "text": "最后一段讲向量检索与融合排序。"},
        ]
        return SimpleNamespace(
            video_id=video_id,
            segments=[SimpleNamespace(**s) for s in segments],
            language="zh", language_probability=0.98,
            duration_ms=duration_ms or 40000, source="asr",
            model_name="fake-whisper", device="cpu", compute_type="int8",
            elapsed_ms=10, subtitle_path=None, dropped_segments=0,
            degraded=False, warnings=[],
        )


class FakeEmbedder:
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _embed(text)

    def embed_images(self, images: Sequence[Any]) -> list[list[float]]:
        return [_embed(f"image {i}") for i, _ in enumerate(images)]

    def embed_cross_modal(self, text: str) -> list[float]:
        return _embed(text)

    def close(self) -> None:
        pass


class FakeVl:
    def describe(self, image_path: Any, prompt: str | None = None, prompt_key: str | None = None,
                 extra_vars: dict[str, Any] | None = None) -> Any:
        return SimpleNamespace(image_path=str(image_path), description="彩色测试图，含色块与网格",
                               ocr_text="TC 00:00:05", degraded=False, error="")


class FakeLlm:
    def complete(self, messages: Sequence[Any], **kwargs: Any) -> Any:
        return SimpleNamespace(
            text="根据 [1]，切片长度默认 10-30 秒；[2] 补充了 1.5 秒重叠。",
            backend="fake", model="fake-llm", degraded=False,
            prompt_tokens=10, completion_tokens=20, elapsed_ms=5, finish_reason="stop",
        )

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 装配
# ---------------------------------------------------------------------------


def build_app(cfg: Any) -> tuple[Any, dict[str, Any]]:
    from api.main import create_app
    from core.pipeline import ContextBuilder, Orchestrator
    from core.rag.chunker import Chunker
    from core.rag.vector_store import VectorStore
    from core.storage.repository import SqliteRepository

    repo = SqliteRepository(WORK / "zhiying.db")
    vectors = VectorStore(WORK / "chroma")
    orch = Orchestrator(
        repository=repo,
        vectors=vectors,
        embedder=FakeEmbedder(),
        asr=FakeAsr(),
        scene_detector=None,
        chunker=Chunker(),
        vl=FakeVl(),
        llm=FakeLlm(),
        context_builder=ContextBuilder(None),
        config=cfg,
    )
    app = create_app(cfg, orchestrator=orch, load_models=False, serve_ui=False)
    return app, {"repo": repo, "vectors": vectors, "orch": orch}


def main() -> int:
    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:
        print(f"缺少依赖: {exc}\n  pip install fastapi httpx")
        return 2

    from core.config import load_app_config

    demo, _low = ensure_fixtures()
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)

    cfg = load_app_config(
        root=str(ROOT),
        overrides={
            "paths.audio": str(WORK / "audio"),
            "paths.frames": str(WORK / "frames"),
            "paths.sqlite": str(WORK / "zhiying.db"),
            "paths.chroma": str(WORK / "chroma"),
            "api.sse_heartbeat_s": 1,
        },
    )
    app, parts = build_app(cfg)

    from core.media.scene_detect import SceneDetectConfig, SceneDetector

    parts["orch"]._scene = SceneDetector(SceneDetectConfig(frames_per_scene=2))

    # 鉴权策略：TestClient 的 client host 是 "testclient"，不在回环白名单里
    # （测试桩绝不进生产白名单），因此全程用 token 模式；无凭据客户端专测 401/403。
    token_env = cfg.api.auth_token_env
    saved = os.environ.pop(token_env, None)
    os.environ[token_env] = "s3cret-token"

    client = TestClient(app)
    client.headers.update({"X-API-Token": "s3cret-token"})
    noauth = TestClient(app)

    print("\n=== 1. 健康检查 ===")
    r = client.get("/api/health")
    check("health 200", r.status_code == 200, r.status_code)
    payload = r.json()
    check("health 报告 orchestrator 就绪", payload["orchestrator"] is True, payload)
    # 本测试全程设了 token（见下），health 应如实报告 token 模式
    check("health auth=token", payload["auth"] == "token", payload["auth"])
    check("根路径探活", client.get("/health").status_code == 200)

    print("\n=== 2. 提交分析任务 ===")
    r = client.post("/api/videos/analyze", json={"source": str(demo), "title": "接口测试片"})
    check("analyze 202", r.status_code == 202, f"{r.status_code} {r.text[:200]}")
    job = r.json()
    task_id = job["task_id"]
    check("返回 task_id", task_id.startswith("t"), task_id)
    check("初始状态可识别", job["status"] in {"queued", "running"}, job["status"])

    print("\n=== 3. 轮询到完成 ===")
    deadline = time.monotonic() + 180
    final: dict[str, Any] = {}
    while time.monotonic() < deadline:
        r = client.get(f"/api/tasks/{task_id}")
        final = r.json()
        if final["status"] in {"done", "failed"}:
            break
        time.sleep(0.5)
    check("任务已结束", final.get("status") == "done", final.get("status"))
    check("无错误", not final.get("error"), final.get("error"))
    report = final.get("report") or {}
    check("报告 ok", report.get("ok") is True, report.get("status"))
    video_id = report.get("video_id") or ""
    check("有 video_id", bool(video_id), video_id)
    check("切片 2 个", report.get("clip_count") == 2, report.get("clip_count"))
    check("关键帧 >= 4", (report.get("keyframe_count") or 0) >= 4, report.get("keyframe_count"))
    check("阶段记录 9 条", len(report.get("stages") or []) == 9, len(report.get("stages") or []))
    check("总结非空", bool((report.get("summary") or "").strip()))

    print("\n=== 4. SSE 进度流 ===")
    with client.stream("GET", f"/api/tasks/{task_id}/events") as stream:
        check("SSE 200", stream.status_code == 200, stream.status_code)
        check("SSE content-type", "text/event-stream" in stream.headers.get("content-type", ""),
              stream.headers.get("content-type"))
        frames: list[str] = []
        for line in stream.iter_lines():
            if line.startswith("data: "):
                frames.append(line[6:])
                break
    check("SSE 首帧是 JSON", bool(frames) and json.loads(frames[0])["task_id"] == task_id, frames[:1])

    print("\n=== 5. 视频查询 ===")
    r = client.get("/api/videos")
    check("videos 200", r.status_code == 200, r.status_code)
    items = r.json()
    check("列表含该视频", any(v["video_id"] == video_id for v in items), len(items))
    listed = next((v for v in items if v["video_id"] == video_id), {})
    check("列表状态 ready", listed.get("status") == "ready", listed.get("status"))
    check("列表时长 40000", listed.get("duration_ms") == 40000, listed.get("duration_ms"))

    r = client.get(f"/api/videos/{video_id}")
    check("详情 200", r.status_code == 200, r.status_code)
    detail = r.json()
    check("详情含 2 切片", detail["clip_count"] == 2, detail["clip_count"])
    check("详情含关键帧", detail["keyframe_count"] >= 4, detail["keyframe_count"])
    clip = detail["clips"][0]
    check("切片带毫秒区间", clip["end_ms"] > clip["start_ms"] > -1, clip)
    check("切片带 HH:MM:SS.mmm", "–" in clip["time_range"] and "." in clip["time_range"],
          clip["time_range"])
    check("切片绑定关键帧", bool(clip["keyframe_ids"]), clip["keyframe_ids"])
    frame = detail["keyframes"][0]
    check("关键帧有时间码", ":" in frame["timecode"], frame["timecode"])

    r = client.get(f"/api/videos/{video_id}/clips")
    check("clips 端点", r.status_code == 200 and len(r.json()) == 2, r.status_code)
    r = client.get(f"/api/videos/{video_id}/keyframes")
    check("keyframes 端点", r.status_code == 200 and len(r.json()) >= 4, r.status_code)

    kf_id = frame["keyframe_id"]
    r = client.get(f"/api/videos/{video_id}/keyframes/{kf_id}/image")
    check("关键帧图片 200", r.status_code == 200, r.status_code)
    check("图片是 JPEG", r.headers.get("content-type", "").startswith("image/jpeg"),
          r.headers.get("content-type"))
    check("图片有字节", len(r.content) > 1024, len(r.content))
    check("图片 <= 200KB", len(r.content) <= 200 * 1024, len(r.content))

    print("\n=== 6. 问答 ===")
    r = client.post(f"/api/videos/{video_id}/ask", json={"question": "切片长度是怎么定的"})
    check("ask 200", r.status_code == 200, f"{r.status_code} {r.text[:200]}")
    ans = r.json()
    check("回答非空", bool(ans["answer"].strip()), ans["answer"][:60])
    check("带引用", len(ans["refs"]) >= 1, len(ans["refs"]))
    ref = ans["refs"][0]
    check("引用含 clip_id", bool(ref["clip_id"]), ref["clip_id"])
    check("引用含毫秒时间戳", ref["end_ms"] > ref["start_ms"] >= 0, ref)
    check("引用含原文", bool(ref["text"]), ref["text"][:40])

    print("\n=== 6b. 章节 ===")
    r = client.get(f"/api/videos/{video_id}/chapters", params={"min_seconds": 600})
    check("chapters 200", r.status_code == 200, f"{r.status_code} {r.text[:200]}")
    payload = r.json()
    check("短视频不拆章", payload["needs_chapters"] is False, payload.get("count"))
    check("chapters 为空列表", payload["chapters"] == [], payload["chapters"])
    r = client.get(f"/api/videos/{video_id}/chapters",
                   params={"min_seconds": 0, "chapter_seconds": 10, "summarize": "false"})
    forced = r.json()
    check("强制分章有结果", forced["needs_chapters"] is True, forced.get("count"))
    check("每章带时间范围",
          all("time_range" in c and c["end_ms"] > c["start_ms"] for c in forced["chapters"]),
          forced["chapters"][:1])
    check("chapter_seconds 被夹紧", 30 <= forced["chapter_seconds"] <= 3600,
          forced["chapter_seconds"])
    r = client.get(f"/api/videos/{video_id}/chapters/report",
                   params={"min_seconds": 0, "chapter_seconds": 10})
    check("report 200", r.status_code == 200, r.status_code)
    check("report 是 markdown", r.headers["content-type"].startswith("text/markdown"),
          r.headers.get("content-type"))
    check("report 有一级标题", r.text.startswith("# "), r.text[:40])
    check("未知视频 chapters 404",
          client.get("/api/videos/vnothing/chapters").status_code == 404)

    print("\n=== 7. 参数校验与 404 ===")
    check("空问题 422", client.post(f"/api/videos/{video_id}/ask",
                                    json={"question": "  "}).status_code == 422)
    check("超长问题 422", client.post(f"/api/videos/{video_id}/ask",
                                      json={"question": "问" * 501}).status_code == 422)
    check("多余字段 422", client.post(f"/api/videos/analyze",
                                      json={"source": "x", "nope": 1}).status_code == 422)
    check("未知视频 404", client.get("/api/videos/vnonexistent").status_code == 404)
    check("未知任务 404", client.get("/api/tasks/tnope").status_code == 404)
    check("问答未知视频 404", client.post("/api/videos/vnonexistent/ask",
                                          json={"question": "在吗"}).status_code == 404)

    print("\n=== 8. 鉴权 ===")
    check("缺 token 401", noauth.get("/api/videos").status_code == 401,
          noauth.get("/api/videos").status_code)
    check("错 token 403",
          noauth.get("/api/videos", headers={"Authorization": "Bearer wrong"}).status_code == 403)
    check("对 token 200",
          client.get("/api/videos",
                     headers={"Authorization": "Bearer s3cret-token"}).status_code == 200)
    check("X-API-Token 也可",
          client.get("/api/videos",
                     headers={"X-API-Token": "s3cret-token"}).status_code == 200)
    check("health 不需要 token", noauth.get("/api/health").status_code == 200)

    # 直测 require_auth 的回环判定（TestClient 的 host 是桩值，走不了这条路径）
    from api.routes import require_auth

    def fake_request(host: str, headers: dict[str, str] | None = None) -> Any:
        raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
        scope = {
            "type": "http", "method": "GET", "path": "/", "query_string": b"",
            "headers": raw, "client": (host, 52341), "server": ("testserver", 80),
            "app": SimpleNamespace(state=SimpleNamespace(config=cfg, orchestrator=None)),
        }
        from fastapi import Request as _Req

        return _Req(scope)

    os.environ.pop(token_env, None)
    try:
        require_auth(fake_request("127.0.0.1"))
        check("回环无 token 放行", True)
        require_auth(fake_request("::1"))
        require_auth(fake_request("::ffff:127.0.0.1"))
        check("IPv6 回环（含映射形式）放行", True)
        try:
            require_auth(fake_request("192.168.1.23"))
            check("非回环无 token 401", False, "未抛异常")
        except Exception as exc:  # noqa: BLE001
            check("非回环无 token 401", getattr(exc, "status_code", 0) == 401, exc)
        os.environ[token_env] = "unit-token"
        require_auth(fake_request("127.0.0.1", {"Authorization": "Bearer unit-token"}))
        check("token 模式回环也要凭据", True)
        try:
            require_auth(fake_request("127.0.0.1"))
            check("token 模式缺凭据 401", False, "未抛异常")
        except Exception as exc:  # noqa: BLE001
            check("token 模式缺凭据 401", getattr(exc, "status_code", 0) == 401, exc)
    finally:
        os.environ.pop(token_env, None)
    os.environ[token_env] = "s3cret-token"

    print("\n=== 9. MCP（直接调分发器）===")
    from api.mcp_server import INTERNAL_ERROR, McpServer

    server = McpServer(parts["orch"], cfg)
    init = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    check("initialize 有 protocolVersion", "protocolVersion" in init["result"], init)
    check("initialize 报 serverInfo",
          init["result"]["serverInfo"]["name"] == cfg.mcp.server_name, init["result"]["serverInfo"])
    check("通知无响应", server.handle(
        {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None)

    tools = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in tools["result"]["tools"]}
    check("工具集完整", {"list_videos", "get_video", "ask_video", "search_video",
                        "summarize_video", "analyze_video", "get_task", "stats"} <= names, names)

    called = server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                            "params": {"name": "list_videos", "arguments": {"limit": 5}}})
    check("tools/call list_videos", called["result"]["isError"] is False, called)
    body = json.loads(called["result"]["content"][0]["text"])
    check("list_videos 有 items", isinstance(body["items"], list) and body["total"] >= 1, body["total"])

    called = server.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                            "params": {"name": "get_video", "arguments": {"video_id": video_id}}})
    body = json.loads(called["result"]["content"][0]["text"])
    check("get_video 带切片", body["clip_count"] == 2, body["clip_count"])
    check("get_video 时间格式", ":" in body["clips"][0]["time_range"], body["clips"][0]["time_range"])

    called = server.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                            "params": {"name": "search_video",
                                       "arguments": {"video_id": video_id, "query": "切片长度", "top_k": 3}}})
    body = json.loads(called["result"]["content"][0]["text"])
    check("search_video 有结果", body["count"] >= 1, body["count"])
    check("search 结果带区间", "start_ms" in body["results"][0], body["results"][0].keys())

    called = server.handle({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                            "params": {"name": "ask_video",
                                       "arguments": {"video_id": video_id, "question": "讲了什么"}}})
    body = json.loads(called["result"]["content"][0]["text"])
    check("ask_video 有回答", bool(body["answer"].strip()), body["answer"][:40])
    check("ask_video 带引用", len(body["refs"]) >= 1, len(body["refs"]))

    called = server.handle({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                            "params": {"name": "stats", "arguments": {}}})
    body = json.loads(called["result"]["content"][0]["text"])
    check("stats 有计数", body["counts"]["videos"] >= 1 and body["counts"]["clips"] >= 2, body["counts"])
    check("stats 有容量上限", "max_clips" in body["limits"], body["limits"])

    err = server.handle({"jsonrpc": "2.0", "id": 8, "method": "nope/method"})
    check("未知方法 -32601", err["error"]["code"] == -32_601, err["error"])
    err = server.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                         "params": {"name": "nope", "arguments": {}}})
    check("未知工具 -32602", err["error"]["code"] == -32_602, err["error"])
    err = server.handle({"jsonrpc": "2.0", "id": 10, "method": "tools/call",
                         "params": {"name": "get_video", "arguments": {"video_id": "vnope"}}})
    check("工具内错误可上报", err["error"]["code"] == -32_602, err["error"])
    err = server.handle({"jsonrpc": "2.0", "id": 11})
    check("缺 method -32600", err["error"]["code"] == -32_600, err["error"])
    check("错误码常量一致", INTERNAL_ERROR == -32_603)

    print("\n=== 10. MCP over HTTP ===")
    r = client.post("/api/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    check("POST /api/mcp 200", r.status_code == 200, r.text[:400])
    check("HTTP MCP 返回工具", len(r.json()["result"]["tools"]) >= 8, r.status_code)
    r = client.get("/api/mcp/tools")
    check("GET /api/mcp/tools", r.status_code == 200 and "tools" in r.json(), r.status_code)

    print("\n=== 11. 统计与删除 ===")
    r = client.get("/api/stats")
    check("stats 200", r.status_code == 200, r.status_code)
    stats = r.json()
    check("stats 视频数 >= 1", stats["videos"] >= 1, stats["videos"])
    check("stats 有 usage 占比", isinstance(stats["usage"], dict), stats["usage"])

    r = client.delete(f"/api/videos/{video_id}")
    check("delete 200", r.status_code == 200, f"{r.status_code} {r.text[:200]}")
    check("delete 报告向量清理", isinstance(r.json().get("vectors"), dict), r.json())
    check("删除后 404", client.get(f"/api/videos/{video_id}").status_code == 404)

    print("\n=== 12. 降级启动 ===")
    app2 = __import__("api.main", fromlist=["create_app"]).create_app(cfg, load_models=False, serve_ui=False)
    client2 = TestClient(app2)
    client2.headers.update({"X-API-Token": os.environ.get(token_env, "s3cret-token")})
    health2 = client2.get("/api/health")
    check("无编排器仍能启动", health2.status_code == 200, health2.status_code)
    check("报告降级", health2.json()["status"] == "degraded", health2.json())
    check("业务接口 503", client2.get("/api/videos").status_code == 503,
          client2.get("/api/videos").status_code)

    # 统一任务体系：HTTP 提交的任务，task_id 必须能从 SQLite / MCP get_task 查到
    print("\n=== 13. 任务 ID 打通 ===")
    r = client.post("/api/videos/analyze", json={"source": str(demo), "title": "二次分析"})
    check("再次提交 202", r.status_code == 202, r.status_code)
    tid2 = r.json()["task_id"]
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        final2 = client.get(f"/api/tasks/{tid2}").json()
        if final2["status"] in {"done", "failed"}:
            break
        time.sleep(0.5)
    check("二次分析完成", final2.get("status") == "done", final2.get("status"))
    from api.mcp_server import McpServer as _Mcp

    got = _Mcp(parts["orch"], cfg).handle(
        {"jsonrpc": "2.0", "id": 20, "method": "tools/call",
         "params": {"name": "get_task", "arguments": {"task_id": tid2}}}
    )
    body = json.loads(got["result"]["content"][0]["text"])
    check("MCP get_task 查得到 HTTP 任务", body.get("task_id") == tid2 and body.get("status") == "done",
          {k: body.get(k) for k in ("task_id", "status")})
    mcp_submit = _Mcp(parts["orch"], cfg).handle(
        {"jsonrpc": "2.0", "id": 21, "method": "tools/call",
         "params": {"name": "analyze_video", "arguments": {"source": str(demo), "wait": False}}}
    )
    sub = json.loads(mcp_submit["result"]["content"][0]["text"])
    check("MCP 异步提交返回 task_id", str(sub.get("task_id", "")).startswith("t"), sub)
    http_check = client.get(f"/api/tasks/{sub['task_id']}")
    check("HTTP 能查 MCP 提交的任务", http_check.status_code == 200, http_check.status_code)

    if saved is not None:
        os.environ[token_env] = saved

    print(f"\nPASS={PASS}  FAIL={FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
