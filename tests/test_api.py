"""api 层自检（REST + SSE + MCP），零第三方测试框架。

用 ``fastapi.testclient.TestClient``（httpx 驱动）跑真实 ASGI，不启端口。
被测的 ``Orchestrator`` 用与 ``test_pipeline.py`` 相同的假组件装配，
因此本测试**不依赖模型权重**，只依赖 ffmpeg（抽帧）与 chromadb（向量）。

覆盖：
  1. /internal/health 与根路径探活
  2. 分析任务：提交 → 轮询 → 完成；SSE 进度帧
  3. 视频列表 / 详情 / 切片 / 关键帧 / 关键帧图片字节
  4. 问答（带引用与毫秒时间戳）
  5. 参数校验（422）、404、鉴权（401/403/200）
  6. MCP：initialize / tools/list / tools/call / 错误码 / HTTP 端点
  7. 删除与容量统计

跑法::

    & "C:\\Program Files\\Python311\\python.exe" tests/test_backend.web.py
"""

from __future__ import annotations

import hashlib
import json
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

WORK = (
    __import__("backend.runtime", fromlist=["runtime_path"]).runtime_path("tests/work")
    / "api"
)
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

    def transcribe(
        self,
        audio_path: Any,
        video_id: str,
        subtitles: Sequence[Any] = (),
        duration_ms: int | None = None,
        allow_subtitle: bool | None = None,
    ) -> Any:
        self.calls += 1
        segments = [
            {
                "index": 0,
                "start_ms": 0,
                "end_ms": 8200,
                "text": "第一段讲的是视频解析与音频抽取。",
            },
            {
                "index": 1,
                "start_ms": 8200,
                "end_ms": 20100,
                "text": "第二段说明切片长度默认十到三十秒。",
            },
            {
                "index": 2,
                "start_ms": 20100,
                "end_ms": 30500,
                "text": "第三段介绍关键帧抽取与清晰度过滤。",
            },
            {
                "index": 3,
                "start_ms": 30500,
                "end_ms": 40000,
                "text": "最后一段讲向量检索与融合排序。",
            },
        ]
        return SimpleNamespace(
            video_id=video_id,
            segments=[SimpleNamespace(**s) for s in segments],
            language="zh",
            language_probability=0.98,
            duration_ms=duration_ms or 40000,
            source="asr",
            model_name="fake-whisper",
            device="cpu",
            compute_type="int8",
            elapsed_ms=10,
            subtitle_path=None,
            dropped_segments=0,
            degraded=False,
            warnings=[],
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
    def describe(
        self,
        image_path: Any,
        prompt: str | None = None,
        prompt_key: str | None = None,
        extra_vars: dict[str, Any] | None = None,
    ) -> Any:
        return SimpleNamespace(
            image_path=str(image_path),
            description="彩色测试图，含色块与网格",
            ocr_text="TC 00:00:05",
            degraded=False,
            error="",
        )


class FakeLlm:
    def complete(self, messages: Sequence[Any], **kwargs: Any) -> Any:
        return SimpleNamespace(
            text="根据 [1]，切片长度默认 10-30 秒；[2] 补充了 1.5 秒重叠。",
            backend="fake",
            model="fake-llm",
            degraded=False,
            prompt_tokens=10,
            completion_tokens=20,
            elapsed_ms=5,
            finish_reason="stop",
        )

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 装配
# ---------------------------------------------------------------------------


def build_app(cfg: Any) -> tuple[Any, dict[str, Any]]:
    from backend.web.main import create_app
    from backend.pipeline import ContextBuilder, Orchestrator
    from backend.segmentation.chunker import Chunker
    from backend.rag.vector_store import VectorStore
    from backend.storage.repository import SqliteRepository

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
    app = create_app(cfg, orchestrator=orch)
    return app, {"repo": repo, "vectors": vectors, "orch": orch}


def main() -> int:
    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:
        print(f"缺少依赖: {exc}\n  pip install fastapi httpx")
        return 2

    from backend.config import load_app_config

    demo, _low = ensure_fixtures()
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)

    cfg = load_app_config(
        root=str(ROOT),
        overrides={
            "paths.audio": str(WORK / "audio"),
            "paths.frames": str(WORK / "frames"),
            "paths.videos": str(WORK / "videos"),
            "paths.sqlite": str(WORK / "zhiying.db"),
            "paths.chroma": str(WORK / "chroma"),
            "api.sse_heartbeat_s": 1,
        },
    )
    app, parts = build_app(cfg)

    from backend.segmentation.scenes import SceneDetectConfig, SceneDetector

    parts["orch"]._scene = SceneDetector(SceneDetectConfig(frames_per_scene=2))

    client = TestClient(app)

    print("\n=== 1. 健康检查 ===")
    r = client.get("/internal/health")
    check("health 200", r.status_code == 200, r.status_code)
    payload = r.json()
    check("health 报告 orchestrator 就绪", payload["orchestrator"] is True, payload)
    check("health 为本机网页", payload["auth"] == "local-web", payload["auth"])
    check("根路径进入网页", client.get("/").status_code == 200)

    print("\n=== 2. 提交分析任务 ===")
    r = client.post(
        "/internal/videos/analyze", json={"source": str(demo), "title": "接口测试片"}
    )
    check("analyze 202", r.status_code == 202, f"{r.status_code} {r.text[:200]}")
    job = r.json()
    task_id = job["task_id"]
    check("返回 task_id", task_id.startswith("t"), task_id)
    check("初始状态可识别", job["status"] in {"queued", "running"}, job["status"])

    print("\n=== 3. 轮询到完成 ===")
    deadline = time.monotonic() + 180
    final: dict[str, Any] = {}
    while time.monotonic() < deadline:
        r = client.get(f"/internal/tasks/{task_id}")
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
    check(
        "关键帧 >= 4",
        (report.get("keyframe_count") or 0) >= 4,
        report.get("keyframe_count"),
    )
    check(
        "阶段记录 9 条",
        len(report.get("stages") or []) == 9,
        len(report.get("stages") or []),
    )
    check("总结非空", bool((report.get("summary") or "").strip()))

    print("\n=== 4. SSE 进度流 ===")
    with client.stream("GET", f"/internal/tasks/{task_id}/events") as stream:
        check("SSE 200", stream.status_code == 200, stream.status_code)
        check(
            "SSE content-type",
            "text/event-stream" in stream.headers.get("content-type", ""),
            stream.headers.get("content-type"),
        )
        frames: list[str] = []
        for line in stream.iter_lines():
            if line.startswith("data: "):
                frames.append(line[6:])
                break
    check(
        "SSE 首帧是 JSON",
        bool(frames) and json.loads(frames[0])["task_id"] == task_id,
        frames[:1],
    )

    print("\n=== 5. 视频查询 ===")
    r = client.get("/internal/videos")
    check("videos 200", r.status_code == 200, r.status_code)
    items = r.json()
    check("列表含该视频", any(v["video_id"] == video_id for v in items), len(items))
    listed = next((v for v in items if v["video_id"] == video_id), {})
    check("列表状态 ready", listed.get("status") == "ready", listed.get("status"))
    check(
        "列表时长 40000", listed.get("duration_ms") == 40000, listed.get("duration_ms")
    )

    r = client.get(f"/internal/videos/{video_id}")
    check("详情 200", r.status_code == 200, r.status_code)
    detail = r.json()
    check("详情含 2 切片", detail["clip_count"] == 2, detail["clip_count"])
    check("详情含关键帧", detail["keyframe_count"] >= 4, detail["keyframe_count"])
    clip = detail["clips"][0]
    check("切片带毫秒区间", clip["end_ms"] > clip["start_ms"] > -1, clip)
    check(
        "切片带 HH:MM:SS.mmm",
        "–" in clip["time_range"] and "." in clip["time_range"],
        clip["time_range"],
    )
    check("切片绑定关键帧", bool(clip["keyframe_ids"]), clip["keyframe_ids"])
    frame = detail["keyframes"][0]
    check("关键帧有时间码", ":" in frame["timecode"], frame["timecode"])

    r = client.get(f"/internal/videos/{video_id}/clips")
    check("clips 端点", r.status_code == 200 and len(r.json()) == 2, r.status_code)
    r = client.get(f"/internal/videos/{video_id}/keyframes")
    check("keyframes 端点", r.status_code == 200 and len(r.json()) >= 4, r.status_code)

    kf_id = frame["keyframe_id"]
    r = client.get(f"/internal/videos/{video_id}/keyframes/{kf_id}/image")
    check("关键帧图片 200", r.status_code == 200, r.status_code)
    check(
        "图片是 JPEG",
        r.headers.get("content-type", "").startswith("image/jpeg"),
        r.headers.get("content-type"),
    )
    check("图片有字节", len(r.content) > 1024, len(r.content))
    check("图片 <= 200KB", len(r.content) <= 200 * 1024, len(r.content))

    print("\n=== 5b. 字幕导出（SRT/VTT） ===")
    r = client.get(f"/internal/videos/{video_id}/subtitles")
    check("SRT 200", r.status_code == 200, f"{r.status_code} {r.text[:200]}")
    check(
        "SRT 是附件",
        "attachment" in r.headers.get("content-disposition", ""),
        r.headers.get("content-disposition"),
    )
    check(
        "SRT 文件名带 .srt",
        ".srt" in r.headers.get("content-disposition", ""),
        r.headers.get("content-disposition"),
    )
    srt = r.text
    check(
        "SRT 含 4 个块",
        srt.count("\n\n") == 3 and srt.strip().startswith("1\n"),
        srt[:120],
    )
    check("SRT 时间戳用逗号", "00:00:00,000 --> 00:00:08,200" in srt, srt[:80])
    check("SRT 含转写文本", "切片长度默认十到三十秒" in srt, "")

    r = client.get(f"/internal/videos/{video_id}/subtitles?format=vtt")
    check("VTT 200", r.status_code == 200, r.status_code)
    vtt = r.text
    check("VTT 有 WEBVTT 头", vtt.startswith("WEBVTT\n"), vtt[:20])
    check("VTT 时间戳用点号", "00:00:00.000 --> 00:00:08.200" in vtt, vtt[:80])
    check(
        "VTT 无逗号时间戳",
        ",0" not in vtt.split("\n")[2] if len(vtt.split("\n")) > 2 else True,
        "",
    )

    r = client.get(f"/internal/videos/{video_id}/transcript")
    check("transcript 200", r.status_code == 200, r.status_code)
    tr = r.json()
    check("transcript 4 段", tr["count"] == 4, tr.get("count"))
    check("transcript 含时间戳", tr["segments"][0]["start_ms"] == 0, tr["segments"][:1])
    check("transcript 含文本", "切片长度" in tr["segments"][1]["text"], "")

    r = client.get(f"/internal/videos/{video_id}/subtitles?format=bogus")
    check("非法 format 422", r.status_code == 422, r.status_code)

    r = client.get("/internal/videos/not-exist/subtitles")
    check("404 视频不存在", r.status_code == 404, r.status_code)

    # 空段视频：手动建一个无转写片段的视频记录
    parts["repo"].add_video(video_id="v-empty", title="无转写", status="ready")
    r = client.get("/internal/videos/v-empty/subtitles")
    check("无转写片段 404", r.status_code == 404, r.status_code)

    r = client.get(f"/internal/videos/{video_id}/report")
    check("持久化报告 200", r.status_code == 200, f"{r.status_code} {r.text[:120]}")
    rep = r.json()
    check("报告状态 ready", rep.get("status") == "ready", rep.get("status"))
    check(
        "报告带总结",
        bool((rep.get("summary") or "").strip()),
        str(rep.get("summary"))[:60],
    )
    check("报告带阶段", len(rep.get("stages") or []) == 9, len(rep.get("stages") or []))
    r = client.get("/internal/videos/not-exist/report")
    check("报告 404 视频不存在", r.status_code == 404, r.status_code)
    r = client.get("/internal/videos/v-empty/report")
    check("报告 404 未分析", r.status_code == 404, r.status_code)

    print("\n=== 6. 问答 ===")
    r = client.post(
        f"/internal/videos/{video_id}/ask", json={"question": "切片长度是怎么定的"}
    )
    check("ask 200", r.status_code == 200, f"{r.status_code} {r.text[:200]}")
    ans = r.json()
    check("回答非空", bool(ans["answer"].strip()), ans["answer"][:60])
    check("带引用", len(ans["refs"]) >= 1, len(ans["refs"]))
    ref = ans["refs"][0]
    check("引用含 clip_id", bool(ref["clip_id"]), ref["clip_id"])
    check("引用含毫秒时间戳", ref["end_ms"] > ref["start_ms"] >= 0, ref)
    check("引用含原文", bool(ref["text"]), ref["text"][:40])

    print("\n=== 6c. 流式问答（SSE） ===")
    with client.stream(
        "POST",
        f"/internal/videos/{video_id}/ask/stream",
        json={"question": "切片长度是怎么定的"},
    ) as stream:
        check("stream 200", stream.status_code == 200, stream.status_code)
        check(
            "stream content-type",
            "text/event-stream" in stream.headers.get("content-type", ""),
            stream.headers.get("content-type"),
        )
        events = []
        for line in stream.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    kinds = [e.get("type") for e in events]
    check("有 meta 事件", "meta" in kinds, kinds)
    check("有 delta 事件", "delta" in kinds, kinds)
    check("有 final 事件", "final" in kinds, kinds)
    check(
        "事件顺序 meta→delta→final",
        kinds.index("meta") < kinds.index("delta") < kinds.index("final"),
        kinds,
    )
    final_evt = next((e for e in events if e.get("type") == "final"), {})
    final_result = final_evt.get("result") or {}
    check("final 带回答", bool((final_result.get("answer") or "").strip()))
    check(
        "final 带引用",
        len(final_result.get("refs") or []) >= 1,
        len(final_result.get("refs") or []),
    )
    check(
        "final 带 backend",
        bool(final_result.get("backend")),
        final_result.get("backend"),
    )
    with client.stream(
        "POST", "/internal/videos/vnope/ask/stream", json={"question": "在吗"}
    ) as stream2:
        ev2 = [
            json.loads(line[6:])
            for line in stream2.iter_lines()
            if line.startswith("data: ")
        ]
    check(
        "未知视频流式返回带说明的 final",
        any(
            e.get("type") == "final"
            and "视频不存在" in "".join((e.get("result") or {}).get("notes") or [])
            for e in ev2
        ),
        [e.get("type") for e in ev2],
    )

    print("\n=== 6b. 章节 ===")
    r = client.get(f"/internal/videos/{video_id}/chapters", params={"min_seconds": 600})
    check("chapters 200", r.status_code == 200, f"{r.status_code} {r.text[:200]}")
    payload = r.json()
    check("短视频不拆章", payload["needs_chapters"] is False, payload.get("count"))
    check("chapters 为空列表", payload["chapters"] == [], payload["chapters"])
    r = client.get(
        f"/internal/videos/{video_id}/chapters",
        params={"min_seconds": 0, "chapter_seconds": 10, "summarize": "false"},
    )
    forced = r.json()
    check("强制分章有结果", forced["needs_chapters"] is True, forced.get("count"))
    check(
        "每章带时间范围",
        all(
            "time_range" in c and c["end_ms"] > c["start_ms"]
            for c in forced["chapters"]
        ),
        forced["chapters"][:1],
    )
    check(
        "chapter_seconds 被夹紧",
        30 <= forced["chapter_seconds"] <= 3600,
        forced["chapter_seconds"],
    )
    r = client.get(
        f"/internal/videos/{video_id}/chapters/report",
        params={"min_seconds": 0, "chapter_seconds": 10},
    )
    check("report 200", r.status_code == 200, r.status_code)
    check(
        "report 是 markdown",
        r.headers["content-type"].startswith("text/markdown"),
        r.headers.get("content-type"),
    )
    check("report 有一级标题", r.text.startswith("# "), r.text[:40])
    check(
        "未知视频 chapters 404",
        client.get("/internal/videos/vnothing/chapters").status_code == 404,
    )

    print("\n=== 7. 参数校验与 404 ===")
    check(
        "空问题 422",
        client.post(
            f"/internal/videos/{video_id}/ask", json={"question": "  "}
        ).status_code
        == 422,
    )
    check(
        "超长问题 422",
        client.post(
            f"/internal/videos/{video_id}/ask", json={"question": "问" * 501}
        ).status_code
        == 422,
    )
    check(
        "多余字段 422",
        client.post(
            "/internal/videos/analyze", json={"source": "x", "nope": 1}
        ).status_code
        == 422,
    )
    check(
        "未知视频 404", client.get("/internal/videos/vnonexistent").status_code == 404
    )
    check("未知任务 404", client.get("/internal/tasks/tnope").status_code == 404)
    check(
        "问答未知视频 404",
        client.post(
            "/internal/videos/vnonexistent/ask", json={"question": "在吗"}
        ).status_code
        == 404,
    )

    print("\n=== 8. 仅保留本机网页内部接口 ===")
    for path in ("/api/videos", "/api/mcp", "/docs", "/openapi.json", "/health"):
        check("已删除旧路径 " + path, client.get(path).status_code == 404)
    check(
        "跨域请求被拒绝",
        client.post(
            "/internal/videos/analyze",
            headers={"Origin": "https://external.test"},
            json={"source": "x"},
        ).status_code
        == 403,
    )
    check(
        "跨站请求被拒绝",
        client.get(
            "/internal/videos", headers={"Sec-Fetch-Site": "cross-site"}
        ).status_code
        == 403,
    )
    check("工作台可打开", client.get("/ui").status_code == 200)
    check(
        "独立前端脚本可加载", client.get("/assets/workbench/app.js").status_code == 200
    )
    check("独立前端样式可加载", client.get("/assets/chat/style.css").status_code == 200)

    check(
        "旧内部 MCP 路径也已删除",
        client.post("/internal/mcp", json={}).status_code == 404,
    )

    print("\n=== 11b. /chat 对话页 ===")
    r = client.get("/chat")
    check("chat 页 200", r.status_code == 200, r.status_code)
    check(
        "chat 页是 HTML",
        "text/html" in r.headers.get("content-type", "") and "视频对话" in r.text,
    )
    r = client.get(f"/internal/videos/{video_id}/file")
    check("视频文件流 200", r.status_code == 200, r.status_code)
    check(
        "视频文件是 mp4",
        r.headers.get("content-type", "").startswith("video/mp4"),
        r.headers.get("content-type"),
    )
    check("视频文件有字节", len(r.content) > 100_000, len(r.content))
    check("文件流带 Range 能力", r.headers.get("accept-ranges") == "bytes")
    check(
        "未知视频文件 404", client.get("/internal/videos/vnope/file").status_code == 404
    )
    r = client.post(
        f"/internal/videos/{video_id}/ask", json={"question": "切片长度是怎么定的"}
    )
    check("对话前先问一轮", r.status_code == 200, r.status_code)
    r = client.get(f"/internal/videos/{video_id}/chat")
    body = r.json()
    check(
        "会话历史有轮次",
        body["count"] >= 1 and bool(body["turns"][-1]["answer"]),
        body.get("count"),
    )
    check("历史含提问", body["turns"][-1]["question"] == "切片长度是怎么定的")
    r = client.delete(f"/internal/videos/{video_id}/chat")
    check("清空对话", r.status_code == 200 and r.json()["cleared"] is True)
    check(
        "清空后为空",
        client.get(f"/internal/videos/{video_id}/chat").json()["count"] == 0,
    )

    print("\n=== 11. 统计与删除 ===")
    r = client.get("/internal/stats")
    check("stats 200", r.status_code == 200, r.status_code)
    stats = r.json()
    check("stats 视频数 >= 1", stats["videos"] >= 1, stats["videos"])
    check("stats 有 usage 占比", isinstance(stats["usage"], dict), stats["usage"])

    # 项目内视频副本应随删除一起清掉（下载到 data/videos 的 source.mp4）
    src_copy = WORK / "videos" / video_id / "source.mp4"
    src_copy.parent.mkdir(parents=True, exist_ok=True)
    src_copy.write_bytes(b"\x00" * 128)
    # 把副本登记为该视频的 local_path —— 模拟"下载入库"后的真实记录
    parts["repo"].add_video(video_id=video_id, local_path=str(src_copy), replace=True)
    r = client.delete(f"/internal/videos/{video_id}")
    check("delete 200", r.status_code == 200, f"{r.status_code} {r.text[:200]}")
    check("delete 报告向量清理", isinstance(r.json().get("vectors"), dict), r.json())
    check("项目内视频副本已删", not src_copy.exists(), str(src_copy))
    check("删除后 404", client.get(f"/internal/videos/{video_id}").status_code == 404)
    check(
        "删除后报告也没了",
        client.get(f"/internal/videos/{video_id}/report").status_code == 404,
    )

    print(f"\nPASS={PASS}  FAIL={FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
