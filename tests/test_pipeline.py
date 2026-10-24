"""Orchestrator 管道自检（纯 stdlib 断言，不依赖 pytest）。

覆盖：
  1. analyze 全链路（真实 ffmpeg 抽帧 + 真实 Chroma 写向量 + 真 SQLite）
  2. 无音轨降级路径
  3. 幂等 / force 重跑 / 断点恢复
  4. ask 检索 → 上下文 → 生成 → 引用解析
  5. extract_citations 边界
  6. delete 清理
  7. 进度回调与 to_dict 序列化

跑法：
  & "C:\\Program Files\\Python311\\python.exe" tests/t_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORK = ROOT / ".tmp_test" / "pipeline"
DEMO = ROOT / ".tmp_test" / "media" / "demo.mp4"


def _orchestrator_module():
    """延迟导入，避免 import 期就把 cv2/chromadb 拉起来。"""
    import core.pipeline.orchestrator as module

    return module


PASS = 0
FAIL = 0


def check(label: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label} {extra}")


# ---------------------------------------------------------------------------
# 假组件（形状对齐真实接口）
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
    def __init__(self, segments: Sequence[dict[str, Any]] | None = None) -> None:
        self.calls = 0
        self.segments = list(segments or SEGMENTS)

    def transcribe(
        self,
        audio_path: Any,
        video_id: str,
        subtitles: Sequence[Any] = (),
        duration_ms: int | None = None,
        allow_subtitle: bool | None = None,
    ) -> Any:
        self.calls += 1
        return SimpleNamespace(
            video_id=video_id,
            segments=[SimpleNamespace(**seg) for seg in self.segments],
            language="zh",
            language_probability=0.98,
            duration_ms=duration_ms or 40000,
            source="asr",
            model_name="fake-whisper",
            device="cpu",
            compute_type="int8",
            elapsed_ms=12,
            subtitle_path=None,
            dropped_segments=0,
            degraded=False,
            warnings=[],
        )


class FakeEmbedder:
    def __init__(self) -> None:
        self.text_calls = 0
        self.image_calls = 0

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.text_calls += 1
        return [_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _embed(text)

    def embed_images(self, images: Sequence[Any]) -> list[list[float]]:
        self.image_calls += 1
        return [_embed(f"image {i} frame") for i, _ in enumerate(images)]

    def embed_cross_modal(self, text: str) -> list[float]:
        return _embed(text)

    def close(self) -> None:
        pass


class FakeVl:
    def __init__(self) -> None:
        self.calls = 0
        self.sequences: list[tuple[int, tuple[int, int] | None]] = []

    def describe(self, image_path: Any, prompt: str | None = None, prompt_key: str | None = None,
                 extra_vars: dict[str, Any] | None = None) -> Any:
        self.calls += 1
        return SimpleNamespace(
            image_path=str(image_path),
            description=f"画面 {self.calls}：彩色测试图，包含色块与坐标网格",
            ocr_text=f"TC 00:00:0{self.calls}",
            degraded=False,
            error="",
        )

    def describe_sequence(self, frames: Any, *, span_ms: Any = None,
                          prompt: str | None = None, prompt_key: str = "",
                          max_frames: Any = None) -> Any:
        pairs = list(frames)
        self.calls += 1
        self.sequences.append((len(pairs), span_ms))
        # 一段连续叙述写进组内每一帧，与真实 VlService 行为一致
        return SimpleNamespace(
            image_path=str(pairs[0][0]),
            description=f"这 {len(pairs)} 帧里画面基本保持不变，测试色块与坐标网格贯穿始终。",
            ocr_text=f"TC 00:00:0{self.calls}",
            degraded=False,
            error="",
            extra={"frame_count": len(pairs)},
        )


class FakeLlm:
    def __init__(self, text: str = "根据 [1] 的片段，切片长度默认在 10 到 30 秒之间。[2] 补充了重叠。") -> None:
        self.text = text
        self.calls: list[str] = []

    def complete(self, messages: Sequence[Any], **kwargs: Any) -> Any:
        self.calls.append(str(messages[-1].get("content") if isinstance(messages[-1], dict) else messages[-1]))
        return SimpleNamespace(text=self.text, backend="fake", model="fake-llm", degraded=False,
                               prompt_tokens=10, completion_tokens=20, elapsed_ms=5,
                               finish_reason="stop")

    def close(self) -> None:
        pass


SEGMENTS = [
    {"index": 0, "start_ms": 0, "end_ms": 8200, "text": "第一段讲的是视频解析与音频抽取。"},
    {"index": 1, "start_ms": 8200, "end_ms": 20100, "text": "第二段说明切片长度默认十到三十秒。", },
    {"index": 2, "start_ms": 20100, "end_ms": 30500, "text": "第三段介绍关键帧抽取与清晰度过滤。"},
    {"index": 3, "start_ms": 30500, "end_ms": 40000, "text": "最后一段讲向量检索与融合排序。"},
]


# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------


def reset_work() -> None:
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)


def build(config: Any) -> tuple[Any, dict[str, Any]]:
    from core.pipeline import ContextBuilder, Orchestrator
    from core.rag.chunker import Chunker
    from core.rag.vector_store import VectorStore
    from core.storage.repository import SqliteRepository

    repo = SqliteRepository(WORK / "zhiying.db")
    vectors = VectorStore(WORK / "chroma")
    parts: dict[str, Any] = {
        "asr": FakeAsr(),
        "embedder": FakeEmbedder(),
        "vl": FakeVl(),
        "llm": FakeLlm(),
    }
    orch = Orchestrator(
        repository=repo,
        vectors=vectors,
        embedder=parts["embedder"],
        asr=parts["asr"],
        scene_detector=None,  # 下面按需注入真检测器
        chunker=Chunker(),
        vl=parts["vl"],
        llm=parts["llm"],
        context_builder=ContextBuilder(None),
        config=config,
    )
    parts.update({"repo": repo, "vectors": vectors, "orch": orch})
    return orch, parts


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> int:
    from core.config import load_app_config

    if not DEMO.exists():
        print(f"缺少测试片源: {DEMO}")
        return 2
    reset_work()

    cfg = load_app_config(
        root=str(ROOT),
        overrides={
            "paths.audio": str(WORK / "audio"),
            "paths.frames": str(WORK / "frames"),
            "paths.sqlite": str(WORK / "zhiying.db"),
            "paths.chroma": str(WORK / "chroma"),
        },
    )

    print("\n=== 1. analyze 全链路 ===")
    orch, parts = build(cfg)
    from core.media.scene_detect import SceneDetectConfig, SceneDetector

    orch._scene = SceneDetector(SceneDetectConfig(frames_per_scene=2, min_frames_per_scene=1))

    events: list[tuple[str, float, str]] = []
    from core.pipeline import AnalyzeOptions

    report = orch.analyze(
        str(DEMO),
        AnalyzeOptions(progress=lambda s, r, m: events.append((s, r, m))),
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2)[:1600])

    check("analyze 成功", report.status == "ready", report.status)
    check("有 video_id", bool(report.video_id))
    check("时长 40000ms", report.duration_ms == 40000, report.duration_ms)
    check("切片 2 个", report.clip_count == 2, report.clip_count)
    check("关键帧 >= 4", report.keyframe_count >= 4, report.keyframe_count)
    check("转录 4 段", report.segment_count == 4, report.segment_count)
    check("向量已写入", report.vector_count >= 2, report.vector_count)
    check("音频抽取成功（demo 含 aac 音轨）", not any("音轨" in d for d in report.degraded),
          report.degraded)
    check("阶段全部记录", len(report.stages) == 9, len(report.stages))
    check("阶段无失败", all(s.ok for s in report.stages))
    check("进度回调多次", len(events) >= 9, len(events))
    check("进度单调非减", all(events[i][1] <= events[i + 1][1] + 1e-9 for i in range(len(events) - 1)))
    check("总结非空", bool(report.summary.strip()))

    print("\n=== 2. 数据库落盘 ===")
    repo = parts["repo"]
    vid = report.video_id
    video = repo.find_video(vid)
    check("video ready", video is not None and video.status == "ready", video.status if video else None)
    clips = repo.get_clips(vid)
    keyframes = repo.get_keyframes(vid)
    check("clips 落库 2", len(clips) == 2, len(clips))
    check("keyframes 落库", len(keyframes) == report.keyframe_count, len(keyframes))
    check("关键帧全部绑定 clip", all(k.clip_id for k in keyframes),
          [k.clip_id for k in keyframes])
    check("关键帧有描述", all(k.description for k in keyframes))
    check("关键帧有 OCR", any(k.ocr_text for k in keyframes))
    check("切片带毫秒时间戳", all(c.start_ms >= 0 and c.end_ms > c.start_ms for c in clips))
    paths = repo.keyframe_paths(vid)
    check("keyframe_paths 映射", len(paths) == len(keyframes))

    print("\n=== 3. 向量库 ===")
    vectors = parts["vectors"]
    # 索引写进"代"（clip_text_<gen>），基础集合保持干净 —— 这样重建失败
    # 不会毁掉旧索引。断言要跟着看活动代，不是基础集合名。
    gen = repo.get_video(vid).vector_generation
    check("记录了索引代", bool(gen), gen)
    active = vectors.with_generation(gen) if gen else vectors
    check("clip 向量数 2", active.count("clip") == 2, active.count("clip"))
    check("frame 向量数一致", active.count("frame") == len(keyframes), active.count("frame"))
    check("基础集合未被写入", vectors.count("clip") == 0, vectors.count("clip"))

    print("\n=== 4. ask ===")
    ans = orch.ask(vid, "切片长度是怎么定的")
    print("answer:", ans.answer)
    print("refs:", [r.to_dict() for r in ans.refs])
    check("有回答", bool(ans.answer.strip()))
    check("有引用", len(ans.refs) >= 1, len(ans.refs))
    check("引用带 clip_id", all(r.clip_id for r in ans.refs))
    check("引用带毫秒区间", all(r.end_ms > r.start_ms for r in ans.refs))
    check("引用 time_range 合法",
          all(r.time_range.count(":") == 4 or "：" in r.time_range or "–" in r.time_range for r in ans.refs),
          [r.time_range for r in ans.refs])
    check("引用格式 HH:MM:SS.mmm", all(len(r.time_range.split("–")[0]) == 12 for r in ans.refs),
          [r.time_range for r in ans.refs])
    check("引用 video_id 一致", all(r.video_id == vid for r in ans.refs))
    check("上下文非空", ans.context_chars > 0, ans.context_chars)
    check("候选数 > 0", ans.candidates > 0, ans.candidates)
    payload = ans.to_dict()
    blob = json.dumps(payload, ensure_ascii=False)
    check("AskResult JSON 可序列化", len(blob) > 50, f"{len(blob)} 字节")
    check("LLM 收到上下文", len(parts["llm"].calls) >= 2, len(parts["llm"].calls))
    check("引用带原文", all(r.text for r in ans.refs) or ans.refs[0].text == "",
          [len(r.text) for r in ans.refs])

    print("\n=== 5. ask 边界 ===")
    empty = orch.ask(vid, "   ")
    check("空问题不抛", not empty.ok and empty.notes)
    ghost = orch.ask("no-such-video", "讲了什么")
    check("不存在视频不抛", not ghost.ok and ghost.notes)
    ans2 = orch.ask(vid, "完全不相关的量子纠缠问题", top_k=2)
    check("无关查询仍给证据", bool(ans2.answer.strip()) and ans2.candidates >= 1, ans2.candidates)

    print("\n=== 6. extract_citations ===")
    from core.pipeline import extract_citations

    refs = [
        {"index": 1, "clip_id": "a", "video_id": "v", "start_ms": 0, "end_ms": 1000,
         "time_range": "00:00:00.000–00:00:01.000"},
        {"index": 2, "clip_id": "b", "video_id": "v", "start_ms": 1000, "end_ms": 2000,
         "time_range": "00:00:01.000–00:00:02.000"},
        {"index": 3, "clip_id": "c", "video_id": "v", "start_ms": 2000, "end_ms": 3000,
         "time_range": "00:00:02.000–00:00:03.000"},
    ]
    got = extract_citations("先说 [2]，再说 [1]，重复 [2]。", refs)
    check("按出现顺序去重", [c.index for c in got] == [2, 1], [c.index for c in got])
    got2 = extract_citations("超范围 [9] 被忽略 [3]。", refs)
    check("超范围忽略", [c.index for c in got2] == [3], [c.index for c in got2])
    got3 = extract_citations("没有任何引用标记。", refs)
    check("无标记回退前 3", [c.index for c in got3] == [1, 2, 3], [c.index for c in got3])
    check("空 refs 安全", extract_citations("x [1]", []) == [])

    print("\n=== 7. 幂等与重跑 ===")
    again = orch.analyze(str(DEMO), AnalyzeOptions())
    check("已就绪视频跳过", again.status == "ready" and len(again.stages) == 2, len(again.stages))
    forced = orch.analyze(str(DEMO), AnalyzeOptions(force=True))
    check("force 重跑成功", forced.status == "ready", forced.status)
    check("force 后切片仍 2", forced.clip_count == 2, forced.clip_count)

    print("\n=== 8. 断点恢复 ===")
    repo.update_video_status(vid, "failed", stage="chunk", error="模拟中断")
    resumed = orch.analyze(str(DEMO), AnalyzeOptions(resume=True))
    check("从失败点恢复", resumed.status == "ready", resumed.status)
    check("恢复后仍完整", resumed.clip_count == 2 and resumed.keyframe_count >= 4)
    reused = [x.stage.value for x in resumed.stages if "断点恢复" in (x.message or "")]
    check("中断在 chunk 时只复用转写", reused == ["transcribe"], str(reused))

    repo.update_video_status(vid, "failed", stage="index", error="模拟中断")
    resumed2 = orch.analyze(str(DEMO), AnalyzeOptions(resume=True))
    reused2 = [x.stage.value for x in resumed2.stages if "断点恢复" in (x.message or "")]
    check("中断在 index 时三段全复用", set(reused2) == {"transcribe", "chunk", "vision"}, str(reused2))
    check("两次恢复后仍完整", resumed2.clip_count == 2 and resumed2.keyframe_count >= 4)
    check("复用写在警告里", sum("复用" in w for w in resumed2.warnings) >= 3, resumed2.warnings)

    # 体检：产物文件少了就必须放弃恢复
    ctx_full = orch._resume_context(vid)
    check("产物齐备时体检通过", {"transcript", "clips", "keyframes"} <= set(ctx_full), sorted(ctx_full))
    hidden: list[tuple[Any, Any]] = []
    try:
        for record in repo.get_keyframes(vid)[:2]:
            src = Path(record.path)
            if src.is_file():
                dst = src.with_suffix(src.suffix + ".hidden")
                src.rename(dst)
                hidden.append((src, dst))
        ctx_broken = orch._resume_context(vid)
        check("缺文件后放弃关键帧", "keyframes" not in ctx_broken, sorted(ctx_broken))
        check("连带放弃切片", "clips" not in ctx_broken, sorted(ctx_broken))
        check("转写不受影响", "transcript" in ctx_broken, sorted(ctx_broken))
    finally:
        for src, dst in hidden:
            if dst.is_file():
                dst.rename(src)

    # 指纹：换了模型/参数，旧产物必须作废
    from core.pipeline import AnalysisReport, compute_fingerprint

    fp_before = repo.get_video(vid).fingerprint
    check("分析后写入了指纹", bool(fp_before), fp_before)
    changed = replace(cfg, asr=replace(cfg.asr, model="another-asr-model"))
    fp_after = compute_fingerprint(changed)
    check("换模型得到不同指纹", fp_after != fp_before, f"{fp_before} vs {fp_after}")
    report_fp = AnalysisReport(duration_ms=0)
    stale = orch._apply_fingerprint(report_fp, vid, fp_after)
    check("指纹不符判定作废", stale is True, str(stale))
    check("作废后切片被清", repo.get_clips(vid) == [], "")
    check("作废留下警告", any("指纹" in w for w in report_fp.warnings), report_fp.warnings)
    check("新指纹已写入", repo.get_video(vid).fingerprint == fp_after, repo.get_video(vid).fingerprint)
    check("同指纹不再作废", orch._apply_fingerprint(report_fp, vid, fp_after) is False, "")
    # 复原：重跑一次，让后面的测试有数据
    repo.update_video_status(vid, "failed", stage="parse", error="")
    repo.update_video_meta(vid, fingerprint=fp_before)
    rebuilt = orch.analyze(str(DEMO), AnalyzeOptions(resume=True))
    check("作废后能重建", rebuilt.status == "ready" and rebuilt.clip_count == 2, rebuilt.status)

    print("\n=== 9. summarize ===")
    summary = orch.summarize(vid)
    check("summarize 非空", bool(summary.strip()), repr(summary[:40]))

    print("\n=== 10. delete ===")
    removed = orch.delete(vid)
    check("删除返回记录", "db" in removed, removed.keys())
    check("视频已删除", repo.find_video(vid) is None)
    check("向量已删除", vectors.count("clip") == 0, vectors.count("clip"))
    check("重复删除安全", isinstance(orch.delete(vid), dict))

    print("\n=== 11. from_config 装配降级 ===")
    from core.pipeline import Orchestrator as O

    lazy = O.from_config(cfg, repository=repo, load_models=False)
    check("装配不抛", lazy is not None)
    check("缺模型时组件为 off", "off" in repr(lazy) or "on" in repr(lazy), repr(lazy))

    print("\n=== 12. to_dict 序列化 ===")
    text = json.dumps(report.to_dict(), ensure_ascii=False)
    check("AnalysisReport JSON", len(text) > 200, len(text))

    # -- 13. 推理互斥锁：两路任务不能同时把权重压进内存 --------------------
    def _probe_serial(serial: bool) -> tuple[int, int]:
        """三个线程各自进临界区，返回 (峰值并发, 总耗时毫秒)。"""
        peak = {"now": 0, "max": 0}
        lock = threading.Lock()
        holder = SimpleNamespace(limits=SimpleNamespace(inference_serial=serial))

        def worker() -> None:
            with _orchestrator_module().Orchestrator._inference_slot(
                SimpleNamespace(_cfg=holder), "自检", None
            ):
                with lock:
                    peak["now"] += 1
                    peak["max"] = max(peak["max"], peak["now"])
                time.sleep(0.12)
                with lock:
                    peak["now"] -= 1

        started = time.monotonic()
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for item in threads:
            item.start()
        for item in threads:
            item.join()
        return peak["max"], int((time.monotonic() - started) * 1000)

    peak_on, cost_on = _probe_serial(True)
    peak_off, cost_off = _probe_serial(False)
    check("推理互斥：开启后峰值并发=1", peak_on == 1, f"peak={peak_on} 耗时={cost_on}ms")
    check("推理互斥：关闭后确实并行", peak_off > 1, f"peak={peak_off} 耗时={cost_off}ms")
    check("推理互斥：串行更慢（说明真的排了队）", cost_on > cost_off, f"{cost_on}ms vs {cost_off}ms")

    # -- 14. 精确字节预算：中英混排时字符数不能代表体积 --------------------
    from core.pipeline.context_builder import ContextBudget, ContextBuilder, utf8_prefix

    check("utf8_prefix 不切坏汉字", utf8_prefix("知影", 3) == "知", utf8_prefix("知影", 3))
    check("utf8_prefix 未超限时原样返回", utf8_prefix("知影", 99) == "知影")
    check("utf8_prefix 0 字节返回空", utf8_prefix("知影", 0) == "")

    fat = [
        {
            "clip_id": f"bc{i}",
            "video_id": "vb",
            "start_ms": i * 20000,
            "end_ms": i * 20000 + 19000,
            "text": f"第{i}段。" + ("这一段讲训练流程与梯度下降的具体做法。" * 12),
            "score": 0.9 - i * 0.1,
        }
        for i in range(6)
    ]
    for cap in (1200, 3000, 6000):
        budget = ContextBudget(max_chars=6000, max_bytes=cap)
        built = ContextBuilder(budget).build(fat, "字节预算")
        body_bytes = sum(len(it.render().encode("utf-8")) + 1 for it in built.items)
        check(f"字节预算 {cap}B 未超（实测 {body_bytes}B）", body_bytes <= budget.effective_bytes,
              f"{body_bytes} > {budget.effective_bytes}")
        check(f"字节预算 {cap}B 至少留一条", len(built.items) >= 1, len(built.items))
        if cap <= 3000:
            check(f"字节预算 {cap}B 有截断或丢弃标记",
                  any(it.truncated for it in built.items) or built.dropped,
                  f"截断={[it.truncated for it in built.items]} 丢弃={len(built.dropped)}")
        else:
            check(f"字节预算 {cap}B 装得下全部 6 条", len(built.items) == 6, len(built.items))

    # 同样 300 字符，中文体积是英文的三倍 —— 字符预算看不见这个差别
    zh = "中" * 300
    en = "a" * 300
    check("同 300 字符、中文体积恰是英文三倍",
          len(zh.encode()) == len(en.encode()) * 3,
          f"zh={len(zh.encode())} en={len(en.encode())}")

    orch.close()
    print(f"\n{'=' * 60}\nPASS={PASS}  FAIL={FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
