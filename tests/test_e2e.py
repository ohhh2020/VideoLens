"""媒体层 ↔ 检索层端到端联动：抽帧 → 切片 → 关键帧绑定 → 溯源链。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.media.scene_detect import SceneDetector, group_by_clip  # noqa: E402
from core.rag.chunker import Chunker, bind_keyframes  # noqa: E402
from fixtures import ensure as ensure_fixtures  # noqa: E402

ok: list[str] = []
fail: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        ok.append(name)
        print(f"PASS {name}" + (f" | {detail}" if detail else ""))
    else:
        fail.append(name)
        print(f"FAIL {name}" + (f" | {detail}" if detail else ""))


video, _low = ensure_fixtures()
base = video.parent

# 模拟 faster-whisper 在 40s 视频上的输出（真实场景由 ASR 产生）
transcript = [
    {"id": 0, "start": 0.4, "end": 4.8, "text": "大家好，今天演示知影的视频分析流程。"},
    {"id": 1, "start": 5.0, "end": 9.6, "text": "第一步是场景检测，它会找出镜头切换点。"},
    {"id": 2, "start": 10.6, "end": 14.9, "text": "第二步在每个场景里挑最清晰的一帧。"},
    {"id": 3, "start": 15.0, "end": 19.8, "text": "第三步把语音合并成十到三十秒的切片。"},
    {"id": 4, "start": 20.5, "end": 24.9, "text": "第四步把切片和关键帧绑定在一起。"},
    {"id": 5, "start": 25.0, "end": 29.7, "text": "这样每条结论都能回溯到原始画面。"},
    {"id": 6, "start": 30.6, "end": 34.8, "text": "最后输出带毫秒时间戳的结构化结果。"},
    {"id": 7, "start": 35.0, "end": 39.4, "text": "整个流程完全在本地完成，不上传视频。"},
]

detector = SceneDetector()
report = detector.extract_keyframes(video, video_id="demo", out_dir=base / "frames")
check("抽帧成功", len(report.keyframes) >= 4, f"{len(report.keyframes)} 帧")
check("场景检测成功", len(report.scenes) >= 3, f"{len(report.scenes)} 场景 via {report.detector_used}")

chunker = Chunker()
clips = chunker.chunk("demo", transcript, total_duration_ms=report.probe.duration_ms)
check("切片成功", len(clips) >= 2, f"{len(clips)} 片 {[c.core_duration_ms for c in clips]}")

# 关键帧 → clip
keyframes = SceneDetector.bind_clips(
    report.keyframes, [(c.clip_id, c.core_start_ms, c.core_end_ms) for c in clips]
)
check("关键帧全部绑定到切片", all(kf.clip_id for kf in keyframes),
      str([kf.clip_id for kf in keyframes]))
kf_by_clip = group_by_clip(keyframes)
clips = bind_keyframes(
    clips, [(kf.keyframe_id, kf.timestamp_ms) for kf in keyframes]
)
check("切片拿到 keyframe_ids", any(c.keyframe_ids for c in clips),
      str([len(c.keyframe_ids) for c in clips]))
check("双向绑定一致", {
    cid: sorted(ids) for cid, ids in kf_by_clip.items()
} == {c.clip_id: sorted(c.keyframe_ids) for c in clips if c.keyframe_ids},
      f"{kf_by_clip} vs {[c.keyframe_ids for c in clips]}")

# 溯源链完整性
rows = chunker.to_json(clips)
check("切片序列化完整", all(
    {"clip_id", "video_id", "start_time", "end_time", "text", "keyframe_ids"} <= set(r)
    for r in rows
), str(sorted(rows[0])))
kf_index = {kf.keyframe_id: kf for kf in keyframes}
check("每个 keyframe_id 可反查", all(
    fid in kf_index for r in rows for fid in r["keyframe_ids"]
))
check("关键帧文件可访问", all(kf_index[fid].path.is_file() for r in rows for fid in r["keyframe_ids"]))
check("引用误差 ≤2s", all(
    abs(kf_index[fid].timestamp_ms - c.core_start_ms) <= 2_000
    or abs(kf_index[fid].timestamp_ms - c.core_end_ms) <= 2_000
    or c.core_start_ms <= kf_index[fid].timestamp_ms <= c.core_end_ms
    for c in clips for fid in c.keyframe_ids
), "关键帧时间戳必须落在所属切片内")
check("时间格式统一", all(
    r["start_time"].count(":") == 2 and "." in r["start_time"] for r in rows
), rows[0]["start_time"])

out = base / "e2e.json"
out.write_text(json.dumps(
    {"video_id": "demo", "duration_ms": report.probe.duration_ms,
     "scenes": [s.to_dict() for s in report.scenes], "clips": rows},
    ensure_ascii=False, indent=2), encoding="utf-8")
check("结果文件已写出", out.is_file() and out.stat().st_size > 500, f"{out.stat().st_size}B")

print(f"\n通过 {len(ok)} / {len(ok) + len(fail)}")
if fail:
    print("失败项:")
    for item in fail:
        print("  - " + item)
sys.exit(0 if not fail else 1)
