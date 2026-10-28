"""scene_detect 模块自检。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.segmentation.scenes import (  # noqa: E402
    SceneDetectConfig,
    SceneDetector,
    group_by_clip,
    ms_to_timecode,
)
from fixtures import ensure as ensure_fixtures  # noqa: E402

VIDEO, LOW = ensure_fixtures()
OUT = __import__("backend.runtime", fromlist=["runtime_path"]).runtime_path(
    "tests/media/frames"
)

ok: list[str] = []
fail: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        ok.append(name)
        print(f"PASS {name}" + (f" | {detail}" if detail else ""))
    else:
        fail.append(name)
        print(f"FAIL {name}" + (f" | {detail}" if detail else ""))


# 1) probe
detector = SceneDetector(SceneDetectConfig(max_workers=4))
info = detector.probe(VIDEO)
check("probe 时长", 39_000 <= info.duration_ms <= 41_000, f"{info.duration_ms}ms")
check(
    "probe 分辨率",
    (info.display_width, info.display_height) == (1280, 720),
    f"{info.display_width}x{info.display_height}",
)
check("probe fps", 24.0 <= info.fps <= 26.0, f"{info.fps}")
check("probe 音轨", info.has_audio is True)
check("短边 720", info.short_side == 720)

# 2) 场景检测（scenedetect）
t0 = time.monotonic()
scenes, used, degraded = detector.detect_scenes(VIDEO, probe=info)
elapsed = time.monotonic() - t0
check(
    "检测到 4 个场景",
    len(scenes) == 4,
    f"{len(scenes)} 段, {elapsed:.2f}s, detector={used}",
)
if len(scenes) == 4:
    bounds = [(s.start_ms, s.end_ms) for s in scenes]
    expected = [(0, 10_000), (10_000, 20_000), (20_000, 30_000), (30_000, 40_000)]
    drift = max(abs(a[0] - b[0]) for a, b in zip(bounds, expected))
    check("切点误差 <200ms", drift < 200, f"最大偏移 {drift}ms")
    check(
        "场景首尾覆盖完整",
        bounds[0][0] == 0 and abs(bounds[-1][1] - info.duration_ms) <= 30,
        f"{bounds[0][0]}~{bounds[-1][1]}",
    )
check("检测器为 scenedetect", used.startswith("scenedetect"), used)

# 3) ffmpeg 降级链路
ff_scenes, ff_used, ff_degraded = SceneDetector(
    SceneDetectConfig(detector="ffmpeg")
).detect_scenes(VIDEO, probe=info)
check(
    "ffmpeg 降级可用",
    ff_used == "ffmpeg:scene" and len(ff_scenes) >= 3,
    f"{len(ff_scenes)} 段 via {ff_used}",
)

# 4) fixed 降级
fixed = SceneDetector(SceneDetectConfig(detector="fixed", fixed_scene_ms=10_000))
fx_scenes, fx_used, _ = fixed.detect_scenes(VIDEO, probe=info)
check(
    "fixed 等分可用", fx_used == "fixed" and len(fx_scenes) == 4, f"{len(fx_scenes)} 段"
)

# 5) 抽帧主流程
import tempfile as _tempfile  # noqa: PLC0415

_probe_before = len(list(Path(_tempfile.gettempdir()).glob("zhiying-kf-*")))
t0 = time.monotonic()
report = detector.extract_keyframes(VIDEO, video_id="demo", out_dir=OUT)
cost = time.monotonic() - t0
kfs = report.keyframes
check("关键帧数量 4~8", 4 <= len(kfs) <= 8, f"{len(kfs)} 张, {cost:.2f}s")
check(
    "每场景至少 1 帧",
    all(len(report.frames_for_scene(i)) >= 1 for i in range(len(scenes))),
    str([(i, len(report.frames_for_scene(i))) for i in range(len(scenes))]),
)
check(
    "每场景最多 2 帧",
    all(len(report.frames_for_scene(i)) <= 2 for i in range(len(scenes))),
)
check(
    "全部 ≤200KB",
    all(kf.size_bytes <= 200 * 1024 for kf in kfs),
    f"最大 {max(kf.size_bytes for kf in kfs)}B",
)
check(
    "全部短边 ≥720",
    all(kf.short_side >= 720 for kf in kfs),
    f"最小 {min(kf.short_side for kf in kfs)}px",
)
check(
    "文件真实存在",
    all(kf.path.is_file() and kf.path.stat().st_size == kf.size_bytes for kf in kfs),
)
check(
    "时间戳单调递增",
    all(kfs[i].timestamp_ms <= kfs[i + 1].timestamp_ms for i in range(len(kfs) - 1)),
)
check(
    "时间戳落在场景内",
    all(any(s.start_ms <= kf.timestamp_ms <= s.end_ms for s in scenes) for kf in kfs),
)
check("溯源字段完整", all(kf.video_id == "demo" and kf.keyframe_id for kf in kfs))
check(
    "黑屏场景触发模糊降级",
    any("偏模糊" in w or "清晰度阈值" in w for w in report.warnings),
    " | ".join(report.warnings)[:160],
)
check(
    "候选数 ≥ 关键帧数",
    report.candidate_count >= len(kfs),
    f"{report.candidate_count} 候选",
)
check("报告可序列化", bool(json.dumps(report.to_dict(), ensure_ascii=False)))
# 探针目录：本轮抽帧结束后不应新增残留（对比运行前快照，历史遗留不算账）
_probe_after = list(Path(_tempfile.gettempdir()).glob("zhiying-kf-*"))
check(
    "探针目录已清理",
    len(_probe_after) <= _probe_before,
    f"运行前 {_probe_before} 个，运行后 {len(_probe_after)} 个",
)

# 6) 清晰度排序：黑屏帧方差应远小于图案帧
dark = [kf for kf in kfs if kf.scene_index == 3]
bright = [kf for kf in kfs if kf.scene_index in (0, 1, 2)]
if dark and bright:
    check(
        "黑屏帧清晰度更低",
        max(k.sharpness for k in dark) < min(k.sharpness for k in bright),
        f"dark={max(k.sharpness for k in dark):.1f} vs bright={min(k.sharpness for k in bright):.1f}",
    )

# 7) clip 绑定
clips = [
    ("demo-c0000", 0, 10_000),
    ("demo-c0001", 10_000, 20_000),
    ("demo-c0002", 20_000, 30_000),
    ("demo-c0003", 30_000, 40_000),
]
bound = SceneDetector.bind_clips(kfs, clips)
check(
    "clip 绑定成功", all(kf.clip_id for kf in bound), str([kf.clip_id for kf in bound])
)
grouped = group_by_clip(bound)
check(
    "按 clip 聚合",
    sum(len(v) for v in grouped.values()) == len(bound),
    str({k: len(v) for k, v in grouped.items()}),
)

# 8) 低分辨率源：应给警告但不放大
low = LOW
if low.is_file():
    low_detector = SceneDetector(
        SceneDetectConfig(detector="fixed", fixed_scene_ms=5_000)
    )
    low_report = low_detector.extract_keyframes(
        low, video_id="low", out_dir=OUT / "low"
    )
    check(
        "低分辨率不放大",
        all(kf.short_side <= 480 for kf in low_report.keyframes),
        f"{[(kf.width, kf.height) for kf in low_report.keyframes][:2]}",
    )
    check(
        "低分辨率有警告",
        any("低于关键帧" in w for w in low_report.warnings),
        " | ".join(low_report.warnings)[:120],
    )

# 9) 非法参数
raised = False
try:
    SceneDetectConfig(frames_per_scene=0).validate()
except ValueError:
    raised = True
check("拒绝非法配置", raised)

# 10) 时间格式
check(
    "毫秒时间码", ms_to_timecode(3_723_456) == "01:02:03.456", ms_to_timecode(3_723_456)
)

# 11) 候选去重：宁可漏判，不可把真换页删掉
_dedupe = SceneDetector(SceneDetectConfig())
_side = _dedupe._cfg.dedupe_probe_side
_dtmp = OUT / "dedupe"
_dtmp.mkdir(parents=True, exist_ok=True)


def _flat(path: Path, color, label: str = "") -> Path:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (640, 360), color)
    if label:
        draw = ImageDraw.Draw(img)
        try:
            base = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 60)
        except Exception:
            base = ImageFont.load_default()
        draw.text((80, 120), label, font=base, fill=(0, 0, 0))
    img.save(path, "PNG")
    return path


_red = _flat(_dtmp / "red.png", (220, 30, 30))
_blue = _flat(_dtmp / "blue.png", (30, 30, 220))
_red2 = _flat(_dtmp / "red2.png", (218, 32, 29))
_p1 = _flat(_dtmp / "p1.png", (250, 250, 252), "01")
_p2 = _flat(_dtmp / "p2.png", (250, 250, 252), "02")

check(
    "去重指纹尺寸",
    _dedupe._fingerprint(_red)[0].shape == (_side, _side),
    f"{_side}x{_side}",
)
check(
    "纯红 vs 纯蓝 不判重",
    not _dedupe._is_duplicate(_dedupe._fingerprint(_red), _dedupe._fingerprint(_blue)),
    "纯色 aHash 完全相同，全靠平均色区分",
)
check(
    "纯红 vs 红2 判重",
    _dedupe._is_duplicate(_dedupe._fingerprint(_red), _dedupe._fingerprint(_red2)),
)
check(
    "01 vs 02 不判重",
    not _dedupe._is_duplicate(_dedupe._fingerprint(_p1), _dedupe._fingerprint(_p2)),
)
check(
    "同一张图 判重",
    _dedupe._is_duplicate(_dedupe._fingerprint(_p1), _dedupe._fingerprint(_p1)),
)
check(
    "去重可整体关闭",
    SceneDetector(SceneDetectConfig(dedupe_frames=False))._cfg.dedupe_frames is False,
)

print(f"\n通过 {len(ok)} / {len(ok) + len(fail)}")
if fail:
    print("失败项:")
    for item in fail:
        print("  - " + item)
sys.exit(0 if not fail else 1)
