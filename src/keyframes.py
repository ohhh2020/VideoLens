from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import Keyframe


class KeyframeError(RuntimeError):
    pass


def _extract_times(duration: float, interval: int, max_frames: int) -> list[float]:
    if duration <= 0:
        return [0.0]

    uniform = list(np.arange(0.0, duration, max(1, interval)))
    if duration - 0.5 > 0:
        uniform.append(round(duration - 0.5, 2))
    uniform = sorted({round(max(0.0, min(value, duration)), 2) for value in uniform})

    if len(uniform) <= max_frames:
        return uniform

    indexes = np.linspace(0, len(uniform) - 1, max_frames).round().astype(int)
    return [uniform[int(index)] for index in sorted(set(indexes.tolist()))]


def _detect_scene_times(
    cap,
    duration: float,
    threshold: float,
    max_scenes: int,
) -> list[float]:
    import cv2

    scene_times: list[float] = []
    previous_hist = None
    previous_time = -10.0

    step = 1.0
    for second in np.arange(0.0, duration, step):
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)

        if previous_hist is not None:
            correlation = cv2.compareHist(previous_hist, hist, cv2.HISTCMP_CORREL)
            if correlation < threshold and second - previous_time >= 2.0:
                scene_times.append(round(second, 2))
                previous_time = second
        previous_hist = hist

        if len(scene_times) >= max_scenes:
            break

    return scene_times


def extract_keyframes(
    video_path: Path,
    output_dir: Path,
    interval: int = 8,
    max_frames: int = 20,
    scene_threshold: float = 0.60,
) -> list[Keyframe]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise KeyframeError(f"无法打开视频文件：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps > 0 else 0.0

    if duration <= 0:
        cap.release()
        raise KeyframeError("无法读取视频时长。")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_times = _extract_times(duration, interval, max_frames)

    # 中短视频额外检测场景变化，参照“运动剧烈片段优先”的思想。
    if duration <= 1800 and len(all_times) < max_frames:
        scene_times = _detect_scene_times(cap, duration, scene_threshold, max_frames)
        all_times.extend(scene_times)
        all_times = sorted({round(value, 2) for value in all_times if 0 <= value <= duration})

    if len(all_times) > max_frames:
        indexes = np.linspace(0, len(all_times) - 1, max_frames).round().astype(int)
        all_times = [all_times[int(index)] for index in sorted(set(indexes.tolist()))]

    results: list[Keyframe] = []
    for index, timestamp in enumerate(all_times, start=1):
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = cap.read()
        if not ok:
            continue

        filename = f"keyframe_{index:03d}_{timestamp:.1f}s.jpg"
        path = output_dir / filename
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        results.append(Keyframe(path=str(path), timestamp=timestamp, scene_score=0.0))

    cap.release()
    return results

