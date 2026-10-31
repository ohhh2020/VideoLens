"""低分辨率流式扫描与按需高清取证；只使用已有FFmpeg/OpenCV。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import logging
import json
import re

logger = logging.getLogger("zhiying.analysis")

from backend.media.ffmpeg_utils import resolve_binary, probe_media


def metrics(previous, current):
    import cv2
    import numpy as np
    diff = cv2.absdiff(previous, current).astype('float32') / 255
    # 局部块取最大变化，避免小数字被全图平均淹没。
    rows, columns = diff.shape
    block = max(4, min(rows, columns) // 16)
    change = max(float(diff[y:y + block, x:x + block].mean())
                 for y in range(0, rows, block) for x in range(0, columns, block))
    left = cv2.Canny(previous, 50, 150)
    right = cv2.Canny(current, 50, 150)
    edge_diff = (left != right).astype('float32')
    edge = max(float(edge_diff[y:y + block, x:x + block].mean())
               for y in range(0, rows, block) for x in range(0, columns, block))
    motion = float(np.mean(diff > .08))
    return change, edge, motion


def scan(path, info, policy):
    import numpy as np
    source_width = getattr(info, 'display_width', info.width)
    source_height = getattr(info, 'display_height', info.height)
    width = min(policy.scan_long_side, source_width)
    height = max(2, round(source_height * width / max(1, source_width)))
    if source_height > source_width:
        height = min(policy.scan_long_side, source_height)
        width = max(2, round(source_width * height / max(1, source_height)))
    command = [resolve_binary('ffmpeg'), '-hide_banner', '-nostdin', '-loglevel', 'error',
               '-i', str(path), '-vf', f'fps={policy.scan_fps},scale={width}:{height},format=gray',
               '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    previous, index, last = None, 0, -policy.max_sample_gap_s
    candidates, motion_windows = [], []
    try:
        while True:
            data = process.stdout.read(width * height)
            if not data:
                break
            if len(data) != width * height:
                raise RuntimeError('扫描帧数据不完整')
            current = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
            timestamp = index / policy.scan_fps
            changed = False
            if previous is not None:
                delta, edge, motion = metrics(previous, current)
                changed = delta >= policy.change_threshold or edge >= policy.edge_threshold
                if motion >= .25:
                    motion_windows.append([max(0, int((timestamp - 1 / policy.scan_fps) * 1000)),
                                           min(info.duration_ms, int((timestamp + 1 / policy.scan_fps) * 1000))])
            if previous is None or timestamp - last >= policy.max_sample_gap_s or changed:
                candidates.append({'timestamp_ms': min(info.duration_ms - 1, int(timestamp * 1000)),
                                   'changed': changed})
                last = timestamp
            previous, index = current, index + 1
        if process.wait(timeout=30) != 0:
            raise RuntimeError('视频扫描失败')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    if not candidates:
        raise RuntimeError('视频没有可扫描的画面')
    candidates.append({'timestamp_ms': max(0, info.duration_ms - 100), 'changed': False})
    return candidates, motion_windows


def last_video_frame(path, info):
    """总时长可能包含更长音轨；用尾部真实视频PTS定位最后一帧。"""
    result = subprocess.run([resolve_binary('ffprobe'), '-v', 'error', '-read_intervals',
        f'{max(0, info.duration_ms/1000-10):.3f}%', '-select_streams', 'v:0',
        '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(path)], capture_output=True, timeout=60)
    if result.returncode:
        raise ValueError('无法确认尾帧实际位置')
    rows = json.loads(result.stdout).get('frames') or []
    times = [float(row['best_effort_timestamp_time']) for row in rows if row.get('best_effort_timestamp_time') is not None]
    if not times:
        raise ValueError('尾部缺少可用视频帧')
    offset = float((info.raw.get('format') or {}).get('start_time') or 0)
    return max(0, int((max(times)-offset)*1000)-1)


def refine_points(path, info, candidates, policy):
    """复查先走低分辨率流，只把真实变化点和区间边界送入高清取证。"""
    import numpy as np
    width, height = info.display_width, info.display_height
    scale = min(1, policy.scan_long_side/max(width,height))
    width, height = max(2,round(width*scale)), max(2,round(height*scale))
    intervals = []
    for row in candidates:
        if not row['changed']:
            continue
        lo = max(0,row['timestamp_ms']-policy.refine_window_s*1000)
        hi = min(info.duration_ms,row['timestamp_ms']+policy.refine_window_s*1000)
        if intervals and lo <= intervals[-1][1]:
            intervals[-1][1] = max(hi,intervals[-1][1])
        else:
            intervals.append([lo,hi])
    points = set()
    for lo, hi in intervals:
        command = [resolve_binary('ffmpeg'),'-nostdin','-hide_banner','-loglevel','error','-ss',f'{lo/1000:.3f}',
            '-i',str(path),'-t',f'{(hi-lo)/1000:.3f}','-vf',f'fps={policy.refine_fps},scale={width}:{height},format=gray',
            '-threads','1','-f','rawvideo','-pix_fmt','gray','pipe:1']
        process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        previous, index = None, 0
        try:
            while True:
                data = process.stdout.read(width*height)
                if not data:
                    break
                if len(data) != width*height:
                    raise ValueError('局部复查帧不完整')
                current = np.frombuffer(data,dtype=np.uint8).reshape(height,width)
                timestamp = int(lo+index*1000/policy.refine_fps)
                if previous is None:
                    points.add(timestamp)
                else:
                    delta, edge, _ = metrics(previous,current)
                    if delta >= policy.change_threshold or edge >= policy.edge_threshold:
                        points.add(timestamp)
                previous, index = current, index+1
            if process.wait(timeout=30):
                raise ValueError('局部信息密度复查失败')
            if index:
                points.add(int(lo+(index-1)*1000/policy.refine_fps))
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill();process.wait()
    return points


def extract(orch, path, video_id, clips, report, opts):
    import cv2
    from backend.segmentation.scenes import Keyframe

    ledger, policy = orch._analysis_run, orch._cfg.integrity
    info = probe_media(path)
    from backend.analysis.fast import enabled, sample_points
    fast = enabled(orch)
    saved = ledger.get('sampling')
    if fast:
        candidates = [{'timestamp_ms': value, 'changed': False} for value in sample_points(info.duration_ms, policy.max_sample_gap_s)]
        motion = []
        ledger.put('sampling', {'candidates': candidates, 'motion': motion, 'mode': 'fixed'})
    elif saved:
        candidates, motion = saved['candidates'], saved['motion']
    else:
        candidates, motion = scan(path, info, policy)
        ledger.put('sampling', {'candidates': candidates, 'motion': motion})
    # 每个变化区间复查，不因为达到某个全片固定数量而丢掉后半段。
    points = {max(0, row['timestamp_ms']): row['changed'] for row in candidates}
    fine = [] if fast else ledger.get('refined_points')
    if fine is None:
        fine = sorted(refine_points(path, info, candidates, policy))
        ledger.put('refined_points', fine)
    points.update({value: True for value in fine})
    last = last_video_frame(path, info)
    points = sorted({min(last, timestamp): changed for timestamp, changed in points.items()}.items())
    logger.info("%s：候选=%d，证据帧=%d，源=%dx%d", '固定间隔采样' if fast else '信息密度扫描完成', len(candidates), len(points), info.display_width, info.display_height)
    output = Path(orch._cfg.path('frames')) / video_id
    output.mkdir(parents=True, exist_ok=True)

    def take(item):
        index, (timestamp, changed) = item
        dest = output / f'{video_id}-kf-{timestamp:09d}.jpg'
        stamp_path = dest.with_suffix(".time.json")
        if not dest.is_file() or not stamp_path.is_file():
            from backend.analysis.resources import guard
            guard(orch._cfg, 'frames', policy.evidence_long_side**2 * 3)
            command = [resolve_binary('ffmpeg'), '-y', '-hide_banner', '-nostdin', '-loglevel', 'info', '-copyts',
                       '-ss', f'{timestamp/1000:.3f}', '-i', str(path), '-frames:v', '1',
                       '-vf', f"showinfo,scale='min({policy.evidence_long_side},iw)':'min({policy.evidence_long_side},ih)':force_original_aspect_ratio=decrease",
                       '-threads', str(orch._cfg.media.scene_detect.ffmpeg_threads), '-q:v', '2', str(dest)]
            result = subprocess.run(command, capture_output=True, timeout=90)
            if result.returncode or not dest.is_file():
                raise RuntimeError(f'取证帧提取失败：{timestamp}ms')
            found = re.search(r'pts_time:([-0-9.e+]+)', result.stderr.decode(errors='replace'))
            if not found:
                raise ValueError('提取帧缺少实际PTS，未生成虚假采样时间')
            start_offset = float((info.raw.get('format') or {}).get('start_time') or 0)
            actual_ms = round((float(found.group(1)) - start_offset)*1000)
            stamp_path.write_text(json.dumps({'requested_ms': timestamp, 'actual_ms': actual_ms}), encoding='utf-8')
            # JPEG按质量编码，不受旧200KiB上限压缩；最终质量由Pillow明确为92。
            from PIL import Image
            with Image.open(dest) as image:
                saved_image = image.copy()
            saved_image.save(dest, quality=92, subsampling=0)
            saved_image.close()
        actual_ms = json.loads(stamp_path.read_text(encoding="utf-8"))["actual_ms"]
        image = cv2.imread(str(dest))
        height, width = image.shape[:2]
        clip = next((clip for clip in clips if clip.core_start_ms <= actual_ms < clip.core_end_ms), None)
        scene = index
        return Keyframe(f'{video_id}-kf-{index:06d}', video_id, actual_ms, dest, width, height,
                        dest.stat().st_size, scene, clip.clip_id if clip else None,
                        requested_ms=timestamp, jpeg_quality=92,
                        notes=('局部变化复查',) if changed else ('基础时间覆盖',))

    with ThreadPoolExecutor(max_workers=orch._cfg.media.scene_detect.max_workers) as pool:
        frames = list(pool.map(take, enumerate(points)))
    frames = list({frame.timestamp_ms: frame for frame in frames}.values())
    if opts.max_keyframes and len(frames) > opts.max_keyframes:
        raise ValueError('显式关键帧上限不足以保留本轮候选，请提高上限；没有静默截取前N帧')
    ledger.put('motion_windows', motion)
    ledger.put('visual_coverage', {'type': 'frame_samples', 'sample_count': len(frames),
                                 'sampling_points_ms': [frame.timestamp_ms for frame in frames],
                                 'scan_fps': policy.scan_fps, 'source_width': info.width,
                                 'source_height': info.height, 'continuous_windows': [],
                                 'meaning': '离散采样不等于连续观看'})
    report.profile.update({'source_width': info.width, 'source_height': info.height})
    return frames
