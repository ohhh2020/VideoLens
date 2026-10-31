"""常规画面、原像素OCR和连续视频分开取证，并保留各自完成状态。"""
from __future__ import annotations
from backend.analysis.prompts import get_prompt

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
from types import SimpleNamespace
import subprocess

from backend.analysis.cloud import request, parse_json
from backend.analysis.evidence import video_windows
from backend.media.ffmpeg_utils import resolve_binary, probe_media
from backend.pipeline.helpers import _keyframe_payload


def normalize_ocr(parsed, count):
    def flatten(value, depth=0):
        if depth > 6:
            raise ValueError('OCR容器嵌套过深')
        if isinstance(value, dict) and 'images' in value:
            return flatten(value['images'], depth+1)
        if isinstance(value, list):
            return [row for child in value for row in flatten(child, depth+1)]
        if isinstance(value, dict) and 'text' in value:
            return [value]
        raise ValueError('OCR结构包含无法识别的节点')
    data = flatten(parsed)
    if not isinstance(data, list) or not data or any(not isinstance(row, dict) or not isinstance(row.get('text'), str) for row in data):
        raise ValueError('OCR格式无效或未明确返回每张图片')
    if count == 1 and all(row.get('index') is None for row in data):
        # 单图的原生区域OCR协议，所有文字块有唯一图片来源；不猜多图归属。
        return [{'index': 0, 'text': '\n'.join(row['text'] for row in data),
                 'uncertain': any(row.get('uncertain', True) for row in data)}]
    indexes = {row.get('index') for row in data}
    if indexes == set(range(1, count + 1)):
        data = [{**row, 'index': row['index'] - 1} for row in data]
    if len(data) != count or {row.get('index') for row in data} != set(range(count)):
        raise ValueError('OCR缺少图片或编号无法明确关联')
    return data


def regions(path):
    import cv2
    image = cv2.imread(str(path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    mask = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)))
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 28 and h >= 8 and h <= height * .9 and w * h >= 400:
            density = cv2.countNonZero(edges[y:y+h, x:x+w]) / max(1, w*h)
            if .025 <= density <= .55:
                boxes.append([max(0, x-8), max(0, y-8), min(width, x+w+8), min(height, y+h+8)])
    if len(boxes) > 12:
        # 候选较多时用覆盖所有候选的网格，避免按数量砍掉尾部文字。
        boxes = [[x, y, min(width, x + width//2 + 16), min(height, y + height//2 + 16)]
                 for y in (0, height//2) for x in (0, width//2)]
    return sorted(boxes, key=lambda b: (b[1], b[0])), (width, height)


def crop_native(video_path, frame, box, preview_size, source_size, directory):
    sx, sy = source_size[0] / preview_size[0], source_size[1] / preview_size[1]
    x, y = int(box[0]*sx), int(box[1]*sy)
    right, bottom = min(source_size[0], int(box[2]*sx)), min(source_size[1], int(box[3]*sy))
    name = hashlib.sha256(f'{frame.timestamp_ms}:{x}:{y}:{right}:{bottom}'.encode()).hexdigest()[:20]
    path = directory / (name + '.png')
    if not path.is_file():
        result = subprocess.run([resolve_binary('ffmpeg'), '-y', '-nostdin', '-hide_banner', '-loglevel', 'error',
                                 '-ss', f'{max(0,frame.timestamp_ms-1)/1000:.3f}', '-i', str(video_path),
                                 '-frames:v', '1', '-vf', f'crop={max(1,right-x)}:{max(1,bottom-y)}:{x}:{y}',
                                 '-threads', '1', str(path)], capture_output=True, timeout=90)
        if result.returncode or not path.is_file():
            raise RuntimeError('原始像素文字裁图失败')
    return path, [x, y, right, bottom]


def describe(orch, video_path, video_id, frames, scenes, report, *, append=False, include_motion=True, progress=None):
    ledger, policy, vl = orch._analysis_run, orch._cfg.integrity, orch._vl
    info = probe_media(video_path)
    source_size = (getattr(info, 'display_width', info.width), getattr(info, 'display_height', info.height))
    runner = vl._runner
    directory = ledger.directory / 'ocr'
    directory.mkdir(exist_ok=True)
    visual_results = (ledger.get('visual_results') or []) if append else []

    def inference(paths, prompt, tokens, task):
        hashes = [hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths]
        return request(ledger, policy, task, {'hashes': hashes, 'prompt': prompt, 'tokens': tokens,
                                             'model': vl._model_name, 'endpoint': runner.endpoint, 'json_mode': task == '独立OCR'},
                       lambda: runner.describe_multi(paths, prompt, max_tokens=tokens, temperature=0.0, native=task == '独立OCR'), channel='vision')

    def ocr(paths, boxes, frame, format_retry=False):
        if not paths:
            return []
        prompt = (get_prompt('integrity_ocr', ledger))
        prompt += f' 本次实际只有{len(paths)}张输入图，images必须恰好{len(paths)}项，index从0开始，不得生成不存在的图片项。'
        if format_retry:
            prompt += ' 上次输出格式无法对应图片。请严格输出对象images数组，不要输出bbox，不要拆成文字块。'
        try:
            result = inference(paths, prompt, policy.ocr_tokens, '独立OCR')
        except Exception as exc:
            # 部分兼容端点在结构化生成中止时返回400，而非finish_reason=length。
            # 只识别此特定响应并拆图；认证、普通参数错误仍直接失败。
            message = str(exc).lower()
            if getattr(exc, 'code', None) != 400 or not ('json' in message and
                    ('model output became abnormal' in message or 'generation was aborted' in message)):
                raise
            result = SimpleNamespace(finish_reason='length', text='')
        try:
            if result.finish_reason == 'length':
                raise ValueError('OCR输出截断')
            data = normalize_ocr(parse_json(result.text), len(paths))
        except (ValueError, TypeError, KeyError):
            if result.finish_reason != 'length' and not format_retry:
                return ocr(paths, boxes, frame, format_retry=True)
            if len(paths) > 1:
                return [item for index in range(len(paths)) for item in ocr([paths[index]], [boxes[index]], frame)]
            if result.finish_reason != 'length':
                ledger.forget_request(result.cache_key)
                raise ValueError('单图OCR响应无法校验来源，请重试此窗口；原响应已保留')
            # 单张文字仍超长时，按原图上下半部带重叠继续识别。
            from PIL import Image
            with Image.open(paths[0]) as image:
                width, height = image.size
                if height <= 96:
                    raise ValueError('最小OCR区域仍不完整，请提高OCR输出预算')
                parts, mapped = [], []
                for number, (top, bottom) in enumerate(((0, height//2+8), (height//2-8, height))):
                    path = directory / f'{Path(paths[0]).stem}-split{height}-{number}.png'
                    image.crop((0, top, width, bottom)).save(path)
                    parts.append(path)
                    box = boxes[0]
                    mapped.append([box[0], box[1]+top, box[2], min(box[3], box[1]+bottom)])
            return ocr(parts[:1], mapped[:1], frame) + ocr(parts[1:], mapped[1:], frame)
        output = []
        for row in data:
            index = row['index']
            output.append({'kind': 'ocr', 'start_ms': frame.timestamp_ms, 'end_ms': frame.timestamp_ms,
                           'text': row['text'], 'uncertain': bool(row.get('uncertain')),
                           'frame_path': str(paths[index]), 'region': boxes[index],
                           'source_ids': [frame.keyframe_id], 'observation': 'frame_sample',
                           'raw_response': result.text, 'finish_reason': result.finish_reason})
        return output

    def frame_work(frame):
        prompt = (get_prompt('integrity_caption', ledger))
        response = inference([frame.path], prompt, 512, '画面概述')
        if response.finish_reason == 'length':
            raise ValueError('画面描述输出被截断')
        records = [{'kind': 'visual', 'start_ms': frame.timestamp_ms, 'end_ms': frame.timestamp_ms,
                    'text': response.text, 'frame_path': str(frame.path), 'source_ids': [frame.keyframe_id],
                    'observation': 'frame_sample'}]
        boxes, preview = regions(frame.path)
        crops, locations = [], []
        for box in boxes:
            from backend.analysis.resources import guard
            guard(orch._cfg, 'frames', source_size[0]*source_size[1]*3)
            path, mapped = crop_native(video_path, frame, box, preview, source_size, directory)
            crops.append(path)
            locations.append(mapped)
        for start in range(0, len(crops), policy.ocr_frames):
            records.extend(ocr(crops[start:start+policy.ocr_frames], locations[start:start+policy.ocr_frames], frame))
        text = '\n'.join(row['text'] for row in records if row['kind'] == 'ocr' and row['text'])
        return _keyframe_payload(frame, response.text, text), records

    # 同内容帧只调用一次，但每个采样点仍保留独立时间证据。
    grouped = {}
    for frame in frames:
        digest = hashlib.sha256(Path(frame.path).read_bytes()).hexdigest()
        grouped.setdefault(digest, []).append(frame)
    payloads = []
    with ThreadPoolExecutor(max_workers=policy.cloud_vision) as pool:
        first_frames = [items[0] for items in grouped.values()]
        values = [None] * len(first_frames)
        pending = {pool.submit(frame_work, frame): i for i, frame in enumerate(first_frames)}
        completed = 0
        for future in as_completed(pending):
            values[pending[future]] = future.result()
            completed += 1
            ledger.put('vision_progress', {'frames_done': completed, 'frames_total': len(first_frames)})
            if progress:
                progress('vision', .68 + .15 * completed / max(1, len(first_frames)), f'画面与独立OCR：{completed}/{len(first_frames)}组')
    for items, (payload, records) in zip(grouped.values(), values):
        for frame in items:
            payloads.append({**payload, 'keyframe_id': frame.keyframe_id, 'timestamp_ms': frame.timestamp_ms,
                             'path': str(frame.path), 'scene_index': frame.scene_index, 'clip_id': frame.clip_id})
            visual_results.extend({**row, 'start_ms': frame.timestamp_ms, 'end_ms': frame.timestamp_ms,
                                   'source_ids': [frame.keyframe_id]} for row in records)
    ledger.put('visual_results', visual_results)

    # 仅运动区间使用视频段；按整个时间轴处理，不使用前40窗截断。
    motion = ledger.get('motion_windows') or []
    if include_motion and motion and getattr(vl, 'supports_video_segments', False):
        spans = []
        for lo, hi in sorted(motion):
            if spans and lo <= spans[-1][1]:
                spans[-1][1] = max(hi, spans[-1][1])
            else:
                spans.append([lo, hi])
        windows = video_windows([SimpleNamespace(start_ms=lo, end_ms=hi) for lo, hi in spans], min(15, orch._cfg.vl.video_segment_max_s))
        for window in windows:
            result = request(ledger, policy, '连续画面', {'path': str(video_path), 'range': [window.start_ms, window.end_ms], 'model': vl._model_name, 'endpoint': runner.endpoint, 'json_mode': False},
                             lambda window=window: vl._segment_runner.describe_segment(video_path, window.start_ms, window.end_ms,
                                 get_prompt('integrity_motion', ledger)), channel='vision')
            if result.finish_reason == 'length':
                raise ValueError('连续画面输出被截断')
            visual_results.append({'kind': 'motion', 'start_ms': window.start_ms, 'end_ms': window.end_ms,
                                   'requested_start_ms': window.start_ms, 'requested_end_ms': window.end_ms,
                                   'actual_start_ms': result.actual_start_ms, 'actual_end_ms': result.actual_end_ms,
                                   'text': result.text, 'observation': 'continuous_video'})
        coverage = ledger.get('visual_coverage') or {}
        coverage['continuous_windows'] = [[window.start_ms, window.end_ms] for window in windows]
        ledger.put('visual_coverage', coverage)
    ledger.put('visual_results', visual_results)
    report.profile['vision_mode_used'] = 'adaptive-frames+independent-ocr'
    return sorted(payloads, key=lambda row: row['timestamp_ms'])
