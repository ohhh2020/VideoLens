"""复用PyAV解码器按时间定位；边产帧边交给视觉队列，保留真实PTS。"""
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
from queue import Queue, Empty, Full
import threading
import time

from backend.analysis.fast import sample_points
from backend.analysis.sampling import last_video_frame
from backend.segmentation.scenes import Keyframe

logger = logging.getLogger('zhiying.analysis')


def stream_frames(orch, path, video_id, info, report, opts):
    policy, settings = orch._cfg.integrity, orch._cfg.media.scene_detect
    ledger = orch._analysis_run
    last = last_video_frame(path, info)
    points = sorted({min(last, point) for point in sample_points(info.duration_ms, policy.max_sample_gap_s)})
    if opts.max_keyframes and len(points) > opts.max_keyframes:
        raise ValueError('显式关键帧上限不足以保留固定采样点，没有截取前N帧')
    directory = Path(orch._cfg.path('frames')) / video_id
    directory.mkdir(parents=True, exist_ok=True)
    workers = min(settings.max_workers, len(points))
    queue, stop = Queue(maxsize=max(2, workers * 2)), threading.Event()
    offset = float((info.raw.get('format') or {}).get('start_time') or 0)
    from backend.analysis.resources import guard
    guard(orch._cfg, 'frames', len(points) * policy.evidence_long_side**2 // 2)

    def send(value):
        while not stop.is_set():
            try:
                queue.put(value, timeout=.1)
                return
            except Full:
                pass

    def worker(entries):
        import av
        from PIL import Image
        container = None
        try:
            for index, target in entries:
                if stop.is_set():
                    break
                destination = directory / f'{video_id}-kf-{target:09d}.jpg'
                stamp = destination.with_suffix('.time.json')
                if destination.is_file() and stamp.is_file():
                    actual = json.loads(stamp.read_text(encoding='utf-8'))['actual_ms']
                    with Image.open(destination) as image:
                        width, height = image.size
                else:
                    if container is None:
                        container = av.open(str(path))
                        stream = container.streams.video[0]
                        stream.codec_context.thread_count = settings.ffmpeg_threads
                    wanted = target / 1000 + offset
                    container.seek(max(0, int(wanted / stream.time_base)), stream=stream, backward=True, any_frame=False)
                    found = None
                    for frame in container.decode(stream):
                        if frame.pts is None:
                            continue
                        timestamp = float(frame.pts * frame.time_base)
                        if timestamp + 1e-7 >= wanted:
                            found = frame
                            break
                    if found is None:
                        raise ValueError(f'采样点{target}ms没有有效帧，未伪造时间')
                    actual = round((float(found.pts * found.time_base) - offset) * 1000)
                    image = found.to_image()
                    if info.rotation % 360:
                        image = image.rotate(info.rotation, expand=True)
                    image.thumbnail((policy.evidence_long_side, policy.evidence_long_side), Image.Resampling.LANCZOS)
                    width, height = image.size
                    temporary = destination.with_suffix('.pending.jpg')
                    image.save(temporary, quality=92, subsampling=0)
                    image.close()
                    temporary.replace(destination)
                    stamp.write_text(json.dumps({'requested_ms': target, 'actual_ms': actual}), encoding='utf-8')
                send(('frame', Keyframe(f'{video_id}-kf-{index:06d}', video_id, actual, destination,
                     width, height, destination.stat().st_size, index, None, requested_ms=target,
                     jpeg_quality=92, notes=('固定时间采样，复用解码器',))))
        except BaseException as exc:
            send(('error', exc))
        finally:
            if container is not None:
                container.close()
            send(('done', None))

    started, frames = time.monotonic(), []
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='frame-decode')
    futures = [pool.submit(worker, list(enumerate(points))[i::workers]) for i in range(workers)]
    try:
        finished = 0
        while finished < workers:
            try:
                kind, value = queue.get(timeout=.2)
            except Empty:
                if all(future.done() for future in futures):
                    raise RuntimeError('抽帧队列提前结束')
                continue
            if kind == 'error':
                raise value
            if kind == 'done':
                finished += 1
            else:
                frames.append(value)
                yield value
        if len(frames) != len(points):
            raise RuntimeError('抽帧数量不完整')
        frames.sort(key=lambda frame: frame.timestamp_ms)
        ledger.put('visual_coverage', {'type': 'frame_samples', 'sample_count': len(frames),
            'sampling_points_ms': [frame.timestamp_ms for frame in frames],
            'requested_points_ms': points, 'source_width': info.display_width, 'source_height': info.display_height,
            'continuous_windows': [], 'meaning': '离散采样，不等于连续观看', 'decoder': 'pyav-reused'})
        ledger.put('sampling', {'mode': 'fixed', 'decoder': 'pyav-reused',
                              'candidates': [{'timestamp_ms': value, 'changed': False} for value in points], 'motion': []})
        report.profile['frame_extract_ms'] = int((time.monotonic() - started) * 1000)
        report.profile['decoder_instances'] = workers
        logger.info('抽帧完成 %.3fs frames=%d video=%s（与云端识别重叠，含队列等待）',
                    report.profile['frame_extract_ms'] / 1000, len(frames), video_id)
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
