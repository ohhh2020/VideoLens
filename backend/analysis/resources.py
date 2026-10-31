"""统一统计运行缓存与业务产物；容量不足时停止写入，不自动删业务数据。"""
import os
from pathlib import Path
import shutil
import threading
import time

_lock = threading.Lock()
_cache = {}


def inventory(config, *, refresh=False):
    root = config.path('data').resolve()
    with _lock:
        previous = _cache.get(str(root))
        if not refresh and previous and time.monotonic() - previous[0] < 2:
            return dict(previous[1])
        result = dict(source_bytes=0, frame_bytes=0, audio_bytes=0, result_bytes=0, upload_partial_bytes=0, other_bytes=0)
        for directory, _, files in os.walk(root):
            for name in files:
                path = Path(directory) / name
                try:
                    # 本地视频软链接只占链接本身的空间，不能累计原视频大小。
                    size = path.lstat().st_size
                except OSError:
                    continue
                parts = set(path.relative_to(root).parts[:-1])
                kind = ('upload_partial_bytes' if path.suffix == '.part' and 'uploads' in parts else
                        'frame_bytes' if parts.intersection({'frames', 'ocr', 'supplement'}) else
                        'audio_bytes' if 'audio' in parts else
                        'source_bytes' if parts.intersection({'source', 'videos', 'uploads'}) else
                        'result_bytes' if path.suffix in {'.sqlite3', '.db', '.json', '.md'} else 'other_bytes')
                result[kind] += size
        disk_root = root
        while not disk_root.exists():
            disk_root = disk_root.parent
        result['free_bytes'] = shutil.disk_usage(disk_root).free
        result['total_bytes'] = sum(value for key, value in result.items() if key != 'free_bytes')
        _cache[str(root)] = (time.monotonic(), result)
        return dict(result)


def guard(config, kind='', incoming=0):
    from backend.analysis.runs import disk_check
    disk_check(config.path('data'), config.integrity.disk_reserve_bytes, incoming)
    if kind:
        stats = inventory(config)
        key = 'frame_bytes' if kind == 'frames' else 'result_bytes'
        maximum = config.storage.limits.max_keyframe_bytes if kind == 'frames' else config.storage.limits.max_result_bytes
        if maximum and stats[key] + incoming > maximum:
            raise OSError(('关键帧/OCR' if kind == 'frames' else '分析结果') + '容量达到上限，请通过视频库管理缓存；已有业务数据不会被自动删除')
