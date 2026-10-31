"""仅对新版活动分析补充原像素取证；旧视频查询不触发媒体写入。"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import subprocess

from backend.analysis.runs import disk_check
from backend.media.ffmpeg_utils import resolve_binary


def supplement(orch, video, rows, ledger):
    # fast-v3 已撤销高清裁图/局部加密；问答也不能重新启用这一链路。
    if (ledger.get('policy') or {}).get('_analysis_strategy') == 'fast-v3':
        return []
    from backend.analysis.vision import describe
    from backend.analysis.summary import collect
    from backend.pipeline.domain import _video_lock
    from backend.segmentation.scenes import Keyframe
    from PIL import Image
    if not video.local_path or not Path(video.local_path).is_file():
        return []
    with _video_lock(video.video_id):
        current = orch._repo.get_video_report(video.video_id) or {}
        if current.get('profile', {}).get('analysis_run_id') != ledger.run_id:
            return []
        before = {row['evidence_id'] for row in ledger.all_evidence()}
        times = list(dict.fromkeys(min(max(0, row['start_ms'] + 500), video.duration_ms - 100) for row in rows))[:2]
        directory = ledger.directory / 'supplement'
        directory.mkdir(exist_ok=True)
        frames = []
        for timestamp in times:
            path = directory / f'{timestamp}.jpg'
            if not path.is_file():
                disk_check(directory, orch._cfg.integrity.disk_reserve_bytes)
                proc = subprocess.run([resolve_binary('ffmpeg'), '-y', '-nostdin', '-loglevel', 'error',
                    '-ss', f'{timestamp/1000:.3f}', '-i', video.local_path, '-frames:v', '1', '-threads', '1', str(path)],
                    capture_output=True, timeout=90)
                if proc.returncode or not path.is_file():
                    raise ValueError('问答补帧失败，未伪造视觉证据')
            with Image.open(path) as image:
                width, height = image.size
            cid = next((row['clip_id'] for row in rows if row['start_ms'] <= timestamp <= row['end_ms']), rows[0]['clip_id'])
            frames.append(Keyframe('supp-' + hashlib.sha256(f'{ledger.run_id}:{timestamp}'.encode()).hexdigest()[:20],
                video.video_id, timestamp, path, width, height, path.stat().st_size, 0, cid))
        import copy
        worker = copy.copy(orch)
        worker._analysis_run = ledger
        describe(worker, Path(video.local_path), video.video_id, frames, [], SimpleNamespace(profile={}), append=True, include_motion=False)
        collect(orch._repo, video.video_id, ledger, orch._cfg.integrity)
        fresh = [row for row in ledger.all_evidence() if row['evidence_id'] not in before]
        if fresh and orch._vectors and video.vector_generation:
            vectors = orch._embedder.embed_texts([row['text'] for row in fresh])
            orch._vectors.with_generation(video.vector_generation).upsert_clips(
                ids=[row['evidence_id'] for row in fresh], embeddings=vectors, documents=[row['text'] for row in fresh],
                metadatas=[{key: row[key] for key in ('clip_id', 'video_id', 'evidence_id', 'kind', 'start_ms', 'end_ms')} for row in fresh])
        ledger.put('supplemented', {'times_ms': times, 'evidence_ids': [row['evidence_id'] for row in fresh]})
        return fresh
