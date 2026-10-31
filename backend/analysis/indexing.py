"""每条子证据参与文本索引，避免长文与OCR只编码前半部分。"""
from pathlib import Path
from uuid import uuid4
import time

from backend.analysis.summary import collect


def build_index(orch, video_id, clips, frames, report):
    ledger, policy = orch._analysis_run, orch._cfg.integrity
    _, evidence = collect(orch._repo, video_id, ledger, policy)
    cached = ledger.get('index')
    if cached:
        store = orch._vectors.with_generation(cached['generation'])
        if store.count('clip') == cached['clips'] and store.count('frame') == cached['frames']:
            orch._repo.update_video_meta(video_id, vector_generation=cached['generation'])
            return cached['written']
    generation = uuid4().hex[:12]
    store = orch._vectors.with_generation(generation)
    try:
        texts = [row['text'] for row in evidence]
        start = time.monotonic()
        with orch._inference_slot('全量证据文本向量', report):
            vectors = orch._embedder.embed_texts(texts, max_batch=policy.gpu_batch_max)
        report.profile['text_encode_ms'] = int((time.monotonic()-start)*1000)
        written = store.upsert_clips(
            ids=[row['id'] for row in evidence], embeddings=vectors, documents=texts,
            metadatas=[{'clip_id': row['clip_id'], 'video_id': video_id,
                        'evidence_id': row['id'], 'kind': row['kind'],
                        'start_ms': row['start_ms'], 'end_ms': row['end_ms'],
                        'core_start_ms': row['start_ms'], 'core_end_ms': row['end_ms']} for row in evidence])
        if frames:
            start = time.monotonic()
            with orch._inference_slot('全量证据图像向量', report):
                image_vectors = orch._embedder.embed_images([Path(frame.path) for frame in frames], max_batch=policy.gpu_batch_max)
            report.profile['image_encode_ms'] = int((time.monotonic()-start)*1000)
            written += store.upsert_frames(
                ids=[frame.keyframe_id for frame in frames], embeddings=image_vectors,
                documents=[str(frame.timestamp_ms) for frame in frames],
                metadatas=[{'keyframe_id': frame.keyframe_id, 'clip_id': frame.clip_id or '',
                            'video_id': video_id, 'timestamp_ms': frame.timestamp_ms,
                            'keyframe_path': str(frame.path)} for frame in frames])
        ledger.put('index', {'generation': generation, 'clips': len(evidence), 'frames': len(frames), 'written': written})
        orch._repo.update_video_meta(video_id, vector_generation=generation)
        return written
    except BaseException:
        store.drop_generation(generation)
        raise
