"""按音频内容与真实ASR参数复用转写，缓存不改变识别设备或模型。"""
from dataclasses import asdict, fields
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import threading

logger = logging.getLogger('zhiying.analysis')
_lock = threading.RLock()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(orch, audio):
    config = orch._asr.config
    parameters = asdict(config)
    # 权重更换或ASR配置变更会失效；不读取或散列数GB权重正文。
    model = Path(orch._asr._model_path())
    weights = [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
               for p in sorted(model.glob('*')) if p.is_file()]
    value = {'version': 1, 'audio_sha256': digest_file(audio), 'asr': parameters,
             'weights': weights, 'model_directory': str(model)}
    key = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
    return key, value


def restore(value, video_id):
    from backend.models.asr import Transcript, TranscriptSegment, Word
    payload = {key: item for key, item in value.items() if key in {f.name for f in fields(Transcript)}}
    segments = []
    for row in payload.pop('segments', []):
        row = dict(row)
        row['words'] = tuple(Word(**word) for word in row.get('words', []))
        segments.append(TranscriptSegment(**row))
    payload.update(video_id=video_id, segments=tuple(segments))
    payload['warnings'] = tuple(payload.get('warnings', []))
    if payload.get('subtitle_path'):
        payload['subtitle_path'] = Path(payload['subtitle_path'])
    return Transcript(**payload)


def read(orch, key, video_id):
    path = orch._cfg.path('data') / 'transcripts' / (key + '.json')
    try:
        with _lock:
            value = json.loads(path.read_text(encoding='utf-8'))
            if value['key'] != key:
                return None
            transcript = restore(value['transcript'], video_id)
            if transcript.is_empty or transcript.degraded:
                return None
            owners = set(value.get('video_ids', [value['transcript']['video_id']]))
            if video_id not in owners:
                value['video_ids'] = sorted(owners | {video_id})
                _atomic(path, value)
            return transcript
    except (OSError, ValueError, TypeError, KeyError):
        return None


def write(orch, key, identity_value, transcript):
    if transcript.is_empty or transcript.degraded:
        return
    directory = orch._cfg.path('data') / 'transcripts'
    directory.mkdir(parents=True, exist_ok=True)
    value = {'key': key, 'identity': identity_value, 'transcript': transcript.to_dict(), 'video_ids': [transcript.video_id]}
    text = json.dumps(value, ensure_ascii=False)
    from backend.analysis.resources import guard
    guard(orch._cfg, 'results', len(text.encode('utf-8')))
    with _lock:
        target = directory / (key + '.json')
        try:
            previous = json.loads(target.read_text(encoding='utf-8'))
            value['video_ids'] = sorted(set(previous.get('video_ids', [])) | set(value['video_ids']))
        except (OSError, ValueError, TypeError):
            pass
        _atomic(directory / (key + '.json'), value)


def _atomic(path, value):
    text = json.dumps(value, ensure_ascii=False)
    stream = tempfile.NamedTemporaryFile('w', dir=path.parent, prefix='.transcript-', encoding='utf-8', delete=False)
    try:
        with stream:
            stream.write(text)
        os.replace(stream.name, path)
    finally:
        Path(stream.name).unlink(missing_ok=True)


def purge(orch, video_id):
    """联动删除时清理仅属于该视频的缓存，仍被其他视频使用时保留。"""
    count = 0
    with _lock:
        for path in (orch._cfg.path('data') / 'transcripts').glob('*.json'):
            value = json.loads(path.read_text(encoding='utf-8'))
            owners = set(value.get('video_ids', [value['transcript']['video_id']]))
            if video_id not in owners:
                continue
            owners.discard(video_id)
            if owners:
                value['video_ids'] = sorted(owners)
                _atomic(path, value)
            else:
                path.unlink()
                count += 1
    return count
