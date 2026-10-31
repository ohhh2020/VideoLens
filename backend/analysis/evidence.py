"""证据原文与时间语义：展示、检索和压缩文本不覆盖原文。"""
from dataclasses import asdict, dataclass, field
import hashlib
from backend.analysis.policy import VERSION


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    run_id: str
    video_id: str
    clip_id: str
    kind: str
    start_ms: int
    end_ms: int
    text: str
    analysis_version: str = VERSION
    frame_path: str = ''
    region: list[int] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    observation: str = 'transcript'

    def to_dict(self):
        return asdict(self)


def split_text(text, max_bytes=6000):
    """按换行/句末优先切分，保证所有字符均能从子证据重建。"""
    if not text:
        return []
    chunks, current, used = [], [], 0
    for char in text:
        width = len(char.encode('utf-8'))
        if used + width > max_bytes and current:
            chunks.append(''.join(current))
            current, used = [], 0
        current.append(char)
        used += width
        if used >= max_bytes // 2 and char in '\n。！？；':
            chunks.append(''.join(current))
            current, used = [], 0
    if current:
        chunks.append(''.join(current))
    return chunks


def make_evidence(run_id, video_id, clip_id, kind, start, end, text, **extra):
    digest = hashlib.sha256(f'{run_id}:{clip_id}:{kind}:{start}:{end}:{text}:{extra}'.encode()).hexdigest()[:24]
    return Evidence('e' + digest, run_id, video_id, clip_id, kind, int(start), int(end), str(text), **extra)


def compose(item):
    """三个来源分别存在时显式组装；兼容旧的纯文本切片。"""
    if item.get('evidence_text') is not None:
        return item['evidence_text']
    speech = item.get('core_text') or item.get('text') or ''
    parts = [speech]
    for label, key in [('画面观察', 'visual_text'), ('画面原文', 'ocr_text')]:
        if item.get(key):
            parts.append(f'【{label}】\n{item[key]}')
    return '\n'.join(p for p in parts if p)


@dataclass(frozen=True)
class VideoWindow:
    start_ms: int
    end_ms: int
    scene_ids: tuple[int, ...]


def video_windows(scenes, max_s=15):
    """覆盖每个场景所有时间，单场景也必须继续拆分。"""
    span = max(1, int(max_s * 1000))
    windows = []
    for index, scene in enumerate(scenes):
        start, end = int(scene.start_ms), int(scene.end_ms)
        while start < end:
            stop = min(end, start + span)
            if windows and windows[-1].end_ms == start and stop - windows[-1].start_ms <= span:
                old = windows.pop()
                windows.append(VideoWindow(old.start_ms, stop, tuple(dict.fromkeys((*old.scene_ids, index)))))
            else:
                windows.append(VideoWindow(start, stop, (index,)))
            start = stop
    return windows
