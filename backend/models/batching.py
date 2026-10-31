"""单GPU推理锁内的自适应批次；OOM只缩批，绝不切设备或模型。"""
from __future__ import annotations

import logging

logger = logging.getLogger('zhiying.models')


def is_oom(exc):
    return any(word in str(exc).lower() for word in ('out of memory', 'cuda_error_out_of_memory', 'cublas_status_alloc_failed'))


def release_oom(exc):
    while exc is not None:
        exc.__traceback__ = None
        exc = exc.__cause__
    try:
        import torch
        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


class AdaptiveBatch:
    def __init__(self, maximum=16):
        self.maximum = maximum
        self.size = min(4, maximum)
        self.peak_per_item = 0
        self.ready_to_grow = False
        self.oom_limit = maximum
        self.oom_free = 0

    def run(self, items, fn, device='cpu', *, lengths=None):
        if not items:
            return []
        if not str(device).startswith('cuda'):
            return fn(items)
        import torch
        order = sorted(range(len(items)), key=lambda index: lengths[index]) if lengths else list(range(len(items)))
        output, offset = [None] * len(items), 0
        while offset < len(items):
            free, _ = torch.cuda.mem_get_info(device)
            if self.oom_free and free > self.oom_free * 1.3:
                self.oom_limit, self.oom_free = self.maximum, 0
            if self.ready_to_grow and self.peak_per_item and free > self.peak_per_item * self.size * 3:
                self.size = min(self.maximum, self.oom_limit, self.size * 2)
            size = min(self.size, self.maximum, self.oom_limit, len(items) - offset)
            selected = order[offset:offset + size]
            before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            try:
                values = fn([items[index] for index in selected])
            except Exception as exc:
                if not is_oom(exc) or size == 1:
                    raise
                release_oom(exc)
                self.size = max(1, size // 2)
                self.oom_limit, self.oom_free = self.size, free
                self.ready_to_grow = False
                logger.warning('GPU显存不足，批次 %d → %d；继续使用 %s', size, self.size, device)
                continue
            peak = max(1, torch.cuda.max_memory_allocated(device) - before)
            self.peak_per_item = max(self.peak_per_item, peak / size)
            self.ready_to_grow = size >= self.size
            if len(values) != size:
                raise ValueError('模型批次返回数量与输入不符')
            for index, value in zip(selected, values):
                output[index] = value
            offset += size
        return output


class GpuMemoryProbe:
    """采样设备空闲显存，包含CTranslate2分配；不冒充进程独占显存。"""
    def __init__(self, device):
        import threading
        self.device = device
        self.stop = threading.Event()
        self.minimum = self.before = 0
        self.thread = None

    def __enter__(self):
        if not str(self.device).startswith('cuda'):
            return self
        import torch
        import threading
        self.minimum = self.before = torch.cuda.mem_get_info(self.device)[0]
        def sample():
            while not self.stop.wait(.05):
                try:
                    self.minimum = min(self.minimum, torch.cuda.mem_get_info(self.device)[0])
                except RuntimeError:
                    break
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        if self.thread:
            self.thread.join()

    @property
    def used(self):
        return max(0, self.before - self.minimum)
