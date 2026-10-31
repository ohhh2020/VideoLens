"""新任务的不可变质量策略；不会部署额外本地模型。"""
from dataclasses import dataclass, fields
import math


@dataclass(frozen=True, slots=True)
class IntegritySection:
    enabled: bool = True
    scan_long_side: int = 640
    scan_fps: float = 1.0
    max_sample_gap_s: float = 10.0
    refine_window_s: float = 2.0
    refine_fps: float = 4.0
    change_threshold: float = 0.04
    edge_threshold: float = 0.08
    evidence_long_side: int = 1920
    ocr_frames: int = 4
    ocr_tokens: int = 2048
    summary_batch_target: int = 24
    summary_batch_max: int = 32
    summary_input_tokens: int = 12000
    summary_output_tokens: int = 2048
    report_input_tokens: int = 24000
    report_output_tokens: int = 4096
    qa_local_k: int = 6
    qa_global_k: int = 20
    qa_local_input_tokens: int = 8000
    qa_global_input_tokens: int = 24000
    qa_local_output_tokens: int = 2048
    qa_global_output_tokens: int = 4096
    qa_extra_rounds: int = 2
    max_continuations: int = 2
    cloud_total: int = 8
    cloud_text: int = 4
    cloud_vision: int = 8
    cloud_retries: int = 3
    gpu_batch_max: int = 16
    asr_batch_max: int = 8
    upload_chunk_bytes: int = 8 * 1024 * 1024
    upload_ttl_hours: int = 24
    disk_reserve_bytes: int = 5 * 1024**3
    warning_requests: int = 200
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0

    def validate(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool):
                continue
            if not math.isfinite(value):
                raise ValueError(f'integrity.{field.name} 必须是有限数值')
            if value < 0:
                raise ValueError(f'integrity.{field.name} 不能为负数')
            if field.name not in {'disk_reserve_bytes', 'qa_extra_rounds', 'max_continuations',
                                  'cloud_retries', 'input_price_per_million', 'output_price_per_million'} and value == 0:
                raise ValueError(f'integrity.{field.name} 必须大于零')
        if not 1 <= self.ocr_frames <= 4:
            raise ValueError('OCR每批必须为1至4张')
        if not 1 <= self.summary_batch_target <= self.summary_batch_max <= 32:
            raise ValueError('摘要批次要求 1 ≤ 目标 ≤ 上限 ≤ 32')
        if self.refine_fps < self.scan_fps or self.evidence_long_side < self.scan_long_side:
            raise ValueError('高清复查规格不能低于扫描规格')
        if self.qa_extra_rounds > 2 or self.max_continuations > 2:
            raise ValueError('追加检索和续写最多2次')
        if self.upload_chunk_bytes > 32 * 1024**2:
            raise ValueError('上传分块不能超过32MiB')
        if self.upload_chunk_bytes != 8 * 1024**2:
            raise ValueError('续传分块大小固定为8MiB')
        for key, low, high in (
            ('scan_long_side', 160, 1280), ('scan_fps', .1, 4), ('max_sample_gap_s', 1, 60),
            ('refine_window_s', .25, 5), ('refine_fps', 1, 8), ('evidence_long_side', 640, 3840),
            ('change_threshold', .001, 1), ('edge_threshold', .001, 1),
            ('qa_local_k', 1, 20), ('qa_global_k', 12, 20), ('gpu_batch_max', 1, 16),
            ('asr_batch_max', 1, 8), ('cloud_total', 1, 32), ('cloud_text', 1, 32),
            ('cloud_vision', 1, 32), ('cloud_retries', 0, 3),
        ):
            if not low <= getattr(self, key) <= high:
                raise ValueError(f'integrity.{key} 必须在{low}至{high}之间')
        for prefix in ('summary', 'report', 'qa_local', 'qa_global'):
            if getattr(self, prefix + '_input_tokens') < getattr(self, prefix + '_output_tokens') + 2048:
                raise ValueError(f'{prefix}总预算必须至少比输出预算多2048')
        for name in ('summary_input_tokens', 'report_input_tokens', 'qa_local_input_tokens',
                     'qa_global_input_tokens', 'summary_output_tokens', 'report_output_tokens',
                     'qa_local_output_tokens', 'qa_global_output_tokens', 'ocr_tokens'):
            if not 128 <= getattr(self, name) <= 131072:
                raise ValueError(f'{name} 必须在128至131072之间')


VERSION = 'fast-v3'


def estimate_tokens(text):
    """未下载额外 tokenizer；UTF-8字节数作为保守预算，实际usage另记。"""
    return len(str(text).encode('utf-8'))
