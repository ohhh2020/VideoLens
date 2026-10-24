"""知影 ZhiYing —— 分析管道层（把各模块串成一条可恢复的流水线）。

两层职责
--------
- ``context_builder``：**只做拼装**。把检索回来的候选切片去重、截断、按 token
  预算装进提示词，产出 ``BuiltContext``（正文 + 编号引用 + 超预算丢弃记录）。
  它不认识 LLM，也不发起检索 —— 纯函数式变换，方便单测。
- ``orchestrator``：**只做调度**。按 ``parse → probe → audio → transcribe →
  scene → chunk → vision → index → summary`` 推进，每阶段写数据库状态与检查点，
  失败逐级降级。它不认识 Chroma 的内部结构，只通过 ``VectorStore`` /
  ``HybridSearcher`` 的接口调用。

依赖方向
--------
::

    core.media ─┐
    core.models ─┼→ core.rag ─→ core.storage ─→ core.pipeline ─→ api / clients
                │                                              ↑
                └──────────────────────────────────────────────┘

管道层是**唯一**知道"一条视频要经过哪些阶段"的地方。上层（FastAPI、Gradio、
CLI、MCP）只调用 :class:`Orchestrator`，不直接碰 ``SceneDetector`` 或
``AsrEngine`` —— 这样换掉任一实现都不影响接口。

典型用法
--------
>>> from core.pipeline import Orchestrator, AnalyzeOptions
>>> orch = Orchestrator.from_config()
>>> report = orch.analyze("D:/videos/demo.mp4", AnalyzeOptions(skip_vision=True))
>>> report.status, report.clip_count
('ready', 2)
>>> ans = orch.ask(report.video_id, "讲了什么")
>>> [c.time_range for c in ans.refs]
['00:00:00.000–00:00:20.100']
"""

from __future__ import annotations

from core.pipeline.context_builder import (
    DEFAULT_CHARS_PER_TOKEN,
    BuiltContext,
    ContextBudget,
    ContextBuilder,
    ContextItem,
    format_hms,
    spread_order,
)
from core.pipeline.orchestrator import (
    AnalysisReport,
    AnalyzeOptions,
    AskResult,
    Citation,
    Orchestrator,
    PipelineError,
    Stage,
    StageReport,
    compute_fingerprint,
    extract_citations,
)

__all__ = [
    # context_builder
    "DEFAULT_CHARS_PER_TOKEN",
    "BuiltContext",
    "ContextBudget",
    "ContextBuilder",
    "ContextItem",
    "format_hms",
    "spread_order",
    # orchestrator
    "AnalysisReport",
    "AnalyzeOptions",
    "AskResult",
    "Citation",
    "Orchestrator",
    "PipelineError",
    "Stage",
    "StageReport",
    "compute_fingerprint",
    "extract_citations",
]
