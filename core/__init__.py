"""知影 (ZhiYing) —— 本地优先的视频智能分析助手。

本包只暴露最稳定的入口；内部分层与依赖方向（自下而上，只允许上层依赖下层）::

    core/
      media/       媒体处理：下载(yt-dlp) / 音视频解析(FFmpeg) / 场景检测抽帧
      models/      模型层：权重供应链(ModelManager) + 推理适配器
                   (asr_adapter / vl_adapter / embedding / llm_gateway)
      rag/         知识层：切片(chunker) / 向量库(vector_store) / 混合检索
      storage/     持久化：SQLite 元数据与切片关系 / 生命周期清理
      pipeline/    编排层：主流程控制器 / 超长上下文组装
      api/         服务层：FastAPI(HTTP+SSE) / MCP(JSON-RPC)

跨层铁律：

- ``media`` 与 ``models`` 互不 import，二者通过 pipeline 编排串联。
- ``rag`` 只依赖注入的 ``Embedder`` / 集合对象，不 import 任何具体模型。
- ``storage`` 不认识 ``models``；``pipeline`` 是唯一同时认识全部下层的模块。
- 任何对外输出的溯源信息必须携带 ``video_id`` / ``clip_id`` / 毫秒时间戳。
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
