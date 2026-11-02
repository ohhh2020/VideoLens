"""独立环境启动：python integrations/videolens_mcp/server.py；仅stdio。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mcp.server.fastmcp import FastMCP
from integrations.videolens_mcp.bridge import Bridge

bridge = Bridge()
server = FastMCP('VideoLens')


@server.tool()
def videolens_analyze(url: str, session_key: str, request_id: str, question: str = '') -> dict:
    """提交Bilibili视频分析，立即返回任务；使用result查询，不等待长分析。"""
    return bridge.analyze(url, session_key, request_id, question)


@server.tool()
def videolens_ask(session_key: str, question: str, request_id: str) -> dict:
    """在已完成的视频消息会话中追问，返回异步任务。"""
    return bridge.ask(session_key, question, request_id)


@server.tool()
def videolens_result(job_id: str) -> dict:
    """读取桥接自己提交的任务进度、报告和问答Markdown。"""
    return bridge.result(job_id)


@server.tool()
def videolens_status() -> dict:
    """检查本机VideoLens是否就绪及实际模型设备，不返回密钥。"""
    return bridge.status()


if __name__ == '__main__':
    server.run(transport='stdio')
