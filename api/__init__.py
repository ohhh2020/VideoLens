"""知影 HTTP / MCP 服务层。

只有这一层认识网络协议（HTTP 状态码、SSE 帧、JSON-RPC）；下层
``core.*`` 完全不感知传输方式，因此同一套能力可以被 CLI、Gradio、MCP 复用。

模块：

- :mod:`api.routes` —— REST 路由（``/api/videos``、``/api/tasks``……）与鉴权
- :mod:`api.main` —— FastAPI 应用装配、生命周期、日志、统一错误信封
- :mod:`api.mcp_server` —— 手写 JSON-RPC 2.0 的 MCP 服务（stdio / HTTP）

启动::

    python -m api.main             # HTTP 服务（默认 127.0.0.1:8000）
    python -m api.mcp_server       # MCP stdio（供 Obsidian / Claude Desktop）
"""

from __future__ import annotations

__all__ = [
    "AnalyzeRequest",
    "AskRequest",
    "AnswerOut",
    "AnalyzeOut",
    "McpError",
    "McpServer",
    "TOOLS",
    "create_app",
    "create_http_router",
    "create_router",
    "get_job",
    "require_auth",
    "run_stdio",
    "set_orchestrator",
    "shutdown_jobs",
]


def __getattr__(name: str) -> object:
    """惰性导入，避免 ``import api`` 就拉起 FastAPI。

    CLI / 脚本只想用 ``api.mcp_server`` 时不该被迫安装 FastAPI；反之亦然。
    """
    if name in {"create_app", "app"}:
        from api import main as _main  # noqa: PLC0415

        return getattr(_main, name)
    if name in {"McpError", "McpServer", "TOOLS", "create_http_router", "run_stdio"}:
        from api import mcp_server as _mcp  # noqa: PLC0415

        return getattr(_mcp, name)
    from api import routes as _routes  # noqa: PLC0415

    try:
        return getattr(_routes, name)
    except AttributeError as exc:
        raise AttributeError(f"module 'api' has no attribute {name!r}") from exc
