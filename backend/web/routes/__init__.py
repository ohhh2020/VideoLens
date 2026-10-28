"""仅装配网页内部工作流，不提供对外 API 或 MCP。"""

from fastapi import APIRouter
from backend.web.routes import system, videos, tasks, qa, memory


def create_router(cfg=None):
    router = APIRouter()
    for module in (system, videos, tasks, qa, memory):
        router.include_router(module.create_router(cfg))
    return router
