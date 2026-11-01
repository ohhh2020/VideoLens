"""仅供本机网页使用的偏好记忆编辑工作流。"""

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from backend.memory.store import MemoryError
from backend.web.dependencies import _orchestrator


class MemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    kind: Literal["interest","question","style"] = "interest"


class MemoryOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool
    auto_learn: StrictBool


class MemoryDeleteMany(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(min_length=1, max_length=200)


def create_router(cfg=None):
    router = APIRouter()

    def service(request):
        memory = getattr(_orchestrator(request),"_memory",None)
        if memory is None:
            raise HTTPException(503,"偏好记忆组件未就绪")
        return memory

    def edit(action):
        try:
            return action()
        except MemoryError as exc:
            raise HTTPException(422,str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404,"记忆不存在或已被删除") from exc
        except OSError as exc:
            raise HTTPException(500,"记忆保存失败，原数据已保留，请检查运行目录写入权限与磁盘空间") from exc

    @router.get("/memory")
    def listing(request: Request):
        return service(request).status()

    @router.post("/memory/items")
    def add(request: Request, payload: MemoryInput):
        memory = service(request)
        return edit(lambda: memory.store.put(payload.text,payload.kind))

    @router.post("/memory/items/delete-many")
    def delete_many(request: Request, payload: MemoryDeleteMany):
        memory = service(request)
        removed = edit(lambda: memory.store.delete_many(payload.ids))
        return {"deleted": True, "deleted_count": len(removed), "ids": removed}

    @router.put("/memory/items/{item_id}")
    def update(request: Request, item_id: str, payload: MemoryInput):
        memory = service(request)
        return edit(lambda: memory.store.put(payload.text,payload.kind,item_id=item_id))

    @router.delete("/memory/items/{item_id}")
    def delete(request: Request, item_id: str):
        memory = service(request)
        edit(lambda: memory.store.delete(item_id))
        return {"deleted":True}

    @router.put("/memory/options")
    def options(request: Request, payload: MemoryOptions):
        memory = service(request)
        memory.store.options(enabled=payload.enabled,auto_learn=payload.auto_learn)
        if payload.auto_learn:
            memory.notify()
        return memory.status()

    @router.post("/memory/refresh",status_code=202)
    def refresh(request: Request):
        memory = service(request)
        if not memory.store.snapshot()["auto_learn"]:
            raise HTTPException(409,"请先开启历史归纳，或直接手动添加记忆")
        memory.notify()
        return memory.status()

    return router
