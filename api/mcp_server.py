"""知影 MCP 服务 —— 让 Obsidian / Claude Desktop 等客户端直接调用本地视频库。

协议实现
--------
手写 **JSON-RPC 2.0**，不依赖官方 ``mcp`` SDK。理由：

1. 需求文档把 MCP 列为"多端接入"的一项，但没要求 SDK；手写 300 行比引入一个
   会连带拉起 pydantic/anyio 版本的依赖更可控（项目已有 FastAPI，版本冲突风险真实存在）。
2. MCP 的 stdio 传输本质就是 **NDJSON**（一行一条 JSON-RPC 消息），HTTP 传输就是
   ``POST`` 一个 JSON-RPC 对象。两者共用同一个 :meth:`McpServer.handle`，逻辑只有一份。
3. 工具集是"只读检索 + 一个写操作（analyze）"，接口面小，手写 schema 成本低于适配成本。

对外暴露的工具（tools）
-----------------------
===================  ===========================================================
``list_videos``      列出视频库（含状态、时长、切片数）
``get_video``        单个视频详情：切片 + 关键帧 + 时间戳
``ask_video``        基于视频内容问答，回答带 ``[n]`` 引用与毫秒区间
``search_video``     只做混合检索，返回 Top-K 证据（不给模型编造的空间）
``summarize_video``  生成/重生成整片摘要
``analyze_video``    提交分析任务（异步，返回 task_id）
``get_task``         查询分析任务进度
``stats``            存储用量（对照需求文档表3容量上限）
===================  ===========================================================

用法::

    python -m api.mcp_server                    # stdio（供 Claude Desktop / Obsidian）
    python -m api.mcp_server --http             # 起 HTTP，POST /mcp 收 JSON-RPC
    python -m api.mcp_server --list-tools        # 打印工具清单后退出

Claude Desktop 配置示例（``claude_desktop_config.json``）::

    {
      "mcpServers": {
        "zhiying": {
          "command": "python",
          "args": ["-m", "api.mcp_server"],
          "cwd": "D:/VideoLens"
        }
      }
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Final

from core.config import AppConfig, get_app_config, load_app_config
from core.pipeline.orchestrator import AnalyzeOptions, Orchestrator, PipelineError
from core.storage.repository import NotFoundError, StorageError

#: FastAPI 的 ``Request`` 必须在模块级可见：本文件开了
#: ``from __future__ import annotations``，注解是字符串，FastAPI 会用
#: ``get_type_hints()`` 在**模块全局命名空间**里求值。若只在函数内 import，
#: 解析失败后 ``request: Request`` 会被当成查询参数，请求直接 422。
try:  # pragma: no cover - 纯 stdio 部署可以没有 FastAPI
    from fastapi import Request
except ImportError:  # pragma: no cover
    Request = None  # type: ignore[assignment,misc]

logger = logging.getLogger("zhiying.api.mcp")

__all__ = ["McpError", "McpServer", "TOOLS", "create_http_router", "run_stdio"]

#: JSON-RPC 错误码（协议规定）。
PARSE_ERROR: Final[int] = -32_700
INVALID_REQUEST: Final[int] = -32_600
METHOD_NOT_FOUND: Final[int] = -32_601
INVALID_PARAMS: Final[int] = -32_602
INTERNAL_ERROR: Final[int] = -32_603

#: MCP 协议版本（客户端会在 initialize 里协商，这里取当前稳定版）。
PROTOCOL_VERSION: Final[str] = "2024-11-05"


class McpError(Exception):
    """工具执行失败，携带 JSON-RPC 错误码。"""

    def __init__(self, message: str, *, code: int = INTERNAL_ERROR, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


# =============================================================================
# 工具定义
# =============================================================================


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


_VIDEO_ID = {"type": "string", "description": "视频 ID（形如 v1a2b3c4d5e6f）"}
_QUESTION = {"type": "string", "description": "自然语言问题（≤500 字符）"}

#: ``name -> {description, inputSchema}``。schema 只写必要约束，详细的
#: 上限校验在工具实现里做（错误信息更贴近用户语言）。
TOOLS: dict[str, dict[str, Any]] = {
    "list_videos": {
        "description": "列出本地视频库。返回 video_id / 标题 / 时长 / 状态 / 切片数。",
        "inputSchema": _schema(
            {
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "status": {
                    "type": "string",
                    "description": "按状态过滤：pending/downloading/transcribing/indexing/ready/failed",
                },
            }
        ),
    },
    "get_video": {
        "description": "取单个视频详情：元数据 + 切片（含毫秒时间区间）+ 关键帧列表。",
        "inputSchema": _schema({"video_id": _VIDEO_ID}, ["video_id"]),
    },
    "ask_video": {
        "description": (
            "基于某个视频的内容回答问题。"
            "回答中的 [n] 对应 refs，refs 里带 clip_id 与毫秒级 start_ms/end_ms，可直接跳转播放。"
        ),
        "inputSchema": _schema(
            {
                "video_id": _VIDEO_ID,
                "question": _QUESTION,
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            ["video_id", "question"],
        ),
    },
    "search_video": {
        "description": "只做三通道混合检索，返回 Top-K 证据片段（不经过大模型，适合让客户端自己组织回答）。",
        "inputSchema": _schema(
            {
                "video_id": _VIDEO_ID,
                "query": {"type": "string", "description": "检索词/句"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 6},
            },
            ["video_id", "query"],
        ),
    },
    "summarize_video": {
        "description": "生成（或重新生成）整片摘要，Markdown 格式，≤5000 字符。",
        "inputSchema": _schema(
            {"video_id": _VIDEO_ID, "max_items": {"type": "integer", "minimum": 1, "maximum": 50}},
            ["video_id"],
        ),
    },
    "analyze_video": {
        "description": (
            "提交一个视频分析任务（下载/转写/切片/抽帧/建向量）。"
            "默认异步：立即返回 task_id，用 get_task 轮询进度（HTTP 端 /api/tasks/{id} 通用）。"
            "wait=true 时同步等待完成，最长等 mcp.call_timeout_s 秒，超时返回错误"
            "（任务仍在后台继续，稍后用 get_task 取结果）。"
        ),
        "inputSchema": _schema(
            {
                "source": {"type": "string", "description": "本地路径或视频 URL"},
                "title": {"type": "string"},
                "language": {"type": "string", "description": "强制语言，如 zh / en"},
                "force": {"type": "boolean", "default": False},
                "wait": {
                    "type": "boolean",
                    "default": False,
                    "description": "同步等待完成（小视频用）；默认 false 走后台任务",
                },
            },
            ["source"],
        ),
    },
    "get_task": {
        "description": "查询分析任务状态与结果（HTTP 与 MCP 提交的任务共用同一编号空间）。",
        "inputSchema": _schema(
            {"task_id": {"type": "string", "description": "analyze_video 返回的 task_id"}},
            ["task_id"],
        ),
    },
    "stats": {
        "description": "存储用量统计（视频/切片/关键帧/向量数与容量占比）。",
        "inputSchema": _schema({}),
    },
}


def _hms(ms: int) -> str:
    total = max(0, int(ms))
    hours, rest = divmod(total, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _as_dict(obj: Any) -> dict[str, Any]:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    raise McpError(f"无法序列化 {type(obj).__name__}")


# =============================================================================
# 服务
# =============================================================================


class McpServer:
    """JSON-RPC 2.0 分发器。**线程安全**（每个请求独立，无共享可变状态）。"""

    def __init__(self, orchestrator: Orchestrator, cfg: AppConfig | None = None) -> None:
        self._orch = orchestrator
        self._cfg = cfg or get_app_config()
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "list_videos": self._tool_list_videos,
            "get_video": self._tool_get_video,
            "ask_video": self._tool_ask_video,
            "search_video": self._tool_search_video,
            "summarize_video": self._tool_summarize,
            "analyze_video": self._tool_analyze,
            "get_task": self._tool_get_task,
            "stats": self._tool_stats,
        }

    # -- JSON-RPC 入口 -----------------------------------------------------
    def handle(self, message: Any) -> dict[str, Any] | None:
        """处理一条 JSON-RPC 消息。通知（无 ``id``）返回 ``None``。"""
        if not isinstance(message, dict):
            return _error(None, INVALID_REQUEST, "消息必须是 JSON 对象")

        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}

        if not isinstance(method, str) or not method:
            return _error(msg_id, INVALID_REQUEST, "缺少 method 字段")
        if not isinstance(params, dict):
            return _error(msg_id, INVALID_PARAMS, "params 必须是对象")

        is_notification = "id" not in message

        try:
            result = self._dispatch(method, params)
        except McpError as exc:
            if is_notification:
                return None
            return _error(msg_id, exc.code, str(exc), exc.data)
        except Exception as exc:  # noqa: BLE001 - 协议层兜底
            logger.exception("处理 %s 时异常", method)
            if is_notification:
                return None
            return _error(
                msg_id,
                INTERNAL_ERROR,
                f"{type(exc).__name__}: {exc}",
                traceback.format_exc(limit=3) if logger.isEnabledFor(logging.DEBUG) else None,
            )

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "initialize":
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": self._cfg.mcp.server_name,
                    "version": self._cfg.mcp.server_version,
                },
                "instructions": (
                    "知影是本地视频分析助手。先用 list_videos 找到 video_id，"
                    "再用 ask_video 提问；回答里的 [n] 与 refs 中的毫秒区间对应，可直接跳转。"
                    "所有数据都在本机，不会外发视频内容。"
                ),
            }
        if method in {"notifications/initialized", "initialized"}:
            return None
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [{"name": name, **spec} for name, spec in TOOLS.items()]}
        if method == "tools/call":
            return self._call_tool(params)
        if method in {"resources/list", "prompts/list"}:
            # 本服务不提供 resources/prompts；返回空集合比报错更友好。
            return {method.split("/")[0]: []}
        if method == "shutdown":
            return {}
        raise McpError(f"未实现的方法: {method}", code=METHOD_NOT_FOUND)

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or name not in self._handlers:
            raise McpError(f"未知工具: {name}", code=INVALID_PARAMS)
        if not isinstance(arguments, dict):
            raise McpError("arguments 必须是对象", code=INVALID_PARAMS)

        payload = self._handlers[name](arguments)
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        return {
            "content": [{"type": "text", "text": text}],
            "isError": False,
            "structuredContent": payload,
        }

    # -- 工具实现 ----------------------------------------------------------
    def _require_video(self, video_id: Any) -> Any:
        if not isinstance(video_id, str) or not video_id.strip():
            raise McpError("video_id 不能为空", code=INVALID_PARAMS)
        try:
            record = self._orch.repository.find_video(video_id.strip())
        except StorageError as exc:
            raise McpError(f"读取视频失败: {exc}") from exc
        if record is None:
            raise McpError(f"视频不存在: {video_id}", code=INVALID_PARAMS)
        return record

    def _tool_list_videos(self, args: dict[str, Any]) -> dict[str, Any]:
        limit = _int_arg(args, "limit", default=20, low=1, high=200)
        offset = _int_arg(args, "offset", default=0, low=0, high=1_000_000)
        status = args.get("status") or None
        if status is not None and not isinstance(status, str):
            raise McpError("status 必须是字符串", code=INVALID_PARAMS)
        records = self._orch.repository.list_videos(
            limit=limit, offset=offset, status=status
        )
        items = []
        for record in records:
            raw = _as_dict(record)
            items.append(
                {
                    "video_id": raw.get("video_id", ""),
                    "title": raw.get("title") or raw.get("display_name", ""),
                    "duration_ms": int(raw.get("duration_ms", 0) or 0),
                    "duration": _hms(int(raw.get("duration_ms", 0) or 0)),
                    "status": raw.get("status", ""),
                    "language": raw.get("language", ""),
                    "created_at": raw.get("created_at", ""),
                }
            )
        return {
            "total": self._orch.repository.count_videos(),
            "count": len(items),
            "items": items,
        }

    def _tool_get_video(self, args: dict[str, Any]) -> dict[str, Any]:
        record = self._require_video(args.get("video_id"))
        video_id = record.video_id
        raw = _as_dict(record)
        clips = []
        for clip in self._orch.repository.get_clips(video_id):
            c = _as_dict(clip)
            start = int(c.get("start_ms", 0) or 0)
            end = int(c.get("end_ms", 0) or 0)
            ids = c.get("keyframe_ids", ()) or ()
            if isinstance(ids, str):
                ids = [p for p in ids.split(",") if p]
            clips.append(
                {
                    "clip_id": c.get("clip_id", ""),
                    "index": int(c.get("index_no", 0) or 0),
                    "start_ms": start,
                    "end_ms": end,
                    "time_range": f"{_hms(start)}–{_hms(end)}",
                    "text": c.get("core_text") or c.get("text", ""),
                    "keyframe_ids": list(ids),
                }
            )
        frames = []
        for frame in self._orch.repository.get_keyframes(video_id):
            k = _as_dict(frame)
            ts = int(k.get("timestamp_ms", 0) or 0)
            frames.append(
                {
                    "keyframe_id": k.get("keyframe_id", ""),
                    "clip_id": k.get("clip_id"),
                    "timestamp_ms": ts,
                    "timecode": _hms(ts),
                    "path": k.get("path", ""),
                    "description": k.get("description", ""),
                    "ocr_text": k.get("ocr_text", ""),
                }
            )
        return {
            "video": {
                "video_id": video_id,
                "title": raw.get("title") or raw.get("display_name", ""),
                "status": raw.get("status", ""),
                "duration_ms": int(raw.get("duration_ms", 0) or 0),
                "duration": _hms(int(raw.get("duration_ms", 0) or 0)),
                "language": raw.get("language", ""),
                "width": int(raw.get("width", 0) or 0),
                "height": int(raw.get("height", 0) or 0),
                "source_kind": raw.get("source_kind", ""),
            },
            "clip_count": len(clips),
            "keyframe_count": len(frames),
            "clips": clips,
            "keyframes": frames,
        }

    def _tool_ask_video(self, args: dict[str, Any]) -> dict[str, Any]:
        record = self._require_video(args.get("video_id"))
        question = args.get("question")
        if not isinstance(question, str) or not question.strip():
            raise McpError("question 不能为空", code=INVALID_PARAMS)
        if len(question) > self._cfg.limits.max_question_chars:
            raise McpError(
                f"问题超过 {self._cfg.limits.max_question_chars} 字符", code=INVALID_PARAMS
            )
        top_k = args.get("top_k")
        if top_k is not None:
            top_k = _int_arg(args, "top_k", default=0, low=1, high=50)

        try:
            result = self._orch.ask(record.video_id, question.strip(), top_k=top_k)
        except PipelineError as exc:
            raise McpError(f"问答失败: {exc}") from exc
        payload = result.to_dict()
        payload["ok"] = result.ok
        return payload

    def _tool_search_video(self, args: dict[str, Any]) -> dict[str, Any]:
        record = self._require_video(args.get("video_id"))
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise McpError("query 不能为空", code=INVALID_PARAMS)
        top_k = _int_arg(args, "top_k", default=self._cfg.retrieval.top_k, low=1, high=50)

        try:
            items = self._orch._retrieve(record.video_id, query.strip(), top_k)
        except Exception as exc:  # noqa: BLE001 - 检索失败给出可读错误
            raise McpError(f"检索失败: {exc}") from exc

        results = []
        for item in items:
            raw = _as_dict(item) if not isinstance(item, dict) else dict(item)
            start = int(raw.get("start_ms", 0) or 0)
            end = int(raw.get("end_ms", 0) or 0)
            results.append(
                {
                    "clip_id": raw.get("clip_id", ""),
                    "video_id": raw.get("video_id", record.video_id),
                    "start_ms": start,
                    "end_ms": end,
                    "time_range": raw.get("time_range") or f"{_hms(start)}–{_hms(end)}",
                    "text": raw.get("text", ""),
                    "score": float(raw.get("normalized_score", raw.get("score", 0.0)) or 0.0),
                    "channel_ranks": raw.get("channel_ranks", {}),
                    "keyframe_paths": list(raw.get("keyframe_paths", ()) or ()),
                }
            )
        return {"query": query, "count": len(results), "results": results}

    def _tool_summarize(self, args: dict[str, Any]) -> dict[str, Any]:
        record = self._require_video(args.get("video_id"))
        max_items = _int_arg(args, "max_items", default=12, low=1, high=50)
        try:
            text = self._orch.summarize(record.video_id, max_items=max_items)
        except PipelineError as exc:
            raise McpError(f"总结失败: {exc}") from exc
        limit = self._cfg.limits.max_report_chars
        truncated = len(text) > limit
        return {
            "video_id": record.video_id,
            "summary": text[:limit],
            "chars": len(text),
            "truncated": truncated,
        }

    def _tool_analyze(self, args: dict[str, Any]) -> dict[str, Any]:
        source = args.get("source")
        if not isinstance(source, str) or not source.strip():
            raise McpError("source 不能为空", code=INVALID_PARAMS)
        wait = bool(args.get("wait", False))
        options = AnalyzeOptions(
            title=str(args.get("title") or ""),
            language=args.get("language") or None,
            force=bool(args.get("force", False)),
        )

        # 与 HTTP 同一条路（api.jobs.JobStore）：task_id 只发一次，
        # SQLite 恢复记录、内存进度流、HTTP /tasks/{id} 三处共用。
        from api.jobs import TERMINAL_STATUS, get_job_store  # noqa: PLC0415

        try:
            job = get_job_store().submit(self._orch, source.strip(), options)
        except RuntimeError as exc:
            raise McpError(f"服务正在关闭: {exc}") from exc

        if not wait:
            snapshot = job.snapshot()
            return {
                "task_id": snapshot["task_id"],
                "status": snapshot["status"],
                "note": "任务已提交，用 get_task 查询进度；HTTP 端 /api/tasks/{task_id} 通用",
            }

        timeout_s = max(1.0, float(self._cfg.mcp.call_timeout_s))
        snapshot = get_job_store().wait(job, timeout_s=timeout_s)
        if snapshot.get("status") not in TERMINAL_STATUS:
            raise McpError(
                f"分析超过 {timeout_s:.0f}s 未完成（任务仍在后台运行，"
                f"task_id={job.task_id}，请稍后用 get_task 查询）"
            )
        report = snapshot.get("report") or {}
        if snapshot.get("status") == "failed":
            raise McpError(f"分析失败: {snapshot.get('error') or report.get('errors')}")
        limit = self._cfg.limits.max_report_chars
        if isinstance(report, dict) and report.get("summary"):
            report = dict(report)
            report["summary"] = str(report["summary"])[:limit]
        return report if isinstance(report, dict) else {"task_id": job.task_id, "status": "done"}

    def _tool_get_task(self, args: dict[str, Any]) -> dict[str, Any]:
        task_id = args.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise McpError("task_id 不能为空", code=INVALID_PARAMS)
        task_id = task_id.strip()
        # 内存优先（带完整进度与报告），进程重启后回落 SQLite 持久化行
        from api.jobs import get_job_store  # noqa: PLC0415

        snapshot = get_job_store().snapshot(task_id)
        if snapshot is not None:
            return snapshot
        try:
            record = self._orch.repository.get_task(task_id)
        except (StorageError, NotFoundError) as exc:
            raise McpError(f"读取任务失败: {exc}") from exc
        if record is None:
            raise McpError(f"任务不存在: {task_id}", code=INVALID_PARAMS)
        return _as_dict(record)

    def _tool_stats(self, args: dict[str, Any]) -> dict[str, Any]:
        del args
        raw = _as_dict(self._orch.repository.stats())
        # 容量上限以配置为准、仓储实测值为准覆盖：即使仓储是裸构造（未注入
        # limits，例如单测里直接 SqliteRepository(path)），对外也要报出需求
        # 文档表3 的容量口径，否则前端画不出用量条。
        cap = self._cfg.storage.limits
        limits: dict[str, int] = {
            "max_videos": int(cap.max_videos),
            "max_clips": int(cap.max_clips),
            "max_transcript_segments": int(cap.max_transcript_segments),
            "max_keyframes": int(cap.max_keyframes),
            "max_keyframe_bytes": int(cap.max_keyframe_bytes),
            "max_text_vectors": int(cap.max_text_vectors),
            "max_image_vectors": int(cap.max_image_vectors),
            "max_result_bytes": int(cap.max_result_bytes),
        }
        for key, value in (raw.get("limits") or {}).items():
            try:
                limits[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return {
            "counts": {
                "videos": int(raw.get("videos", 0)),
                "clips": int(raw.get("clips", 0)),
                "transcript_segments": int(raw.get("transcript_segments", 0)),
                "keyframes": int(raw.get("keyframes", 0)),
                "tasks_pending": int(raw.get("tasks_pending", 0)),
                "tasks_running": int(raw.get("tasks_running", 0)),
            },
            "bytes": {
                "keyframes": int(raw.get("keyframe_bytes", 0)),
                "database": int(raw.get("db_bytes", 0)),
            },
            "limits": limits,
            "usage": {k: round(float(v), 4) for k, v in (raw.get("usage") or {}).items()},
        }


# =============================================================================
# 小工具
# =============================================================================


def _error(
    msg_id: Any, code: int, message: str, data: Any = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        payload["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": payload}


def _int_arg(
    args: dict[str, Any], key: str, *, default: int, low: int, high: int
) -> int:
    value = args.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value.strip())
        else:
            raise McpError(f"{key} 必须是整数", code=INVALID_PARAMS)
    if not low <= value <= high:
        raise McpError(f"{key} 必须在 {low}..{high} 之间", code=INVALID_PARAMS)
    return value


def _parse_line(line: str) -> Any:
    """解析一行 NDJSON；宽容处理 Markdown 代码块包裹（有些客户端会加 ```）。"""
    text = line.strip()
    if not text:
        raise McpError("空消息", code=PARSE_ERROR)
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise McpError(f"JSON 解析失败: {exc}", code=PARSE_ERROR) from exc


# =============================================================================
# stdio 传输
# =============================================================================


def run_stdio(server: McpServer) -> int:
    """NDJSON 主循环：读一行 → 处理 → 写一行。

    注意：**stdout 只允许写协议消息**，任何日志都必须走 stderr，否则会破坏
    客户端的帧解析。``configure_logging`` 已把 console handler 指向 stderr。
    """
    stdin = sys.stdin
    stdout = sys.stdout
    try:
        stdin.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - 已被重定向时保持原样
        pass

    logger.info("MCP stdio 服务启动（tools=%d）", len(TOOLS))
    for line in stdin:
        if not line.strip():
            continue
        try:
            message = _parse_line(line)
        except McpError as exc:
            response: dict[str, Any] | None = _error(None, exc.code, str(exc))
        else:
            if isinstance(message, list):  # JSON-RPC 批请求
                batch = [r for r in (server.handle(m) for m in message) if r is not None]
                response = batch or None
            else:
                response = server.handle(message)

        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False, default=str) + "\n")
            stdout.flush()

        if isinstance(message, dict) and message.get("method") == "shutdown":
            break
    logger.info("MCP stdio 服务退出")
    return 0


# =============================================================================
# HTTP 传输（挂到 api.main 的 /mcp）
# =============================================================================


def create_http_router(orchestrator: Orchestrator | None = None) -> Any:
    """构造 ``POST /mcp`` 的 ``APIRouter``（JSON-RPC over HTTP）。"""
    from fastapi import APIRouter, Body, Depends, HTTPException  # noqa: PLC0415

    from api.routes import require_auth  # noqa: PLC0415

    router = APIRouter()

    def _server(request: Request) -> McpServer:
        orch = orchestrator or getattr(request.app.state, "orchestrator", None)
        if orch is None:
            raise HTTPException(status_code=503, detail="编排器未就绪")
        return McpServer(orch, getattr(request.app.state, "config", None) or get_app_config())

    @router.post("/mcp", dependencies=[Depends(require_auth)], include_in_schema=False)
    def mcp_endpoint(request: Request, payload: Any = Body(...)) -> Any:
        server = _server(request)
        if isinstance(payload, list):
            return [r for r in (server.handle(m) for m in payload) if r is not None]
        return server.handle(payload)

    @router.get("/mcp/tools", dependencies=[Depends(require_auth)])
    def mcp_tools() -> dict[str, Any]:
        return {"tools": [{"name": name, **spec} for name, spec in TOOLS.items()]}

    return router


# =============================================================================
# CLI
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="api.mcp_server", description="知影 MCP 服务")
    parser.add_argument("--http", action="store_true", help="以 HTTP 方式启动（POST /mcp）")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--list-tools", action="store_true", help="打印工具清单后退出")
    parser.add_argument(
        "--no-models", action="store_true", help="不加载模型（仅验证协议握手）"
    )
    args = parser.parse_args(argv)

    if args.list_tools:
        print(json.dumps(TOOLS, ensure_ascii=False, indent=2))
        return 0

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,  # 关键：stdout 留给协议
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_app_config()
    try:
        orch = Orchestrator.from_config(cfg, load_models=not args.no_models)
    except Exception as exc:  # noqa: BLE001
        logger.error("编排器装配失败: %s", exc)
        return 2

    if args.http:
        try:
            import uvicorn  # noqa: PLC0415
        except ImportError:
            logger.error("缺少依赖：pip install 'uvicorn[standard]'")
            return 2
        from api.main import create_app  # noqa: PLC0415

        application = create_app(cfg, orchestrator=orch)
        application.include_router(create_http_router(orch))
        uvicorn.run(
            application,
            host=args.host or cfg.api.host,
            port=args.port or cfg.api.port,
            log_config=None,
        )
        return 0

    try:
        return run_stdio(McpServer(orch, cfg))
    finally:
        orch.close()


if __name__ == "__main__":
    raise SystemExit(main())
