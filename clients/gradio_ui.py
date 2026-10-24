"""知影 Web 端（Gradio）：上传/贴链接 → 分析 → 时间线 → 问答。

为什么用 Gradio 而不是自己写前端：本项目是"本地优先的单机工具"，Gradio 自带
文件上传、视频预览、画廊、表格与队列，几百行就能覆盖需求文档里"多端接入"的
Web 那条。要长得更像产品界面时，前端可以换成 React 走同一套 ``/api`` —— 这
一层只做"把管道可视化"，不含任何业务逻辑。

三个刻意为之的设计
------------------
1. **Gradio 延迟导入**：``import gradio`` 会拉起 ~1GB 的依赖树。CLI/服务端
   不需要它，所以本模块顶层不导入；缺了只影响这一个端。
2. **分析在后台线程跑**：``Orchestrator.analyze`` 是同步阻塞的（1 小时视频在
   CPU 上要几十分钟），直接放在回调里会把 UI 卡死。这里用
   ``threading.Thread`` + ``queue.Queue`` 把进度点转成 generator 的 yield，
   界面边跑边刷新。
3. **进度文案统一走 ``describe_stage``**：阶段名是英文枚举（parsing/asr/…），
   给人看要翻译成中文，且只在这一处翻译，CLI 与 Web 一致。

启动::

    python -m clients.gradio_ui            # 默认 127.0.0.1:7860
    python -m clients.gradio_ui --share    # 生成临时公网链接（谨慎）
"""

from __future__ import annotations

import argparse
import functools
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

__all__ = ["UiError", "build_ui", "main", "describe_stage", "load_orchestrator"]

#: 编排器重建互斥：_State 是进程级单例（本产品按"单机单用户"设计），
#: 两个保存并发触发会同时 close 掉对方正在用的编排器。
_REBUILD_LOCK = threading.Lock()


class UiError(RuntimeError):
    """Web 端不可用（缺 gradio、编排器装配失败等）。"""


#: 阶段英文名 → 给人看的中文。只在这里翻译一次，CLI 复用同一份。
_STAGE_LABELS: dict[str, str] = {
    "parsing": "解析来源",
    "downloading": "下载",
    "probing": "探测媒体",
    "extracting": "抽取音频",
    "transcribing": "语音转写",
    "chunking": "切片",
    "detecting": "场景检测",
    "keyframes": "抽取关键帧",
    "vision": "视觉理解",
    "embedding": "向量化",
    "indexing": "入库",
    "summarizing": "生成总结",
    "done": "完成",
    "failed": "失败",
}


def describe_stage(stage: str) -> str:
    """把阶段枚举翻成中文；未登记的原样返回（不吞信息）。"""
    key = (stage or "").strip().lower()
    return _STAGE_LABELS.get(key, stage or "运行中")


# =============================================================================
# 依赖与装配
# =============================================================================


def _grab_gradio() -> Any:
    try:
        import gradio  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise UiError(
            "缺少 gradio。安装后再试：pip install 'gradio>=4.44'"
        ) from exc
    return gradio


def load_orchestrator(*, config_path: str | None = None,
                      load_models: bool = False) -> Any:
    """装配编排器（失败抛 ``UiError`` —— Web 端没管道就没意义）。"""
    from core.config import load_app_config  # noqa: PLC0415
    from core.pipeline.orchestrator import Orchestrator  # noqa: PLC0415

    try:
        cfg = load_app_config(config_path)
    except Exception as exc:  # noqa: BLE001
        raise UiError(f"配置加载失败: {exc}") from exc
    try:
        return Orchestrator.from_config(cfg, load_models=load_models)
    except Exception as exc:  # noqa: BLE001
        raise UiError(f"编排器装配失败: {exc}") from exc


@dataclass(slots=True)
class _State:
    """一次 Web 会话共享的东西（Gradio 回调无状态，靠闭包传）。"""

    orch: Any
    cfg: Any
    frames_dir: Path

    def video_choices(self, limit: int = 50) -> list[str]:
        try:
            records = self.orch.repository.list_videos(limit=limit)
        except Exception:  # noqa: BLE001
            return []
        out: list[str] = []
        for record in records:
            title = (getattr(record, "title", "") or getattr(record, "display_name", ""))[:28]
            out.append(f"{record.video_id} · {title}" if title else str(record.video_id))
        return out


def _ms_to_hms(ms: int | float) -> str:
    total = max(0, int(ms or 0))
    hours, rest = divmod(total, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _human_bytes(size: int | float) -> str:
    value = float(size or 0)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024.0 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GiB"


def _parse_video_choice(text: str) -> str:
    """下拉框是 ``video_id · 标题``，这里只取前面的 id。"""
    return (text or "").split("·", 1)[0].strip()


# =============================================================================
# 分析（后台线程 + generator 进度）
# =============================================================================


def _run_analyze(state: _State, source: str, opts: dict[str, Any],
                 progress_queue: "queue.Queue[tuple[str, float, str]]",
                 box: dict[str, Any]) -> None:
    """在后台线程里跑一次分析，把进度点塞进队列、结果塞进 ``box``。"""
    from core.pipeline.orchestrator import AnalyzeOptions  # noqa: PLC0415

    def on_progress(stage: str, value: float, message: str) -> None:
        progress_queue.put((stage, value, message))

    try:
        options = AnalyzeOptions(
            language=opts.get("language") or None,
            skip_vision=bool(opts.get("skip_vision")),
            skip_summary=bool(opts.get("skip_summary")),
            skip_vectors=bool(opts.get("skip_vectors")),
            force=bool(opts.get("force")),
            title=str(opts.get("title") or ""),
            progress=on_progress,
        )
        box["report"] = state.orch.analyze(source, options)
    except Exception as exc:  # noqa: BLE001 - 线程里必须兜住，否则 UI 永远转圈
        box["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        progress_queue.put(("__end__", 1.0, ""))


def _render_report(report: Any) -> str:
    """把 ``AnalysisReport`` 渲染成 Markdown（结构化字段 + 阶段表）。"""
    if report is None:
        return ""
    status = str(getattr(report, "status", "") or "")
    lines = [
        f"### {'✅ 分析完成' if status == 'ready' else f'❌ 分析未完成（{status}）'}",
        "",
        f"- **video_id**: `{report.video_id}`",
        f"- **标题**: {report.title or '（未命名）'}",
        f"- **时长**: {_ms_to_hms(report.duration_ms)}",
        f"- **切片**: {report.clip_count} 个 ｜ **关键帧**: {report.keyframe_count} 张"
        f" ｜ **向量**: {report.vector_count} 条",
        f"- **转写来源**: {report.transcript_source or '-'}"
        f" ｜ **语言**: {report.language or '-'}",
    ]
    stages = list(getattr(report, "stages", []) or [])
    if stages:
        lines += ["", "| 阶段 | 状态 | 耗时 | 说明 |", "| --- | --- | --- | --- |"]
        for stage in stages:
            if getattr(stage, "skipped", False):
                flag = "跳过"
            elif getattr(stage, "degraded", False):
                flag = "降级"
            elif getattr(stage, "ok", False):
                flag = "ok"
            else:
                flag = "失败"
            note = str(getattr(stage, "message", "") or getattr(stage, "error", "") or "")
            lines.append(
                f"| {describe_stage(getattr(stage, 'stage', ''))} | {flag} "
                f"| {getattr(stage, 'elapsed_ms', 0)}ms | {note[:60]} |"
            )
    degraded = list(getattr(report, "degraded", []) or [])
    if degraded:
        lines += ["", "**降级项**", ""]
        lines += [f"- {note}" for note in degraded]
    summary = str(getattr(report, "summary", "") or "").strip()
    if summary:
        lines += ["", "---", "", "### 内容总结", "", summary]
    return "\n".join(lines)


def _analyze_stream(state: _State, source: str, opts: dict[str, Any]
                    ) -> Iterator[tuple[str, str]]:
    """generator：边跑边 yield ``(进度条, 进度文案, 报告)``。"""
    source = (source or "").strip().strip('"').strip("'")
    if not source:
        yield "### ⚠️ 请填写本地路径或视频链接", "等待输入"
        return

    progress_queue: "queue.Queue[tuple[str, float, str]]" = queue.Queue(maxsize=512)
    box: dict[str, Any] = {}
    worker = threading.Thread(
        target=_run_analyze,
        args=(state, source, opts, progress_queue, box),
        name="zhiying-analyze",
        daemon=True,
    )
    worker.start()

    started = time.monotonic()
    last_stage = "parsing"
    last_message = "已提交"
    last_value = 0.0
    yield "### ⏳ 正在分析 …", f"0% {last_message}"

    while True:
        try:
            stage, value, message = progress_queue.get(timeout=0.4)
        except queue.Empty:
            if not worker.is_alive():
                break
            elapsed = time.monotonic() - started
            yield (
                f"### ⏳ {describe_stage(last_stage)} …",
                f"{int(last_value * 100)}% · {last_message} · 已用 {elapsed:.0f}s",
            )
            continue
        if stage == "__end__":
            break
        last_stage, last_value, last_message = stage, float(value or 0.0), message or ""
        yield (
            f"### ⏳ {describe_stage(stage)} …",
            f"{int(last_value * 100)}% · {last_message}",
        )

    worker.join(timeout=5.0)
    if box.get("error"):
        yield f"### ❌ 分析失败\n\n```\n{box['error']}\n```", "失败"
        return
    report = box.get("report")
    if report is None:
        yield "### ❌ 分析结束但没有结果", "失败"
        return
    yield _render_report(report), f"100% · 完成（{time.monotonic() - started:.1f}s）"


def _analyze_click(
    state: _State,
    src: str,
    lang: Any,
    ttl: Any,
    skip_vision: Any,
    skip_summary: Any,
    skip_vectors: Any,
    force: Any,
) -> Iterator[tuple[str, str]]:
    """「开始分析」按钮的处理函数。

    必须是真正的 generator function（``yield from``），**不能**用 ``lambda``
    包一层：Gradio 靠 ``inspect.isgeneratorfunction`` 判断要不要走流式，
    ``lambda`` 只是「返回了一个 generator 对象」，会被当成一个普通返回值，
    于是报 ``A function didn't return enough output values
    (needed: 2, returned: 1)``。
    """
    yield from _analyze_stream(
        state,
        src,
        {
            "language": None if lang in {"自动", "", None} else lang,
            "title": ttl,
            "skip_vision": bool(skip_vision),
            "skip_summary": bool(skip_summary),
            "skip_vectors": bool(skip_vectors),
            "force": bool(force),
        },
    )


# =============================================================================
# 读取：库列表 / 详情 / 问答
# =============================================================================


def _refresh_videos(state: _State, limit: int = 50) -> list[list[Any]]:
    try:
        records = state.orch.repository.list_videos(limit=int(limit))
    except Exception:  # noqa: BLE001
        return []
    rows: list[list[Any]] = []
    for record in records:
        rows.append([
            str(record.video_id),
            str(getattr(record, "status", "") or ""),
            _ms_to_hms(getattr(record, "duration_ms", 0)),
            int(getattr(record, "clip_count", 0) or 0),
            str(getattr(record, "title", "") or getattr(record, "display_name", "") or "")[:40],
            str(getattr(record, "created_at", "") or "")[:19],
        ])
    return rows


def _video_detail(state: _State, video_id: str) -> tuple[str, list[list[Any]], list[tuple[str, str]]]:
    """返回 (Markdown 概览, 切片表格, 关键帧画廊数据)。"""
    video_id = _parse_video_choice(video_id)
    if not video_id:
        return "### 请选择一个视频", [], []
    repo = state.orch.repository
    try:
        record = repo.find_video(video_id)
    except Exception as exc:  # noqa: BLE001
        return f"### 读取失败\n\n```\n{exc}\n```", [], []
    if record is None:
        return f"### 找不到视频 `{video_id}`", [], []

    clips = []
    try:
        clips = list(repo.get_clips(video_id))
    except Exception:  # noqa: BLE001
        clips = []
    frames = []
    try:
        frames = list(repo.get_keyframes(video_id))
    except Exception:  # noqa: BLE001
        frames = []

    head = [
        f"### {getattr(record, 'title', '') or getattr(record, 'display_name', '') or video_id}",
        "",
        f"- **video_id**: `{video_id}`",
        f"- **状态**: {getattr(record, 'status', '')}"
        f"（{getattr(record, 'stage', '') or '-'}）",
        f"- **时长**: {_ms_to_hms(getattr(record, 'duration_ms', 0))}",
        f"- **分辨率**: {getattr(record, 'width', 0)}x{getattr(record, 'height', 0)}"
        f" @ {getattr(record, 'fps', 0)}fps",
        f"- **切片 / 关键帧**: {len(clips)} / {len(frames)}",
    ]
    source_url = str(getattr(record, "source_url", "") or "")
    if source_url:
        head.append(f"- **来源**: {source_url}")

    rows = [
        [
            str(clip.clip_id),
            f"{_ms_to_hms(clip.start_ms)} – {_ms_to_hms(clip.end_ms)}",
            len(str(getattr(clip, "core_text", "") or getattr(clip, "text", "") or "")),
            ", ".join(clip.keyframe_ids or ()),
            str(getattr(clip, "core_text", "") or getattr(clip, "text", "") or "")[:60],
        ]
        for clip in clips
    ]
    gallery: list[tuple[str, str]] = []
    for frame in frames:
        path = Path(str(getattr(frame, "path", "") or ""))
        if path.is_file():
            caption = f"{_ms_to_hms(getattr(frame, 'timestamp_ms', 0))} · {frame.keyframe_id}"
            gallery.append((str(path), caption))
    return "\n".join(head), rows, gallery


def _ask(state: _State, video_id: str, question: str, top_k: int
         ) -> tuple[str, list[list[Any]], list[tuple[str, str]]]:
    video_id = _parse_video_choice(video_id)
    question = (question or "").strip()
    if not video_id or not question:
        return "### 请选择视频并填写问题", [], []

    result = state.orch.ask(video_id, question, top_k=int(top_k) if top_k else None)
    answer = str(getattr(result, "answer", "") or "").strip()
    if not answer:
        notes = list(getattr(result, "notes", []) or [])
        body = "### 没有得到回答\n\n" + ("\n".join(f"- {n}" for n in notes) or "（无更多信息）")
        return body, [], []

    lines = [f"### 回答", "", answer, ""]
    meta = [
        f"后端 {getattr(result, 'backend', '') or '-'}",
        f"{int(getattr(result, 'elapsed_ms', 0) or 0)}ms",
        f"上下文 {int(getattr(result, 'context_chars', 0) or 0)} 字",
    ]
    if getattr(result, "degraded", False):
        meta.append("降级")
    lines.append("> " + " · ".join(meta))

    rows: list[list[Any]] = []
    gallery: list[tuple[str, str]] = []
    for ref in list(getattr(result, "refs", []) or []):
        rows.append([
            int(getattr(ref, "index", 0) or 0),
            str(getattr(ref, "clip_id", "") or ""),
            str(getattr(ref, "time_range", "")
                or f"{_ms_to_hms(getattr(ref, 'start_ms', 0))} – "
                   f"{_ms_to_hms(getattr(ref, 'end_ms', 0))}"),
            str(getattr(ref, "text", "") or "")[:80],
        ])
        for path in list(getattr(ref, "keyframe_paths", ()) or ()):
            candidate = Path(str(path))
            if candidate.is_file():
                gallery.append((str(candidate), f"[{int(getattr(ref, 'index', 0) or 0)}] "
                                              f"{candidate.name}"))
    return "\n".join(lines).strip(), rows, gallery


def _stats_markdown(state: _State) -> str:
    try:
        stats = state.orch.repository.stats()
    except Exception as exc:  # noqa: BLE001
        return f"### 统计读取失败\n\n```\n{exc}\n```"
    limits = dict(getattr(stats, "limits", {}) or {})
    lines = [
        "### 存储用量",
        "",
        "| 表 | 当前 | 上限 | 占用 |",
        "| --- | --- | --- | --- |",
    ]
    for key, label in (
        ("videos", "视频"),
        ("clips", "切片"),
        ("transcript_segments", "转录片段"),
        ("keyframes", "关键帧"),
    ):
        current = int(getattr(stats, key, 0) or 0)
        limit = int(limits.get(f"max_{key}", 0) or 0)
        ratio = f"{current / limit * 100:.1f}%" if limit else "-"
        lines.append(f"| {label} | {current} | {limit or '-'} | {ratio} |")
    lines += [
        "",
        f"- 关键帧占用 **{_human_bytes(getattr(stats, 'keyframe_bytes', 0))}**",
        f"- 数据库大小 **{_human_bytes(getattr(stats, 'db_bytes', 0))}**",
        f"- 任务 排队 {int(getattr(stats, 'tasks_pending', 0) or 0)} / "
        f"运行 {int(getattr(stats, 'tasks_running', 0) or 0)}",
    ]
    return "\n".join(lines)


# =============================================================================
# 设置（API 密钥 / 模型切换 / 本地模型开关）
# =============================================================================

#: 界面上密钥框显示的是打码值（``sk-a******5678``）。用户没动它时不能把
#: 打码串当成新密钥写回去 —— 那会把真密钥覆盖成一串星号，而且是静默的。
_MASK_MARK: str = "******"


def _looks_masked(value: str) -> bool:
    """判断密钥输入框里是不是"原样没动"的打码值。"""
    return _MASK_MARK in (value or "")


def _settings_values(state: _State) -> list[Any]:
    """当前配置摊平成组件初值，顺序必须与 :func:`_save_settings` 的入参一致。"""
    from core.user_settings import current_settings  # noqa: PLC0415

    cfg = state.cfg
    saved = current_settings(cfg.root_dir)
    llm = cfg.llm
    vl = cfg.vl
    key_shown = saved.get("llm.cloud.api_key", "")
    vl_key_shown = saved.get("vl.cloud_api_key", "")
    return [
        llm.mode,
        llm.cloud.base_url,
        llm.cloud.model,
        key_shown if key_shown else "",
        bool(vl.cloud_enabled),
        vl.cloud_base_url,
        vl.cloud_model,
        vl_key_shown if vl_key_shown else "",
        int(vl.cloud_sequence_frames),
        cfg.models.device,
        False,  # 「加载本地模型」默认不勾：勾了会等几分钟
    ]


def _save_settings(
    state: _State,
    llm_mode: str,
    llm_base: str,
    llm_model: str,
    llm_key: str,
    vl_enabled: bool,
    vl_base: str,
    vl_model: str,
    vl_key: str,
    vl_frames: float,
    device: str,
    load_local: bool,
) -> Iterator[str]:
    """保存设置 → 重建编排器 → 产出状态报告（generator，边重建边汇报）。

    重建而不是"热改属性"：配置对象是 ``frozen`` dataclass，后端实例在装配时
    就已按配置构造完毕；改一半再跑会出现"网关还是旧的、视觉已是新的"这种状态，
    比多等两秒糟糕得多。勾了「加载本地模型」时重建要几分钟 —— 流式产出
    至少让界面先把"正在重建"亮出来，而不是静默卡死。
    """
    if not _REBUILD_LOCK.acquire(blocking=False):
        yield (
            "### ⚠️ 上一次保存还在进行中\n\n"
            "编排器正在重建，请等它完成再改设置（两个重建并发会互相踩内存）。"
        )
        return
    try:
        yield _save_settings_locked(
            state, llm_mode, llm_base, llm_model, llm_key, vl_enabled, vl_base,
            vl_model, vl_key, vl_frames, device, load_local,
        )
    finally:
        _REBUILD_LOCK.release()


def _save_settings_locked(
    state: _State,
    llm_mode: str,
    llm_base: str,
    llm_model: str,
    llm_key: str,
    vl_enabled: bool,
    vl_base: str,
    vl_model: str,
    vl_key: str,
    vl_frames: float,
    device: str,
    load_local: bool,
) -> str:
    from core.config import load_app_config  # noqa: PLC0415
    from core.pipeline.orchestrator import Orchestrator  # noqa: PLC0415
    from core.user_settings import (  # noqa: PLC0415
        SettingsError,
        format_backends,
        save_settings,
    )

    payload: dict[str, Any] = {
        "llm.mode": (llm_mode or "auto").strip(),
        "llm.cloud.base_url": (llm_base or "").strip(),
        "llm.cloud.model": (llm_model or "").strip(),
        "vl.cloud_enabled": bool(vl_enabled),
        "vl.cloud_base_url": (vl_base or "").strip(),
        "vl.cloud_model": (vl_model or "").strip(),
        "vl.cloud_sequence_frames": int(vl_frames or 8),
        "models.device": (device or "auto").strip(),
    }
    # 密钥：只有用户真的改了才写。打码值 / 空串一律跳过（空串在模块里表示删除，
    # 但这里"没填"和"想删掉"分不清，宁可不动）。
    if not _looks_masked(llm_key):
        payload["llm.cloud.api_key"] = (llm_key or "").strip()
    if not _looks_masked(vl_key):
        payload["vl.cloud_api_key"] = (vl_key or "").strip()

    try:
        save_settings(payload, root=state.cfg.root_dir)
    except SettingsError as exc:
        return f"### ❌ 设置未保存\n\n```\n{exc}\n```"
    except OSError as exc:
        return f"### ❌ 写入设置文件失败\n\n```\n{exc}\n```"

    try:
        cfg = load_app_config()
    except Exception as exc:  # noqa: BLE001
        return f"### ⚠️ 已保存，但重新加载配置失败\n\n```\n{exc}\n```"

    note = ""
    previous = state.orch
    try:
        fresh = Orchestrator.from_config(cfg, load_models=bool(load_local))
        state.orch = fresh
        state.cfg = cfg
        state.frames_dir = Path(cfg.path("frames"))
        # 旧编排器带着模型句柄和数据库连接，不关就是每次保存泄一份
        if previous is not None and previous is not fresh:
            close = getattr(previous, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - 关旧的失败不该影响新实例
                    pass
    except Exception as exc:  # noqa: BLE001
        note = (
            f"\n\n### ⚠️ 设置已保存，但编排器重建失败\n\n```\n{exc}\n```"
            f"\n\n旧编排器仍在内存里，重启服务后新设置才会完全生效。"
        )

    head = "### ✅ 设置已保存" + ("（已加载本地模型）" if load_local else "")
    body = format_backends(cfg)
    extra = (
        "\n\n> 本地视觉/大模型的 GGUF 权重需要先下载；"
        "点「下载本地模型」或运行 `python scripts/download_models.py --all`。"
    )
    return f"{head}\n\n{body}{extra}{note}"


def _test_llm(state: _State, base_url: str, model: str, api_key: str,
              timeout_s: float = 30.0) -> str:
    """发一个最小请求验证云端大模型连通性（不改任何配置）。"""
    from core.models.llm import LlmError, LlmMessage, OpenAICompatibleBackend  # noqa: PLC0415

    base = (base_url or "").strip()
    name = (model or "").strip()
    if not base or not name:
        return "### ⚠️ 还差参数\n\n先填 **base_url** 和 **model** 再测试。"
    if _looks_masked(api_key):
        key = state.cfg.llm.cloud.resolved_key()
    else:
        key = (api_key or "").strip()

    backend = OpenAICompatibleBackend(
        base, name, key, timeout_s=float(timeout_s), max_retries=0
    )
    try:
        reply = backend.complete(
            [LlmMessage(role="user", content="只回复两个字：可用")],
            max_tokens=16,
            temperature=0.0,
        )
    except LlmError as exc:
        return f"### ❌ 云端大模型不可用\n\n```\n{exc}\n```"
    except Exception as exc:  # noqa: BLE001
        return f"### ❌ 云端大模型不可用\n\n```\n{type(exc).__name__}: {exc}\n```"

    return (
        "### ✅ 云端大模型可用\n\n"
        f"- 端点 `{backend.endpoint}`\n"
        f"- 模型 `{reply.model}`\n"
        f"- 回复 {reply.text[:40]!r}（{reply.elapsed_ms}ms，"
        f"prompt {reply.prompt_tokens} / completion {reply.completion_tokens} token）"
    )


def _test_vl(base_url: str, model: str, api_key: str,
             timeout_s: float = 30.0) -> str:
    """验证云端视觉端点（只打 ``/models`` 之类的轻量探测，不烧图 token）。"""
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    base = (base_url or "").strip().rstrip("/")
    name = (model or "").strip()
    if not base:
        return "### ⚠️ 还差参数\n\n先填 **base_url** 再测试。"
    if _looks_masked(api_key):
        return (
            "### ⚠️ 请重填密钥\n\n"
            "密钥框里是打码值，没法用它发请求。清空重填后『保存』再测。"
        )

    url = f"{base}/models"
    headers = {"Accept": "application/json"}
    if api_key:
        # 空密钥不发 Authorization 头：本地 vLLM/Ollama 常不校验，
        # 发一个空 Bearer 反而会被部分网关判 401。
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:
            status = getattr(response, "status", 200)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
        except Exception:  # noqa: BLE001
            pass
        return f"### ❌ 云端视觉端点返回 {exc.code}\n\n```\n{detail or exc}\n```"
    except Exception as exc:  # noqa: BLE001
        return f"### ❌ 云端视觉连接失败\n\n```\n{type(exc).__name__}: {exc}\n```"

    return (
        f"### ✅ 端点可达（HTTP {status}）\n\n"
        f"- `{url}`\n"
        f"- 模型 `{name or '（未填）'}`\n\n"
        "> 这里只验证连通与鉴权；真正的视觉请求会在分析关键帧时发生。"
    )


def _download_models(state: _State, scope: str) -> Iterator[str]:
    """下载本地模型权重（generator：后台线程下载，界面逐条刷新进度）。

    权重动辄几个 GB，同步回调会让按钮点下去界面静默卡到天荒地老；
    下载跑在守护线程里，这里只负责把 ``start/ok/fail`` 事件翻译成 Markdown。
    """
    from core.models.manager import ModelManager, ModelManagerError  # noqa: PLC0415

    groups: dict[str, tuple[str, ...]] = {
        "core": ("faster-whisper-large-v3-turbo", "bge-m3", "siglip2-base"),
        "qwen": ("qwen2-vl-2b-gguf", "qwen2.5-7b-gguf"),
        "all": (),
    }
    names = groups.get(scope, ())
    try:
        manager = ModelManager(
            config_path=str(getattr(state.cfg, "source_file", "") or "") or None,
            show_progress=False,
        )
    except Exception as exc:  # noqa: BLE001
        yield f"### ❌ 模型管理器装配失败\n\n```\n{exc}\n```"
        return

    try:
        targets = list(names) if names else manager.list_models()
    except Exception:  # noqa: BLE001
        targets = list(names)

    events: "queue.Queue[tuple[str, str]]" = queue.Queue()

    def work() -> None:
        for name in targets:
            events.put(("start", name))
            try:
                events.put(("ok", f"{name} → {manager.ensure_model(name)}"))
            except ModelManagerError as exc:
                events.put(("fail", f"{name}：{exc}"))
            except Exception as exc:  # noqa: BLE001
                events.put(("fail", f"{name}：{type(exc).__name__}: {exc}"))
        events.put(("end", ""))

    worker = threading.Thread(target=work, name="zhiying-download", daemon=True)
    worker.start()

    lines = [f"### 开始下载 {len(targets)} 个模型", ""]
    current = "准备中 …"
    while True:
        try:
            kind, detail = events.get(timeout=2.0)
        except queue.Empty:
            if not worker.is_alive():
                break
            yield "\n".join(lines) + f"\n\n⏳ {current}（仍在下载，请勿关闭页面）"
            continue
        if kind == "end":
            break
        if kind == "start":
            current = f"正在下载 {detail}"
            yield "\n".join(lines) + f"\n\n⏳ {current} …"
        elif kind == "ok":
            lines.append(f"- ✅ `{detail}`")
        else:
            lines.append(f"- ❌ `{detail}`")
    worker.join(timeout=5.0)
    lines += ["", "> 下载会走 ModelScope，失败自动回退 HuggingFace 镜像。"]
    yield "\n".join(lines)


def _format_backends_now(state: _State) -> str:
    """渲染"当前到底在用哪个后端"，供设置页与状态页共用。"""
    from core.user_settings import format_backends  # noqa: PLC0415

    try:
        return format_backends(state.cfg)
    except Exception as exc:  # noqa: BLE001
        return f"### 后端状态读取失败\n\n```\n{exc}\n```"


# =============================================================================
# 界面
# =============================================================================


def build_ui(*, config_path: str | None = None, orchestrator: Any | None = None,
             load_models: bool = False) -> Any:
    """构造 ``gr.Blocks``。``orchestrator`` 传入时不再装配（便于测试/复用）。"""
    gr = _grab_gradio()

    orch = orchestrator or load_orchestrator(config_path=config_path, load_models=load_models)
    cfg = getattr(orch, "config", None)
    if cfg is None:
        from core.config import get_app_config  # noqa: PLC0415

        cfg = get_app_config()
    state = _State(orch=orch, cfg=cfg, frames_dir=Path(cfg.path("frames")))

    # Gradio 6 起 theme 从 Blocks() 移到了 launch()，这里只留 title
    with gr.Blocks(title="知影 · 视频智能分析") as demo:
        gr.Markdown(
            "# 知影 · 视频智能分析\n"
            "本地优先：视频、转录、关键帧与向量都留在本机。"
            "每条回答都带 `video_id` / `clip_id` / 毫秒时间戳。"
        )

        # -- 分析 ----------------------------------------------------------
        with gr.Tab("① 分析"):
            with gr.Row():
                with gr.Column(scale=3):
                    source = gr.Textbox(
                        label="视频来源",
                        placeholder="本地路径（如 D:\\videos\\a.mp4）或视频链接",
                        lines=2,
                    )
                    with gr.Row():
                        language = gr.Dropdown(
                            choices=["自动", "zh", "en", "ja"],
                            value="自动", label="语言",
                        )
                        title = gr.Textbox(label="标题（可选）", placeholder="留空自动生成")
                    with gr.Row():
                        skip_vision = gr.Checkbox(label="跳过视觉理解", value=False)
                        skip_summary = gr.Checkbox(label="跳过总结", value=False)
                        skip_vectors = gr.Checkbox(label="跳过向量入库", value=False)
                        force = gr.Checkbox(label="强制重跑", value=False)
                    run_btn = gr.Button("开始分析", variant="primary")
                with gr.Column(scale=2):
                    progress_text = gr.Textbox(label="进度", value="等待开始", interactive=False)
                    report_md = gr.Markdown()

            run_btn.click(
                fn=functools.partial(_analyze_click, state),
                inputs=[source, language, title, skip_vision, skip_summary, skip_vectors, force],
                outputs=[report_md, progress_text],
                show_progress="hidden",
                api_name="analyze",
            )

        # -- 视频库 --------------------------------------------------------
        with gr.Tab("② 视频库"):
            with gr.Row():
                refresh_btn = gr.Button("刷新", size="sm")
                limit_box = gr.Number(value=50, label="条数", precision=0, scale=1)
            video_table = gr.Dataframe(
                headers=["video_id", "状态", "时长", "切片", "标题", "创建时间"],
                datatype=["str", "str", "str", "number", "str", "str"],
                interactive=False,
                wrap=True,
            )
            refresh_btn.click(fn=lambda n: _refresh_videos(state, int(n or 50)),
                              inputs=[limit_box], outputs=[video_table],
                              api_name="list_videos")
            demo.load(fn=lambda: _refresh_videos(state, 50), outputs=[video_table],
                      api_name="load_videos")

        # -- 详情 ----------------------------------------------------------
        with gr.Tab("③ 时间线"):
            with gr.Row():
                video_pick = gr.Dropdown(choices=state.video_choices(), label="视频", scale=3)
                detail_btn = gr.Button("加载", size="sm", scale=1)
                reload_pick_btn = gr.Button("刷新列表", size="sm", scale=1)
            detail_md = gr.Markdown()
            clip_table = gr.Dataframe(
                headers=["clip_id", "时间区间", "字数", "关键帧", "正文"],
                datatype=["str", "str", "number", "str", "str"],
                interactive=False,
                wrap=True,
            )
            frame_gallery = gr.Gallery(label="关键帧", columns=4, height="auto",
                                      object_fit="contain")
            detail_btn.click(fn=lambda v: _video_detail(state, v),
                             inputs=[video_pick],
                             outputs=[detail_md, clip_table, frame_gallery],
                             api_name="video_detail")
            reload_pick_btn.click(fn=lambda: gr.update(choices=state.video_choices()),
                                  outputs=[video_pick], api_name="video_choices")

        # -- 问答 ----------------------------------------------------------
        with gr.Tab("④ 问答"):
            with gr.Row():
                q_video = gr.Dropdown(choices=state.video_choices(), label="视频", scale=2)
                q_topk = gr.Slider(minimum=1, maximum=20, value=6, step=1, label="检索条数",
                                   scale=1)
            question = gr.Textbox(label="问题", placeholder="例如：切片长度是怎么定的？",
                                   lines=2)
            ask_btn = gr.Button("提问", variant="primary")
            answer_md = gr.Markdown()
            ref_table = gr.Dataframe(
                headers=["#", "clip_id", "时间区间", "原文"],
                datatype=["number", "str", "str", "str"],
                interactive=False,
                wrap=True,
            )
            ref_gallery = gr.Gallery(label="引用关键帧", columns=4, height="auto",
                                    object_fit="contain")
            ask_btn.click(fn=lambda v, q, k: _ask(state, v, q, int(k or 6)),
                          inputs=[q_video, question, q_topk],
                          outputs=[answer_md, ref_table, ref_gallery],
                          api_name="ask")

        # -- 设置 ----------------------------------------------------------
        with gr.Tab("⑤ 状态"):
            stat_btn = gr.Button("刷新用量", size="sm")
            stats_md = gr.Markdown(_stats_markdown(state))
            stat_btn.click(fn=lambda: _stats_markdown(state), outputs=[stats_md],
                          api_name="stats")
            cfg_text = gr.Textbox(
                label="配置摘要",
                value=(
                    f"项目根 {cfg.root_dir}\n"
                    f"数据目录 {cfg.paths.data}\n"
                    f"模型目录 {cfg.paths.models}\n"
                    f"设备 {cfg.models.device}（CPU 降级 "
                    f"{'开' if cfg.models.allow_cpu_fallback else '关'}）\n"
                    f"切片 {cfg.chunk.min_ms}–{cfg.chunk.max_ms}ms，目标 {cfg.chunk.target_ms}ms，"
                    f"重叠 {cfg.chunk.overlap_ms}ms\n"
                    f"检索 top_k={cfg.retrieval.top_k}，RRF k={cfg.retrieval.rrf_k}，"
                    f"权重 {dict(cfg.retrieval.weights)}"
                ),
                lines=7,
                interactive=False,
            )

        # -- 设置 ----------------------------------------------------------
        with gr.Tab("⑥ 设置"):
            gr.Markdown(
                "改完点「保存并应用」。设置写在 `data/user_settings.json`，"
                "优先于 `configs/app_config.yaml`，但会被环境变量 `ZH_YING__*` 顶掉。"
            )
            with gr.Accordion("大模型（问答与总结）", open=True):
                llm_mode = gr.Radio(
                    choices=["auto", "cloud", "local"],
                    value="auto",
                    label="模式",
                    info="auto = 有云端密钥就优先云端，失败自动回退本地；local = 只用本地 GGUF",
                )
                with gr.Row():
                    llm_base = gr.Textbox(
                        label="API 地址（base_url）",
                        placeholder="https://api.deepseek.com/v1",
                        scale=3,
                    )
                    llm_model = gr.Textbox(
                        label="模型名", placeholder="deepseek-chat", scale=2
                    )
                with gr.Row():
                    llm_key = gr.Textbox(
                        label="API 密钥（留空则读环境变量）",
                        type="password",
                        scale=3,
                    )
                    llm_test_btn = gr.Button("测试连通", size="sm", scale=1)
                llm_test_md = gr.Markdown()

            with gr.Accordion("视觉理解（关键帧描述）", open=True):
                vl_enabled = gr.Checkbox(
                    label="启用云端视觉", value=False,
                    info="关 = 用本地 Qwen2-VL GGUF（CPU 上约 11.7 秒/帧）",
                )
                with gr.Row():
                    vl_base = gr.Textbox(
                        label="API 地址（base_url）",
                        placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        scale=3,
                    )
                    vl_model = gr.Textbox(
                        label="模型名", placeholder="qwen-vl-plus", scale=2
                    )
                with gr.Row():
                    vl_key = gr.Textbox(
                        label="API 密钥（留空则读环境变量）",
                        type="password",
                        scale=3,
                    )
                    vl_test_btn = gr.Button("测试端点", size="sm", scale=1)
                vl_frames = gr.Slider(
                    minimum=1, maximum=16, value=8, step=1,
                    label="每轮最多几帧",
                    info="云端可以放宽到 8；本地 Qwen2-VL-2B 超过 2 帧会开始编造画面内容",
                )
                vl_test_md = gr.Markdown()

            with gr.Accordion("本地模型", open=False):
                device = gr.Dropdown(
                    choices=["auto", "cpu", "cuda"], value="auto", label="推理设备"
                )
                load_local = gr.Checkbox(
                    label="保存时加载本地模型权重", value=False,
                    info="勾上会等几分钟（7B 在 CPU 上加载很慢）；不勾则按需加载",
                )
                with gr.Row():
                    dl_core_btn = gr.Button("下载核心三件套（约 1.9 GB）", size="sm")
                    dl_qwen_btn = gr.Button("下载 Qwen 视觉+大模型（约 6 GB）", size="sm")
                    dl_all_btn = gr.Button("全部下载", size="sm")
                dl_md = gr.Markdown()

            with gr.Row():
                save_btn = gr.Button("保存并应用", variant="primary")
                reload_btn = gr.Button("重新读取当前设置", size="sm")
            save_md = gr.Markdown()
            backend_md = gr.Markdown()

            _setting_inputs = [
                llm_mode, llm_base, llm_model, llm_key,
                vl_enabled, vl_base, vl_model, vl_key, vl_frames,
                device, load_local,
            ]
            demo.load(
                fn=lambda: [*_settings_values(state), _format_backends_now(state)],
                outputs=[*_setting_inputs, backend_md],
                api_name="load_settings",
            )
            reload_btn.click(
                fn=lambda: [*_settings_values(state), _format_backends_now(state)],
                outputs=[*_setting_inputs, backend_md],
                api_name="reload_settings",
            )
            save_btn.click(
                fn=functools.partial(_save_settings, state),
                inputs=_setting_inputs,
                outputs=[save_md],
                api_name="save_settings",
            ).then(fn=lambda: _format_backends_now(state), outputs=[backend_md])
            llm_test_btn.click(
                fn=lambda b, m, k: _test_llm(state, b, m, k),
                inputs=[llm_base, llm_model, llm_key],
                outputs=[llm_test_md],
                api_name="test_llm",
            )
            vl_test_btn.click(
                fn=lambda b, m, k: _test_vl(b, m, k),
                inputs=[vl_base, vl_model, vl_key],
                outputs=[vl_test_md],
                api_name="test_vl",
            )
            for button, scope in (
                (dl_core_btn, "core"), (dl_qwen_btn, "qwen"), (dl_all_btn, "all"),
            ):
                button.click(
                    fn=lambda s=scope: _download_models(state, s),
                    outputs=[dl_md],
                    api_name=f"download_{scope}",
                )

        demo._zhiying_state = state  # type: ignore[attr-defined]  # 便于测试拿到编排器
        demo._zhiying_theme = str(getattr(cfg.gradio, "theme", "soft") or "soft")
    return demo


# =============================================================================
# CLI
# =============================================================================


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zhiying-web", description="知影 Web（Gradio）")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--share", action="store_true", help="生成临时公网链接")
    parser.add_argument("--load-models", action="store_true", help="启动时加载模型权重")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        ui = build_ui(config_path=args.config, load_models=args.load_models)
    except UiError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    state: _State = getattr(ui, "_zhiying_state")
    cfg = state.cfg
    host = args.host or cfg.gradio.host
    port = args.port or cfg.gradio.port
    print(f"知影 Web: http://{host}:{port}", file=sys.stderr)
    theme = None
    try:  # Gradio 6：theme 传 launch()；老版本传 Blocks() 时不认这个参数，忽略即可
        themes = __import__("gradio").themes
        name = getattr(ui, "_zhiying_theme", "soft") or "soft"
        theme = getattr(themes, name.capitalize(), None) or themes.Soft
        if isinstance(theme, type):  # gradio.themes.X 是类，launch() 需要实例
            theme = theme()
    except Exception:  # noqa: BLE001
        theme = None
    ui.launch(
        server_name=host,
        server_port=port,
        share=args.share or cfg.gradio.share,
        allowed_paths=[str(state.frames_dir)],
        show_error=True,
        theme=theme,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
