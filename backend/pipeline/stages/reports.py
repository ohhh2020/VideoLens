"""reports 阶段实现，仅通过编排器状态与其他阶段协作。"""

from __future__ import annotations

import logging
import re
from typing import Any

from backend.pipeline.factory import _build_llm, _build_context

from backend.pipeline.domain import AnalyzeOptions, PipelineError
from backend.pipeline.helpers import (
    _chapter_fallback,
    _flatten_qa_summary,
    _ms_label,
    _parse_milestones,
    _strip_chapter_refs,
    _summary_budget,
)

logger = logging.getLogger("zhiying.pipeline")


def _preference_query(title, payloads):
    """只用视频证据召回偏好，排除总结提示词中的步骤/详略等格式要求。"""
    return title + "\n" + "\n".join(
        str(item.get("text") or item.get("summary") or item.get("visual_text") or "")
        for item in payloads
    )[:1900]


class ReportStages:
    def summarize(self, video_id: str, *, max_items: int | None = None) -> str:
        """重新生成整片总结（不重跑分析）。

        ``max_items=None``（默认）跟随"覆盖全片"预算；显式传值用于裁剪。
        """
        self._require_cloud_ready()
        video = self._repo.find_video(video_id)
        if video is None:
            raise PipelineError(f"视频不存在: {video_id}")
        payloads = list(self._repo.iter_clip_payloads(video_id))
        payloads = self._attach_visuals(video_id, payloads)
        from backend.pipeline.context_builder import ContextBuilder  # noqa: PLC0415
        from backend.prompts import get_prompt_registry  # noqa: PLC0415

        # 与分析期的 _summarize 同一套"覆盖全片"预算（历史 bug：这里沿用
        # 问答的 8 条预算 + 兜底提示词，重新生成的总结与分析期一样被压扁）。
        built = ContextBuilder(
            _summary_budget(len(payloads)), prompts=get_prompt_registry()
        ).build_summary(
            payloads,
            video_title=video.title or video.display_name,
            duration_ms=video.duration_ms,
            max_items=max_items,
        )
        if not built.ok:
            return "（没有足够的切片可供总结）"
        llm = self._llm or _build_llm(self._cfg, None)
        response = llm.complete(self._preference_messages(
            [{"role": "user", "content": built.text}], _preference_query(video.title or "", payloads), report=True
        ))
        return response.text

    def chapters(
        self,
        video_id: str,
        *,
        chapter_seconds: int = 300,
        max_chapters: int = 40,
        max_items: int = 8,
        min_seconds: int = 300,
        summarize: bool = True,
    ) -> list[Any]:
        """把长视频按章节切开并逐章摘要，返回 :class:`Chapter` 列表。

        短视频（短于 ``min_seconds``）返回**空列表** —— 半分钟就能读完的内容拆成
        三章只是让读者多点两次。调用方拿到空列表就继续走 :meth:`summarize`。

        每章独立调一次模型，所以长视频的耗时与章节数成正比；``summarize=False``
        只做分组与提取式摘要，用于"先看骨架再决定要不要逐章细化"的场景。
        """
        from backend.pipeline.chapters import Chapter, plan_chapters  # noqa: PLC0415

        video = self._repo.find_video(video_id)
        if video is None:
            raise PipelineError(f"视频不存在: {video_id}")
        if video.duration_ms and video.duration_ms < int(min_seconds) * 1000:
            return []
        payloads = list(self._repo.iter_clip_payloads(video_id))
        payloads = self._attach_visuals(video_id, payloads)
        if not payloads:
            return []
        by_id = {str(p.get("clip_id") or ""): p for p in payloads}
        plans = plan_chapters(
            payloads,
            chapter_seconds=chapter_seconds,
            duration_ms=video.duration_ms,
            max_chapters=max_chapters,
        )
        builder = self._ctx or _build_context(self._cfg)
        title = video.title or video.display_name or video_id
        llm = (self._llm or _build_llm(self._cfg, None)) if summarize else None

        out: list[Any] = []
        for plan in plans:
            items = [by_id[cid] for cid in plan.clip_ids if cid in by_id]
            if not items:
                continue
            built = builder.build_summary(
                items,
                video_title=title,
                duration_ms=plan.duration_ms,
                prompt_key="chapter_summarize",
                max_items=max_items,
            )
            summary, backend, degraded = "", "", False
            notes: list[str] = []
            if built.ok and llm is not None:
                try:
                    with self._inference_slot("章节摘要", None, local=False):
                        response = llm.complete(
                            self._preference_messages([{"role": "user", "content": built.text}], _preference_query(title, items), report=True),
                            temperature=0.2,
                        )
                    summary = (response.text or "").strip()
                    backend = response.backend
                except Exception as exc:  # noqa: BLE001
                    logger.warning("第 %d 章摘要失败: %s", plan.index, exc)
                    notes.append(f"摘要失败: {exc}")
                    degraded = True
            if not summary:
                summary = _chapter_fallback(items)
                backend = backend or "evidence"
                if not built.ok:
                    degraded = True
                    notes.append("片段太少，未能组成本章上下文")
            else:
                summary = _flatten_qa_summary(_strip_chapter_refs(summary))
            out.append(
                Chapter(
                    index=plan.index,
                    start_ms=plan.start_ms,
                    end_ms=plan.end_ms,
                    summary=summary,
                    clip_ids=plan.clip_ids,
                    backend=backend,
                    degraded=degraded,
                    notes=tuple(notes),
                )
            )
        return out

    def chapter_report(
        self,
        video_id: str,
        *,
        chapter_seconds: int = 300,
        max_chapters: int = 40,
        min_seconds: int = 300,
        total_summary: str = "",
    ) -> str:
        """对外报告入口（CLI export / ``/chapters/report``）。

        自 2026-10 起产出**三层报告**（综合总结 / 内容驱动脉络 / 修字解说词，
        见 :meth:`full_report`）；旧的"逐章摘要拼接"格式仍可经
        ``chapters_markdown`` 取到，但不再是默认导出。
        """
        return self.full_report(
            video_id,
            chapter_seconds=chapter_seconds,
            max_chapters=max_chapters,
            min_seconds=min_seconds,
            total_summary=total_summary,
        )

    def full_report(
        self,
        video_id: str,
        *,
        chapter_seconds: int = 300,
        max_chapters: int = 40,
        min_seconds: int = 300,
        total_summary: str = "",
    ) -> str:
        """三层报告：综合总结 / 内容驱动的脉络 / 修字后的详细解说词。

        改进 A：解说词是转写原文经 LLM **只修错字**（不增删不总结）的版本，
        单独成节 —— 长视频的"几千字详细内容"由它承担，而不是靠总结注水。
        改进 B：脉络不再按 30s 网格让模型逐格填空，而是逐章从素材里挑
        真实节点（时间取自证据时间戳，禁止均匀切分），再合并成全片脉络。

        每层都有降级：LLM 不可用 → 脉络取章摘要首句、解说词用原文、
        综合用章节摘要拼接 —— 报告永远出得来，只是层数变薄。
        """
        from backend.pipeline.chapters import Chapter, _first_clause  # noqa: PLC0415
        from backend.pipeline.context_builder import format_hms  # noqa: PLC0415
        from backend.prompts import get_prompt_registry  # noqa: PLC0415

        video = self._repo.find_video(video_id)
        if video is None:
            raise PipelineError(f"视频不存在: {video_id}")
        title = video.title or video.display_name or video_id
        duration_ms = video.duration_ms or 0
        cfg_report = self._cfg.report
        registry = get_prompt_registry()
        llm = self._llm or _build_llm(self._cfg, None)

        try:
            segment_rows = self._repo.get_segments(video_id)
        except Exception:  # noqa: BLE001
            segment_rows = []

        # 1) 章节骨架：长视频走既有逐章摘要；短视频合并为单章
        chapters = self.chapters(
            video_id,
            chapter_seconds=chapter_seconds,
            max_chapters=max_chapters,
            min_seconds=min_seconds,
        )
        if not chapters:
            summary = total_summary or self.summarize(video_id)
            chapters = [
                Chapter(
                    index=1, start_ms=0, end_ms=duration_ms, summary=summary.strip()
                )
            ]

        # 2) 逐章：内容驱动的里程碑 + 修字解说词
        milestone_lines: list[str] = []
        transcript_parts: list[str] = []
        for chapter in chapters:
            lo, hi = chapter.start_ms, max(chapter.end_ms, chapter.start_ms + 1)
            seg_texts = [
                (
                    int(getattr(r, "start_ms", 0) or 0),
                    str(getattr(r, "text", "") or "").strip(),
                )
                for r in segment_rows
                if lo <= int(getattr(r, "start_ms", 0) or 0) < hi
                and str(getattr(r, "text", "") or "").strip()
            ]
            chapter_notes: list[str] = []

            # 2a) 里程碑（改进 B）
            if seg_texts:
                context = "\n".join(
                    f"{_ms_label(ms)} {text}" for ms, text in seg_texts[:400]
                )[:6_000]
            else:
                context = f"（本章无语音，画面摘要）{chapter.summary}"
            try:
                prompt = registry.render(
                    "chapter_milestones",
                    title=title,
                    start_time=_ms_label(lo),
                    end_time=_ms_label(hi),
                    max_items=int(cfg_report.milestones_max_per_chapter),
                    context=context,
                )
                with self._inference_slot("章节脉络", None, local=False):
                    response = llm.complete(
                        [{"role": "user", "content": prompt}], temperature=0.2
                    )
                lines = _parse_milestones(
                    str(response.text or ""),
                    lo,
                    hi,
                )
            except Exception as exc:  # noqa: BLE001 - 脉络失败降级为章摘要首句
                logger.debug("第 %d 章里程碑失败: %s", chapter.index, exc)
                lines = []
            if not lines:
                first_sentence = _first_clause(chapter.summary) or "本章内容"
                lines = [(lo, first_sentence)]
                chapter_notes.append("脉络降级：章摘要首句")
            milestone_lines.extend(
                # 脉络是秒级信息，毫秒尾数只是噪声
                f"- **{_ms_label(ms)[:-4]}** {text}"
                for ms, text in lines
            )

            # 2b) 修字解说词（改进 A）
            header = f"### {_ms_label(lo)} – {_ms_label(hi)}"
            if not seg_texts:
                transcript_parts.append(
                    f"{header}\n\n（本章暂无转写文字，请查看分析中的转写提示）"
                )
                continue
            raw_text = "\n".join(text for _, text in seg_texts)
            if not cfg_report.polish_transcript:
                transcript_parts.append(f"{header}\n\n{raw_text}")
                continue
            polished = self._polish_transcript(
                llm, registry, raw_text, lo, hi, notes=chapter_notes
            )
            transcript_parts.append(f"{header}\n\n{polished or raw_text}")

        # 3) 综合总结：各章摘要再综合（改进 A 第一层）
        if total_summary.strip():
            synthesis = total_summary.strip()
        else:
            digest = "\n\n".join(
                f"[第 {c.index} 章 {_ms_label(c.start_ms)}–{_ms_label(c.end_ms)}] "
                f"{(c.summary or '').strip()}"
                for c in chapters
            )
            try:
                prompt = registry.render(
                    "report_synthesis",
                    title=title,
                    duration=format_hms(duration_ms) if duration_ms else "未知",
                    context=digest[:12_000],
                )
                with self._inference_slot("综合总结", None, local=False):
                    response = llm.complete(
                        self._preference_messages([{"role": "user", "content": prompt}],title+digest[:1900], report=True), temperature=0.3
                    )
                synthesis = str(response.text or "").strip()
            except Exception as exc:  # noqa: BLE001
                logger.debug("综合总结生成失败，退回章节摘要拼接: %s", exc)
                synthesis = digest
            if not synthesis:
                synthesis = digest
        # 综合层自带 "## 内容概览/## 关键结论" 小节，降为三级避免与
        # 报告的二级节标题（综合总结/主要脉络/详细解说词）平级。
        synthesis = re.sub(r"^##\s+", "### ", synthesis, flags=re.M)

        transcript_chars = sum(len(p) for p in transcript_parts)
        lines = [
            f"# {title}",
            "",
            f"时长 {format_hms(duration_ms) if duration_ms else '未知'} ｜ "
            f"{len(chapters)} 章 ｜ 脉络 {len(milestone_lines)} 条 ｜ "
            f"解说词 {transcript_chars} 字",
            "",
            "## 综合总结",
            "",
            synthesis,
            "",
            "## 主要脉络",
            "",
            *milestone_lines,
            "",
            "## 详细解说词",
            "",
            *transcript_parts,
            "",
        ]
        return "\n".join(lines)

    def _polish_transcript(
        self,
        llm: Any,
        registry: Any,
        raw_text: str,
        lo: int,
        hi: int,
        *,
        notes: list[str] | None = None,
    ) -> str:
        """修字（只改错字/标点，不改写）。超长按块切分，逐块调用后拼接。"""
        cap = max(500, int(self._cfg.report.polish_block_chars))
        blocks: list[str] = []
        current: list[str] = []
        used = 0
        for line in raw_text.splitlines():
            if used + len(line) > cap and current:
                blocks.append("\n".join(current))
                current, used = [], 0
            current.append(line)
            used += len(line) + 1
        if current:
            blocks.append("\n".join(current))

        out: list[str] = []
        total = len(blocks)
        for i, block in enumerate(blocks, start=1):
            span_lo = _ms_label(lo) if i == 1 else f"（续 {i}/{total}）"
            span_hi = _ms_label(hi) if i == total else f"（待续 {i}/{total}）"
            try:
                prompt = registry.render(
                    "transcript_polish",
                    start_time=span_lo,
                    end_time=span_hi,
                    transcript=block,
                )
                with self._inference_slot("解说词修字", None, local=False):
                    response = llm.complete(
                        [{"role": "user", "content": prompt}], temperature=0.0
                    )
                text = str(response.text or "").strip()
                # 修字层不许 LLM 偷懒总结：输出明显短于输入一半时视为违约，
                # 该块退回原文（宁可保留错字，不能丢内容）。
                if text and len(text) >= len(block) * 0.5:
                    out.append(text)
                else:
                    out.append(block)
                    if notes is not None:
                        notes.append(f"解说词第 {i} 块修字输出过短，保留原文")
            except Exception as exc:  # noqa: BLE001 - 修字失败退原文
                logger.debug("解说词修字失败（块 %d/%d）: %s", i, total, exc)
                out.append(block)
                if notes is not None:
                    notes.append(f"解说词第 {i} 块修字失败，保留原文")
        return "\n\n".join(out)

    def _summarize(
        self, video_id: str, title: str, duration_ms: int, opts: AnalyzeOptions
    ) -> tuple[str, str]:
        if opts.skip_summary:
            return "", "skipped"
        if getattr(self, "_analysis_run", None) is not None:
            from backend.analysis.optimized import enabled as optimized
            if optimized(self):
                from backend.analysis.packing import summarize
                # 当前偏好仅由视频标题查询决定；预算试算不得反复编码同一查询。
                preference_prefix = self._preference_messages([], title, report=True)
                text = summarize(self._repo, video_id, title, self._llm, self._cfg.integrity, self._analysis_run,
                    preference=lambda messages: [*preference_prefix, *messages])
                return text, 'cloud'
            from backend.analysis.fast import enabled
            if enabled(self):
                from backend.analysis.fast import summarize
            else:
                from backend.analysis.summary import summarize
            text = summarize(self._repo, video_id, title, self._llm, self._cfg.integrity, self._analysis_run,
                preference=lambda messages: self._preference_messages(messages, title, report=True))
            return text, "cloud"
        payloads = list(self._repo.iter_clip_payloads(video_id))
        payloads = self._attach_visuals(video_id, payloads)
        from backend.pipeline.context_builder import ContextBuilder  # noqa: PLC0415
        from backend.prompts import get_prompt_registry  # noqa: PLC0415

        built = ContextBuilder(
            _summary_budget(len(payloads)), prompts=get_prompt_registry()
        ).build_summary(payloads, video_title=title, duration_ms=duration_ms)
        if not built.ok:
            return "", "none"
        llm = self._llm or _build_llm(self._cfg, None)
        try:
            with self._inference_slot("总结生成", None, local=False):
                response = llm.complete(self._preference_messages(
                    [{"role": "user", "content": built.text}], _preference_query(title, payloads), report=True
                ))
            return response.text, response.backend
        except Exception as exc:  # noqa: BLE001
            logger.warning("总结失败: %s", exc)
            raise PipelineError(f"云端总结失败：{exc}") from exc
