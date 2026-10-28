"""vision 阶段实现，仅通过编排器状态与其他阶段协作。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from backend.concurrency import submit
from concurrent.futures import as_completed


from backend.pipeline.domain import AnalysisReport
from backend.pipeline.helpers import (
    _group_by_scene,
    _keyframe_payload,
    _kf_ms,
    _ms_label,
    _segment_chunks,
)

logger = logging.getLogger("zhiying.pipeline")


class VisionStages:
    def _describe_visual(
        self,
        local_path: Path,
        keyframes: list[Any],
        report: AnalysisReport,
        scenes: list[Any],
        high_motion: bool,
    ) -> list[dict[str, Any]]:
        """视觉描述总分派：视频段 / 抽帧（均为云端）按配置与画像选路。"""
        vl = self._vl
        if vl is None or not keyframes:
            return [_keyframe_payload(kf, "", "") for kf in keyframes]
        mode = str(getattr(self._cfg.vl, "video_mode", "auto") or "auto")
        supports_seg = bool(getattr(vl, "supports_video_segments", False))
        want_segments = mode == "segments" or (mode == "auto" and high_motion)
        if want_segments and supports_seg and local_path and Path(local_path).is_file():
            report.profile["vision_mode_used"] = "segments"
            payloads = self._describe_by_segments(
                vl, Path(local_path), keyframes, report, scenes
            )
            if payloads:
                return payloads
            # 整条段通道没产出（全部失败）→ 落回抽帧，不能让视觉整体缺失
            report.profile["vision_mode_used"] = "segments→frames"
        report.profile["vision_mode_used"] = "frames" + (
            "-downgraded" if high_motion else ""
        )
        cloud = bool(getattr(vl, "is_cloud", False))
        if cloud:
            # D5：云端是网络型调用，不吃本地显存，无需推理锁，且可并发。
            return self._describe_keyframes(keyframes, report, parallel=True)
        with self._inference_slot("画面理解", report):
            return self._describe_keyframes(keyframes, report)

    def _describe_by_segments(
        self,
        vl: Any,
        video_path: Path,
        keyframes: list[Any],
        report: AnalysisReport,
        scenes: list[Any],
    ) -> list[dict[str, Any]]:
        """P1 视频段理解：场景聚成 ≤ video_segment_max_s 的时间窗，整段发云端。

        每窗产出一段含运动语义的描述（顺带读出画面烧录字幕），写进该窗全部
        关键帧；单窗失败退回该窗抽帧描述；段数受 ``video_segment_budget``
        约束（费用闸门），超出部分直接走抽帧。
        """
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        cfg_vl = self._cfg.vl
        chunks = _segment_chunks(scenes or [], int(cfg_vl.video_segment_max_s))
        if not chunks:
            return []
        budget = max(1, int(cfg_vl.video_segment_budget))
        seg_chunks, rest_chunks = chunks[:budget], chunks[budget:]
        if rest_chunks:
            report.warnings.append(
                f"视频段预算 {budget} 用满，其余 {len(rest_chunks)} 窗走抽帧描述"
            )
        by_scene: dict[int, list[Any]] = {}
        for kf in keyframes:
            by_scene.setdefault(int(getattr(kf, "scene_index", 0)), []).append(kf)

        payloads_by_id: dict[str, dict[str, Any]] = {}
        ok = degraded = 0
        errors: dict[str, int] = {}

        def run_segment(chunk: list[int]) -> tuple[list[int], Any]:
            start = int(scenes[chunk[0]].start_ms)
            end = int(scenes[chunk[-1]].end_ms)
            return chunk, vl.describe_video_segment(video_path, start, end)

        workers = max(1, min(cfg_vl.cloud_workers, len(seg_chunks)))
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="vl-seg"
        ) as pool:
            futures = [submit(pool, run_segment, chunk) for chunk in seg_chunks]
            for future in as_completed(futures):
                chunk, result = future.result()
                frames = [kf for idx in chunk for kf in by_scene.get(idx, [])]
                if result is None or getattr(result, "degraded", False):
                    degraded += 1
                    reason = str(getattr(result, "error", "") or "未知原因")
                    errors[reason] = errors.get(reason, 0) + 1
                    for payload in self._describe_frames_subset(vl, frames):
                        payloads_by_id[payload["keyframe_id"]] = payload
                else:
                    ok += 1
                    for kf in frames:
                        payloads_by_id[kf.keyframe_id] = _keyframe_payload(
                            kf, result.description, result.ocr_text
                        )

        # 预算外的窗：走抽帧描述（云端并发 / 本地串行）
        rest_scene_ids = {idx for chunk in rest_chunks for idx in chunk}
        rest_frames = [
            kf
            for kf in keyframes
            if int(getattr(kf, "scene_index", 0)) in rest_scene_ids
        ]
        if rest_frames:
            for payload in self._describe_keyframes(
                rest_frames, report, parallel=bool(getattr(vl, "is_cloud", False))
            ):
                payloads_by_id[payload["keyframe_id"]] = payload

        if ok or degraded:
            report.warnings.append(
                f"视频段理解: {ok} 窗成功、{degraded} 窗回退抽帧（≤{cfg_vl.video_segment_max_s}s/窗）"
            )
        for reason, count in errors.items():
            report.degraded.append(f"视频段理解降级 {count} 窗: {reason[:120]}")
        return [
            payloads_by_id.get(kf.keyframe_id, _keyframe_payload(kf, "", ""))
            for kf in keyframes
        ]

    def _describe_frames_subset(
        self, vl: Any, frames: list[Any]
    ) -> list[dict[str, Any]]:
        """一小段关键帧的抽帧描述（段通道单窗失败时的回退）。"""
        if not frames:
            return []
        describe_sequence = getattr(vl, "describe_sequence", None)
        if (
            bool(self._cfg.vl.sequence_mode)
            and callable(describe_sequence)
            and len(frames) > 1
        ):
            span = (_kf_ms(frames[0]), _kf_ms(frames[-1]))
            result = describe_sequence(
                [(kf.path, _kf_ms(kf)) for kf in frames], span_ms=span
            )
            return [
                _keyframe_payload(kf, result.description, result.ocr_text)
                for kf in frames
            ]
        out = []
        for kf in frames:
            result = vl.describe(kf.path)
            out.append(_keyframe_payload(kf, result.description, result.ocr_text))
        return out

    def _describe_keyframes(
        self, keyframes: list[Any], report: AnalysisReport, *, parallel: bool = False
    ) -> list[dict[str, Any]]:
        """关键帧 → 画面描述。连续模式下按场景分组，一次推理出一段叙述。

        ``parallel=True`` 仅用于云端后端（网络型，不吃本地显存）；
        测试注入后端可选择串行；生产只装配云端视觉后端。
        """
        vl = self._vl
        if vl is None:
            return [_keyframe_payload(kf, "", "") for kf in keyframes]
        payloads: list[dict[str, Any]] = []
        # 逐帧 append 会把"模型权重缺失"这一件事写成 N 条同样的告警（7 张帧 7 条），
        # 用户读到的是一屏废话。这里按原因归并，最后每个原因只报一次并带上帧数。
        vision_errors: dict[str, int] = {}
        describe_sequence = getattr(vl, "describe_sequence", None)
        sequence_ok = bool(self._cfg.vl.sequence_mode) and callable(describe_sequence)

        if sequence_ok:
            groups = _group_by_scene(
                keyframes,
                self._cfg.vl.cloud_sequence_frames
                if parallel
                else self._cfg.vl.sequence_frames,
            )

            def work(group: list[Any]) -> tuple[Any, list[Any]]:
                span = (_kf_ms(group[0]), _kf_ms(group[-1]))
                return describe_sequence(
                    [(kf.path, _kf_ms(kf)) for kf in group], span_ms=span
                ), group

            if parallel and len(groups) > 1:
                from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

                with ThreadPoolExecutor(
                    max_workers=min(self._cfg.vl.cloud_workers, len(groups)),
                    thread_name_prefix="vl-cloud",
                ) as pool:
                    futures = [submit(pool, work, group) for group in groups]
                    results = [f.result() for f in as_completed(futures)]
            else:
                results = [work(g) for g in groups]
            for result, group in results:
                span = (_kf_ms(group[0]), _kf_ms(group[-1]))
                if result.degraded:
                    reason = (str(result.error or "未知原因")).strip()
                    vision_errors[reason] = vision_errors.get(reason, 0) + len(group)
                else:
                    report.warnings.append(
                        f"连续画面理解: {_ms_label(span[0])}–{_ms_label(span[1])}"
                        f"（{len(group)} 帧）"
                    )
                for kf in group:
                    payloads.append(
                        _keyframe_payload(kf, result.description, result.ocr_text)
                    )
        else:

            def single(kf):
                return kf, vl.describe(kf.path)

            if parallel and len(keyframes) > 1:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(
                    max_workers=min(self._cfg.vl.cloud_workers, len(keyframes)),
                    thread_name_prefix="vl-frame",
                ) as pool:
                    futures = [submit(pool, single, kf) for kf in keyframes]
                    results = [f.result() for f in as_completed(futures)]
            else:
                results = [single(kf) for kf in keyframes]
            for kf, result in results:
                if result.degraded:
                    reason = (str(result.error or "未知原因")).strip()
                    vision_errors[reason] = vision_errors.get(reason, 0) + 1
                payloads.append(
                    _keyframe_payload(kf, result.description, result.ocr_text)
                )
        for reason, count in vision_errors.items():
            report.degraded.append(
                f"关键帧视觉理解降级 {count}/{len(keyframes)} 张: {reason[:120]}"
            )
        by_id = {payload["keyframe_id"]: payload for payload in payloads}
        return [by_id[kf.keyframe_id] for kf in keyframes]
