"""segmentation 阶段实现，仅通过编排器状态与其他阶段协作。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


from backend.pipeline.domain import AnalysisReport, AnalyzeOptions
from backend.pipeline.helpers import (
    _clip_resolver,
    _keyframe_payload,
    _merge_scenes,
    _synthetic_clips,
)

logger = logging.getLogger("zhiying.pipeline")


class SegmentationStages:
    def _detect_scenes(
        self, local_path: Path, probe_info: Any
    ) -> tuple[list[Any], str, list[str]]:
        if self._scene is None:
            return [], "none", ["场景检测不可用，切片将只依赖转录"]
        detector = self._scene
        warnings: list[str] = []
        # D4/Tier2：长视频检测加速。优先 ffmpeg+NVDEC 硬解+降采样（快一个
        # 量级）；不可用时退 pyscedetect 跳帧档。PySceneDetect 在 Python 里
        # 逐帧算 HSV 差分，高运动长视频实测 6 分半要 10 分钟 —— 检测不该
        # 比视频本身长。
        try:
            duration_ms = int(getattr(probe_info, "duration_ms", 0) or 0)
        except Exception:  # noqa: BLE001
            duration_ms = 0
        long_video = 0 < self._cfg.media.scene_detect.long_video_min_ms <= duration_ms
        if long_video and detector is not None:
            boosted = self._long_video_detector()
            if boosted is not None:
                detector = boosted
                warnings.append(
                    "长视频场景检测走 ffmpeg/NVDEC 硬解 + 480px 打分（切点精度略降，耗时大幅下降）"
                )
            else:
                boost = self._scene_boost(duration_ms)
                if boost is not None:
                    try:
                        detector = self._boosted_scene_detector(*boost)
                        warnings.append(
                            f"长视频场景检测加速（frame_skip={boost[0]}, downscale={boost[1]}）："
                            f"切点精度略降，检测耗时大幅下降"
                        )
                    except Exception as exc:  # noqa: BLE001 - 加速失败退回原配置
                        logger.debug("构造加速检测器失败，退回默认: %s", exc)
                        detector = self._scene
        try:
            scenes, det, warns = detector.detect_scenes(local_path, probe_info)
            return list(scenes), str(det), list(warns) + warnings
        except Exception as exc:  # noqa: BLE001
            logger.warning("场景检测失败: %s", exc)
            return [], "none", [f"场景检测失败: {exc}"]

    def _scene_boost(self, duration_ms: int) -> tuple[int, int] | None:
        """长视频的 pyscedetect 加速档（frame_skip/downscale）。

        注意 downscale 不是 SceneSection 的配置项（检测器默认 2），
        这里以"不小于默认值"为下限取 long_video_downscale。
        """
        sec = self._cfg.media.scene_detect
        if sec.long_video_min_ms <= 0 or duration_ms < sec.long_video_min_ms:
            return None
        if sec.detector not in {"content", "adaptive", "hash"}:
            return None  # ffmpeg/fixed 档本来就不慢
        return (
            max(int(sec.frame_skip or 0), int(sec.long_video_frame_skip)),
            max(2, int(sec.long_video_downscale)),
        )

    def _boosted_scene_detector(self, frame_skip: int, downscale: int) -> Any:
        return self._scene_detector_with(frame_skip=frame_skip, downscale=downscale)

    def _long_video_detector(self) -> Any | None:
        """长视频专用检测器（Tier 2）：ffmpeg + NVDEC 硬解 + 480px 打分。

        PySceneDetect 是 Python 逐帧软解，高运动长视频实测比视频本身还长；
        ffmpeg 档走解码器硬解 + 缩小画面后再打分，快一个量级。构造失败
        （配置/参数问题）返回 None，调用方退回 pyscedetect 加速档。
        """
        sec = self._cfg.media.scene_detect
        if not sec.long_video_use_ffmpeg:
            return None
        try:
            return self._scene_detector_with(
                detector="ffmpeg",
                ffmpeg_hwaccel=bool(sec.long_video_hwaccel),
                ffmpeg_detect_width=int(sec.long_video_detect_width),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("构造 ffmpeg 硬解检测器失败: %s", exc)
            return None

    def _chunk(self, video_id: str, segments: list[Any], duration_ms: int) -> list[Any]:
        if self._chunker is None:
            return []
        if not segments:
            # 无转录时按固定间隔造切片，至少让视觉链路有落点

            return _synthetic_clips(video_id, duration_ms, self._cfg.chunk.max_ms)
        return list(
            self._chunker.chunk(
                video_id, segments, total_duration_ms=duration_ms or None
            )
        )

    def _prepare_keyframes(
        self,
        local_path: Path,
        video_id: str,
        scenes: list[Any],
        clips: list[Any],
        opts: AnalyzeOptions,
        report: AnalysisReport,
        *,
        high_motion: bool = False,
    ) -> tuple[list[Any], list[Any]]:
        if self._scene is None or opts.skip_vision:
            return [], scenes
        out_dir = Path(self._cfg.path("frames")) / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        clip_bounds = [(c.clip_id, c.core_start_ms, c.core_end_ms) for c in clips]
        resolver = _clip_resolver(clip_bounds)

        # D3 高运动降档：场景密度高的视频（体育/比赛），把相邻场景合并到
        # >= high_motion_scene_span_ms 再抽帧、每场景只留 1 帧 —— 静帧本来就
        # 区分不出两次进攻，多抽只是烧钱烧时间。
        detector = self._scene
        eff_scenes = scenes
        if high_motion and scenes:
            span_ms = int(self._cfg.vl.high_motion_scene_span_ms)
            eff_scenes = _merge_scenes(scenes, span_ms)
            try:
                detector = self._scene_detector_with(
                    frames_per_scene=1, min_frames_per_scene=1
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("构造降档检测器失败，沿用原配置: %s", exc)
            report.warnings.append(
                f"高运动视频视觉降档：{len(scenes)} 场景合并为 {len(eff_scenes)} 窗"
                f"（≥{span_ms // 1000}s/窗，每窗 1 帧）"
            )

        try:
            extraction = detector.extract_keyframes(
                local_path,
                video_id,
                out_dir,
                scenes=eff_scenes or None,
                clip_resolver=resolver,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("关键帧抽取失败: %s", exc)
            report.degraded.append(f"关键帧抽取失败: {exc}")
            return [], scenes

        keyframes = list(getattr(extraction, "keyframes", []) or [])
        if opts.max_keyframes:
            keyframes = keyframes[: opts.max_keyframes]
        report.warnings.extend(getattr(extraction, "warnings", []) or [])
        if getattr(extraction, "degraded", False):
            report.degraded.append("关键帧抽取降级（部分帧质量不达标）")

        return keyframes, eff_scenes

    def _describe_and_store(
        self, local_path, video_id, keyframes, scenes, opts, report, high_motion
    ):
        # 视觉理解（描述）：segments / frames 按 vl.video_mode 与高运动画像分派
        payloads: list[dict[str, Any]] = []
        if self._vl is not None and not opts.skip_vision:
            payloads = self._describe_visual(
                local_path, keyframes, report, scenes, high_motion
            )
        else:
            payloads = [_keyframe_payload(kf, "", "") for kf in keyframes]
        if payloads:
            self._repo.add_keyframes(video_id, payloads)
            try:
                self._repo.set_keyframe_binding(video_id)
            except Exception:  # noqa: BLE001
                logger.debug("回填 clip.keyframe_ids 失败", exc_info=True)
        else:
            # 整轮抽帧失败：清掉上一轮的记录，别让指向已删文件的孤儿帧继续被检索到
            try:
                self._repo.clear_keyframes(video_id)
            except Exception:  # noqa: BLE001
                logger.debug("清理旧关键帧记录失败", exc_info=True)
        return keyframes

    def _scene_detector_with(self, **overrides: Any) -> Any:
        """按当前配置构造场景检测器，允许临时覆盖若干参数（D3 降档用）。"""
        from backend.segmentation.scenes import SceneDetectConfig, SceneDetector  # noqa: PLC0415

        payload = self._cfg.media.scene_detect.flattened()
        payload.update({k: v for k, v in overrides.items()})
        known = set(SceneDetectConfig.__dataclass_fields__)
        return SceneDetector(
            SceneDetectConfig(**{k: v for k, v in payload.items() if k in known})
        )
