"""indexing 阶段实现，仅通过编排器状态与其他阶段协作。"""

from __future__ import annotations

import logging
from uuid import uuid4
import time
from backend.concurrency import parallel_pair
from backend.models.devices import active_devices
from pathlib import Path
from typing import Any

from backend.pipeline.domain import PipelineError

from backend.pipeline.domain import AnalysisReport, AnalyzeOptions
from backend.pipeline.helpers import _clip_index

logger = logging.getLogger("zhiying.pipeline")


class IndexingStages:
    def _encode_index_inputs(self, clips, keyframes, report, opts):
        if opts.skip_vectors:
            return None
        if self._embedder is None:
            raise PipelineError("缺少必需编码器，禁止降级")

        def encode(label, model, count, fn):
            started = time.monotonic()
            logger.info(
                "向量编码开始 model=%s device=%s items=%d",
                model, active_devices(self)[model], count,
            )
            with self._inference_slot(label, report):
                values = fn()
            logger.info("向量编码完成 model=%s items=%d", model, len(values))
            return values, int((time.monotonic() - started) * 1000)

        started = time.monotonic()
        text, image = parallel_pair(
            lambda: encode(
                "文本向量编码",
                "bge-m3", len(clips),
                lambda: self._embedder.embed_texts(
                    [c.text or c.core_text for c in clips]
                ),
            ),
            lambda: (
                encode(
                    "图像向量编码",
                    "siglip2-base", len(keyframes),
                    lambda: self._embedder.embed_images(
                        [Path(k.path) for k in keyframes]
                    ),
                )
                if keyframes
                else ([], 0)
            ),
            enabled=not self._cfg.limits.inference_serial,
            prefix="index-encode",
        )
        return {
            "text": text[0],
            "image": image[0],
            "text_ms": text[1],
            "image_ms": image[1],
            "elapsed_ms": int((time.monotonic() - started) * 1000),
        }

    def _index(
        self,
        video_id: str,
        clips: list[Any],
        keyframes: list[Any],
        report: AnalysisReport,
        opts: AnalyzeOptions,
        *,
        encoded=None,
    ) -> int:
        if opts.skip_vectors:
            return 0
        if self._vectors is None or self._embedder is None:
            raise PipelineError("缺少必需向量库或编码器，禁止降级")

        encoded = encoded or self._encode_index_inputs(clips, keyframes, report, opts)
        if encoded.get("error") is not None:
            raise PipelineError(f"必需向量编码失败：{encoded['error']}") from encoded[
                "error"
            ]
        written = 0
        # 本轮写进新的一代，全部写完才切指针；中途失败只丢半成品，
        # 上一代索引保持可用（否则重跑一次失败的代价是检索彻底空手）。
        store = self._vectors
        generation = uuid4().hex[:12]
        try:
            store = self._vectors.with_generation(generation)
        except Exception as exc:  # noqa: BLE001
            raise PipelineError(f"无法创建新索引代：{exc}") from exc

        # 文本向量
        try:
            texts = [c.text or c.core_text for c in clips]
            embeddings = encoded["text"]
            written += store.upsert_clips(
                ids=[c.clip_id for c in clips],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "clip_id": c.clip_id,
                        "video_id": video_id,
                        "index_no": _clip_index(c),
                        "start_ms": int(c.start_ms),
                        "end_ms": int(c.end_ms),
                        "core_start_ms": int(c.core_start_ms),
                        "core_end_ms": int(c.core_end_ms),
                    }
                    for c in clips
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("文本向量写入失败: %s", exc)
            store.drop_generation(generation)
            raise PipelineError(f"BGE-M3 编码或写入失败: {exc}") from exc

        # 图像向量
        if keyframes:
            try:
                image_vectors = encoded["image"]
                descriptions = {k.keyframe_id: "" for k in keyframes}
                for record in self._repo.get_keyframes(video_id):
                    descriptions[record.keyframe_id] = record.description
                written += store.upsert_frames(
                    ids=[k.keyframe_id for k in keyframes],
                    embeddings=image_vectors,
                    documents=[
                        descriptions.get(k.keyframe_id, "") or k.keyframe_id
                        for k in keyframes
                    ],
                    metadatas=[
                        {
                            "keyframe_id": k.keyframe_id,
                            "clip_id": k.clip_id or "",
                            "video_id": video_id,
                            "timestamp_ms": int(k.timestamp_ms),
                            "keyframe_path": str(k.path),
                        }
                        for k in keyframes
                    ],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("图像向量写入失败: %s", exc)
                store.drop_generation(generation)
                raise PipelineError(f"SigLIP2 编码或写入失败: {exc}") from exc

        try:
            self._repo.checkpoint()
        except Exception:  # noqa: BLE001
            pass

        if generation and written > 0:
            self._commit_generation(video_id, generation, report)
        elif generation:
            try:
                self._vectors.drop_generation(generation)
            except Exception:  # noqa: BLE001
                pass
        return written

    def _commit_generation(
        self, video_id: str, generation: str, report: AnalysisReport
    ) -> None:
        """把活动指针切到新代，再清掉上一代。

        顺序不能反：先删旧的再写指针，中间崩溃就没有任何一代可用。
        """
        previous = ""
        try:
            video = self._repo.find_video(video_id)
            previous = video.vector_generation if video else ""
            self._repo.update_video_meta(video_id, vector_generation=generation)
        except Exception as exc:  # noqa: BLE001
            logger.warning("索引指针切换失败: %s", exc)
            # 指针没切过去，检索仍走上一代；本轮写满的新代已成孤儿，必须清掉，
            # 否则它会永远留在向量库里（stats 双计数、磁盘白占）。
            try:
                self._vectors.drop_generation(generation)
            except Exception:  # noqa: BLE001
                logger.debug("清理未提交的索引代失败: %s", generation, exc_info=True)
            raise PipelineError(f"索引指针切换失败：{exc}") from exc
        if previous and previous != generation:
            try:
                self._vectors.drop_generation(previous)
            except Exception as exc:  # noqa: BLE001
                logger.debug("清理上一代索引失败（不影响检索）: %s", exc)
        elif not previous:
            # 旧数据没有代号：基础集合里可能还留着上一轮按 video_id 写的向量。
            # 新代已经接管检索，这些行留着只占盘。
            try:
                self._vectors.delete_video(video_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("清理旧基础集合失败（不影响检索）: %s", exc)
