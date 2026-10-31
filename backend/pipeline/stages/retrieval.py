"""retrieval 阶段实现，仅通过编排器状态与其他阶段协作。"""

from __future__ import annotations

import logging
from typing import Any

from backend.pipeline.domain import PipelineError

from backend.pipeline.helpers import (
    _item_keyframe_ids,
    _item_text,
    _item_with_text,
    _ms_label,
)

logger = logging.getLogger("zhiying.pipeline")


class RetrievalStages:
    def _attach_visuals(self, video_id: str, items: list[Any]) -> list[Any]:
        """把命中关键帧的画面描述拼进证据文本。

        图像通道能把「问的是什么画面」映射到相关帧，但帧是二进制、
        LLM 看不见；只有 ``VlService`` 写下的 description / ocr_text 是文字。
        不拼进上下文，问「画面里有什么颜色、屏幕上写着什么」时模型只能
        看到转写文本，会答成「没有提到」。这是跨模态检索能不能落地
        的关键一步，不是可选的加分项。
        """
        if not items:
            return items
        try:
            frames = {kf.keyframe_id: kf for kf in self._repo.get_keyframes(video_id)}
        except Exception as exc:  # noqa: BLE001 - 视觉注入失败不应影响问答
            logger.debug("读取关键帧失败，跳过视觉注入: %s", exc)
            return items
        if not frames:
            return items

        out: list[Any] = []
        # D2 跨片段折叠：运动类视频每段画面的描述几乎全同（"球员奔跑传球"），
        # 14 段同质证据进总结就是 8 遍复读 + 逼模型在结尾幻觉差异。
        # 首次出现的描述保留全文，相似（bigram-Dice ≥ 0.75）的后续片段折叠成
        # "画面与前段类似"，时间跨度保留 —— 覆盖率不丢，重复信息只算一次。
        seen_descs: list[str] = []
        for item in items:
            notes: list[str] = []
            # 一段连续画面会被写进同一组内的每一帧（同一段叙述 + 不同时间戳），
            # 这里按"叙述内容"归并，并把跨度显示成时间区间 —— 这才能让
            # 模型看出「这是 0:01 到 0:12 之间的一段」，而不是两个孤立帧。
            grouped: dict[str, list[int]] = {}
            for fid in _item_keyframe_ids(item):
                kf = frames.get(fid)
                if kf is None:
                    continue
                desc = str(getattr(kf, "description", "") or "").strip()
                ocr = str(getattr(kf, "ocr_text", "") or "").strip()
                if ocr and ocr not in desc:
                    desc = (
                        f"{desc}（画面文字：{ocr}）" if desc else f"（画面文字：{ocr}）"
                    )
                if not desc:
                    continue
                ms = int(getattr(kf, "timestamp_ms", 0) or 0)
                grouped.setdefault(desc, []).append(ms)
            if not grouped:
                out.append(item)
                continue
            joined = " ".join(
                d for d, _ in sorted(grouped.items(), key=lambda kv: min(kv[1]))
            )
            similar = next((s for s in seen_descs if s == joined), None)
            if similar is None:
                seen_descs.append(joined)
                for desc, stamps in sorted(grouped.items(), key=lambda kv: min(kv[1])):
                    lo, hi = min(stamps), max(stamps)
                    span = (
                        f"{_ms_label(lo)}–{_ms_label(hi)}"
                        if hi - lo > 500
                        else _ms_label(lo)
                    )
                    notes.append(f"【画面 {span}】{desc}")
            else:
                stamps_all = [ms for sts in grouped.values() for ms in sts]
                lo, hi = min(stamps_all), max(stamps_all)
                notes.append(
                    f"【画面 {_ms_label(lo)}–{_ms_label(hi)}】（画面与前段类似，此处不重复展开）"
                )
            extra = "\n".join(notes)
            base = _item_text(item).strip()
            out.append(_item_with_text(item, f"{base}\n{extra}" if base else extra))
        return out

    def _retrieve(self, video_id: str, query: str, top_k: int | None) -> list[Any]:
        from backend.rag.hybrid_search import (  # noqa: PLC0415
            CallableEmbedder,
            HybridConfig,
            HybridSearcher,
            VectorChannel,
        )

        channels: list[Any] = []
        # 检索走这个视频当前活动的索引代。代为空说明是旧数据或还没建过索引，
        # 用基础集合名 —— 不强行猜，查不到就由 BM25 兜底。
        store = self._vectors
        if store is not None:
            try:
                video = self._repo.find_video(video_id)
                generation = (video.vector_generation if video else "") or ""
                if generation:
                    store = store.with_generation(generation)
            except Exception as exc:  # noqa: BLE001
                logger.debug("读取索引代号失败，退回基础集合: %s", exc)
        if store is not None and self._embedder is not None:
            # 查询向量必须在这里同步算好。通道有 channel_timeout_ms（默认 3s）
            # 的硬超时，把首次模型调用放进通道线程，会让每个新进程的第一次
            # 提问必然超时、整段检索退化成 fallback —— 用户看到的就是
            # 「第一问永远答没有提到，第二问开始才正常」。
            from backend.concurrency import parallel_pair

            text_vec, image_vec = parallel_pair(
                lambda: self._query_vector(query, cross_modal=False),
                lambda: self._query_vector(query, cross_modal=True),
                enabled=not self._cfg.limits.inference_serial,
                prefix="query-encode",
            )
            if text_vec is not None:
                try:
                    # create=False：检索是读路径，集合不存在就跳过该通道，
                    # 不能因为一次查询凭空建出空集合（stats 的集合数会虚涨）。
                    clip_coll = store.collection("clip", create=False)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("文本向量通道不可用: %s", exc)
                    clip_coll = None
                if clip_coll is not None:
                    channels.append(
                        VectorChannel(
                            "text",
                            clip_coll,
                            CallableEmbedder(lambda _text, _v=text_vec: _v),
                        )
                    )
            if image_vec is not None:
                try:
                    frame_coll = store.collection("frame", create=False)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("图像向量通道不可用: %s", exc)
                    frame_coll = None
                if frame_coll is not None:
                    channels.append(
                        VectorChannel(
                            "image",
                            frame_coll,
                            CallableEmbedder(lambda _text, _v=image_vec: _v),
                            id_field="keyframe_id",
                        )
                    )

        payloads = list(self._repo.iter_clip_payloads(video_id))
        if getattr(self, '_use_integrity', False):
            from backend.analysis.qa import active_ledger
            ledger = active_ledger(self, video_id)
            if ledger:
                from collections import defaultdict
                bodies = defaultdict(list)
                for evidence in ledger.all_evidence():
                    bodies[evidence['clip_id']].append(evidence['text'])
                payloads = [{**p, 'text': '\n'.join(bodies.get(p['clip_id'], [])) or p.get('text', ''),
                             'core_text': '\n'.join(bodies.get(p['clip_id'], [])) or p.get('core_text', '')} for p in payloads]
        clip_index = {str(p["clip_id"]): p for p in payloads}
        bm25 = self._cached_bm25(video_id, payloads)
        if bm25 is not None:
            channels.append(bm25)

        if not channels:
            return []

        cfg = HybridConfig(
            top_k=top_k or self._cfg.retrieval.top_k,
            weights=dict(self._cfg.retrieval.weights),
            k=self._cfg.retrieval.rrf_k,
            min_score=self._cfg.retrieval.min_score,
            timeout_ms=self._cfg.retrieval.channel_timeout_ms,
        )
        keyframe_paths = self._repo.keyframe_paths(video_id)
        searcher = HybridSearcher(
            channels,
            cfg,
            clip_index=clip_index,
            keyframe_resolver=lambda fid: keyframe_paths.get(fid),
        )
        try:
            response = searcher.search(query, filters={"video_id": video_id})
        finally:
            # searcher 自带线程池（每实例最多 3 线程），不关就随提问泄漏。
            searcher.close()
        return list(response.items)

    def _cached_bm25(self, video_id: str, payloads: list[Any]) -> Any | None:
        """BM25 通道按视频缓存（切片数/末片 ID 变化即失效），最多留 4 个视频。"""
        if not payloads:
            return None
        from backend.rag.hybrid_search import BM25Channel  # noqa: PLC0415

        stamp = (video_id, len(payloads), hash(tuple((p.get("clip_id"), p.get("text")) for p in payloads)))
        with self._bm25_lock:
            hit = self._bm25_cache.get(video_id)
            if hit is not None and hit[0] == stamp:
                self._bm25_cache.move_to_end(video_id)
                return hit[1]
        try:
            channel = BM25Channel.from_clips(payloads)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 通道不可用: %s", exc)
            return None
        with self._bm25_lock:
            self._bm25_cache[video_id] = (stamp, channel)
            self._bm25_cache.move_to_end(video_id)
            while len(self._bm25_cache) > 4:
                self._bm25_cache.popitem(last=False)
        return channel

    def _query_vector(self, query, *, cross_modal):
        if self._embedder is None:
            raise PipelineError("缺少必需向量编码器")
        with self._inference_slot("查询向量编码", None):
            return (
                self._embedder.embed_cross_modal(query)
                if cross_modal
                else self._embedder.embed_query(query)
            )

    def _fallback_items(self, video_id: str, limit: int = 6) -> list[dict[str, Any]]:
        payloads = sorted(
            self._repo.iter_clip_payloads(video_id),
            key=lambda p: int(p.get("start_ms", 0)),
        )
        return payloads[:limit]
