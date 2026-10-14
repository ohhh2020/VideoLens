"""多路检索与证据构建。

不同模型的余弦分数不可直接比较。RRF 只比较各路的名次，再按权重融合。
Chroma 的每一代集合都有独立名字，完成写入之后才更新 SQLite 的活动指针。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from uuid import uuid4


def timestamp(ms: int) -> str:
    seconds, milli = divmod(max(0, int(ms)), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milli:03d}"


def keywords(text: str, limit: int = 12) -> list[str]:
    """关键词是离线降级的依据，不要求任何生成模型可用。"""
    try:
        import jieba.analyse

        return jieba.analyse.extract_tags(text, topK=limit)
    except ImportError:
        return list(dict.fromkeys(re.findall(r"[\w\u4e00-\u9fff]{2,}", text.lower())))[:limit]


def fuse_rankings(rankings: dict[str, list[str]], weights: dict[str, float], constant: int = 60) -> list[tuple[str, float]]:
    active = {key: values for key, values in rankings.items() if values and weights.get(key, 0) > 0}
    total = sum(weights[key] for key in active)
    scores: dict[str, float] = {}
    if not total:
        return []
    for name, values in active.items():
        # 同一切片可能有多个命中帧，每一路只计算一次该切片。
        for rank, identity in enumerate(dict.fromkeys(values), 1):
            scores[identity] = scores.get(identity, 0) + weights[name] / total / (constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def clip_refs(clip: dict, frames: list[dict]) -> list[dict]:
    refs = []
    start, end = clip["start_ms"], clip["end_ms"]
    if clip.get("text", "").strip():
        refs.append({"id": "t:" + clip["id"], "clip_id": clip["id"], "start_ms": start, "end_ms": end,
                     "start_time": timestamp(start), "end_time": timestamp(end), "text": clip["text"], "source": "transcript"})
    for frame in frames:
        if frame["clip_id"] != clip["id"]:
            continue
        content = "\n".join(str(frame.get(key, "")) for key in ("description", "ocr")).strip()
        if content:
            ms = frame["timestamp_ms"]
            refs.append({"id": "f:" + frame["id"], "clip_id": clip["id"], "frame_id": frame["id"],
                         "start_ms": ms, "end_ms": ms, "start_time": timestamp(ms), "end_time": timestamp(ms),
                         "text": content, "source": "frame"})
    return refs


class IndexManager:
    def __init__(self, settings, store, gateway):
        self.settings, self.store, self.gateway = settings, store, gateway
        self._client = None
        self.last_warnings: list[str] = []
        self.pending_meta: dict | None = None

    def client(self):
        if self._client is None:
            os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
            from chromadb import PersistentClient
            from chromadb.config import Settings as ChromaSettings

            self._client = PersistentClient(path=str(Path(self.settings.data_dir) / "chroma"),
                                            settings=ChromaSettings(anonymized_telemetry=False))
        return self._client

    def build(self, video_id: str, clips: list[dict], frames: list[dict], check, activate: bool = True) -> list[str]:
        self._building = []
        try:
            return self._build(video_id, clips, frames, check, activate)
        except BaseException:
            for name in self._building:
                try:
                    self.client().delete_collection(name)
                except Exception:
                    pass
            raise

    def _build(self, video_id: str, clips: list[dict], frames: list[dict], check, activate: bool) -> list[str]:
        generation = uuid4().hex[:12]
        ready: dict[str, str] = {}
        warnings = []
        previous = self.store.get_meta("index:" + video_id, {})
        for route in ("text", "image"):
            check()
            name = f"vl_{route}_{video_id}_{generation}"
            try:
                collection = self.client().create_collection(name, embedding_function=None, metadata={"hnsw:space": "cosine"})
                self._building.append(name)
                if route == "text":
                    rows = [(c["id"], "\n".join(r["text"] for r in clip_refs(c, frames))) for c in clips]
                    rows = [(identity, text) for identity, text in rows if text.strip()]
                    for offset in range(0, len(rows), 16):
                        check()
                        batch = rows[offset:offset + 16]
                        vectors = self.gateway.embed_texts([text for _, text in batch], check)
                        collection.upsert(ids=[identity for identity, _ in batch], embeddings=vectors,
                                          documents=[text for _, text in batch],
                                          metadatas=[{"clip_id": identity} for identity, _ in batch])
                else:
                    rows = frames
                    for offset in range(0, len(rows), 8):
                        check()
                        batch = rows[offset:offset + 8]
                        vectors = self.gateway.embed_images([f["path"] for f in batch], check)
                        collection.upsert(ids=[f["id"] for f in batch], embeddings=vectors,
                                          metadatas=[{"clip_id": f["clip_id"]} for f in batch])
                if rows:
                    ready[route] = name
                else:
                    self.client().delete_collection(name)
            except Exception as exc:
                from .errors import TaskInterrupted

                if isinstance(exc, TaskInterrupted):
                    raise
                warnings.append(f"{'文本' if route == 'text' else '图像'}向量索引不可用，已保留关键词检索")
                if self._client:
                    try:
                        self.client().delete_collection(name)
                    except Exception:
                        pass
            finally:
                self.gateway.release()
        check()
        self.pending_meta = ({"collections": ready, "generation": generation,
                              "text_model": self.settings.text_embedding_model,
                              "image_model": self.settings.image_embedding_model,
                              "model_versions": self.settings.model_versions()} if ready else None)
        if not activate:
            return warnings
        # 重建失败时保留上一代可用索引，避免一次网络/模型故障毁掉旧索引。
        if ready and (not previous or all(key in ready for key in previous.get("collections", {}))):
            self.store.set_meta("index:" + video_id, self.pending_meta)
            for name in previous.get("collections", {}).values():
                try:
                    self.client().delete_collection(name)
                except Exception:
                    pass
        else:
            for name in ready.values():
                try:
                    self.client().delete_collection(name)
                except Exception:
                    pass
        return warnings

    def delete(self, video_id: str) -> None:
        if not (Path(self.settings.data_dir) / "chroma").exists() and not self.store.get_meta("index:" + video_id):
            return
        self.prune(video_id, set())

    def prune(self, video_id: str, keep: set[str]):
        """删除该视频所有未使用的代，包括崩溃遗留；不会触及其他视频集合。"""
        prefixes = (f"vl_text_{video_id}_", f"vl_image_{video_id}_")
        for item in self.client().list_collections():
            name = getattr(item, "name", item)
            if name.startswith(prefixes) and name not in keep:
                self.client().delete_collection(name)

    def search(self, video_id: str, query: str, image_path: Path | None = None, check=lambda: None) -> list[dict]:
        clips, frames = self.store.clips(video_id), self.store.frames(video_id)
        by_id = {c["id"]: c for c in clips}
        meta = self.store.get_meta("index:" + video_id, {})
        rankings: dict[str, list[str]] = {}
        self.last_warnings = []
        terms = keywords(query, 20)
        if query.strip():
            terms = list(dict.fromkeys([query.strip().lower(), *terms]))
            scores = []
            for clip in clips:
                body = "\n".join(r["text"] for r in clip_refs(clip, frames)).lower()
                score = sum(body.count(term.lower()) * (2 if term == query.lower() else 1) for term in terms)
                if score:
                    scores.append((clip["id"], score))
            rankings["keyword"] = [x[0] for x in sorted(scores, key=lambda x: -x[1])[:self.settings.candidate_count]]
        for route in ("text", "image"):
            name = meta.get("collections", {}).get(route)
            if not name or (route == "text" and not query.strip()):
                continue
            expected = self.settings.text_embedding_model if route == "text" else self.settings.image_embedding_model
            if meta.get(route + "_model") != expected or meta.get("model_versions", {}).get(route) != self.settings.model_versions().get(route):
                self.last_warnings.append("向量模型已更换，请重建索引")
                continue
            try:
                check()
                if route == "text":
                    vector = self.gateway.embed_texts([query], check)[0]
                elif image_path:
                    vector = self.gateway.embed_images([str(image_path)], check)[0]
                else:
                    vector = self.gateway.embed_image_query(query, check)
                collection = self.client().get_collection(name, embedding_function=None)
                result = collection.query(query_embeddings=[vector], n_results=min(self.settings.candidate_count, collection.count()), include=["metadatas"])
                rankings[route] = [r["clip_id"] for r in result["metadatas"][0]]
            except Exception as exc:
                from .errors import TaskInterrupted

                if isinstance(exc, TaskInterrupted):
                    raise
                self.last_warnings.append(f"{route} 检索失败，使用其他可用检索结果")
            finally:
                self.gateway.release()
        weights = {"text": self.settings.text_weight, "image": self.settings.image_weight, "keyword": self.settings.keyword_weight}
        ranking = fuse_rankings(rankings, weights, self.settings.rrf_constant)
        results = []
        for identity, score in ranking[:self.settings.retrieval_top_k]:
            clip = by_id.get(identity)
            if clip is None:
                continue
            related = [public_frame(f) for f in frames if f["clip_id"] == identity]
            results.append({"clip_id": identity, "start_ms": clip["start_ms"], "end_ms": clip["end_ms"],
                            "start_time": timestamp(clip["start_ms"]), "end_time": timestamp(clip["end_ms"]),
                            "text": clip.get("text", ""), "score": score, "refs": clip_refs(clip, frames), "frames": related})
        return results


def public_frame(frame: dict) -> dict:
    return {**{k: v for k, v in frame.items() if k not in ("path", "internal_path")}, "timestamp": timestamp(frame["timestamp_ms"])}
