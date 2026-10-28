"""仅重建已入库视频的检索向量，不重新转写或调用云端模型。

使用前停止服务。默认备份数据库和向量库；新一代索引全部写入并校验后，
才切换视频的活动索引指针。失败时保留原索引。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
import socket
import sqlite3
import sys
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import load_app_config
from backend.runtime import runtime_path
from backend.models.embed import EmbeddingService
from backend.models.manager import ModelManager
from backend.rag.vector_store import VectorStore
from backend.storage.repository import SqliteRepository

logger = logging.getLogger(__name__)


def reindex_video(
    repo: Any, vectors: Any, embedder: Any, video_id: str
) -> dict[str, Any]:
    """编码全部成功后写新代，写入及条数校验成功后才发布。"""
    if embedder.degraded:
        raise RuntimeError("编码器仍在降级，拒绝用哈希向量覆盖现有索引")
    video = repo.get_video(video_id)
    clips = repo.get_clips(video_id)
    frames = repo.get_keyframes(video_id)
    if not clips:
        raise RuntimeError(f"视频 {video_id} 没有切片，无法重建")
    for frame in frames:
        if not Path(frame.path).is_file():
            raise FileNotFoundError(f"关键帧不存在：{frame.path}")
    # 两侧先完成实际推理；任一模型不可用时不会写入新索引。
    texts = [c.text or c.core_text for c in clips]
    text_vectors = embedder.embed_texts(texts)
    image_vectors = embedder.embed_images([f.path for f in frames]) if frames else []
    if len(text_vectors) != len(clips) or len(image_vectors) != len(frames):
        raise RuntimeError("编码条数与输入条数不一致，未写入索引")
    generation = uuid4().hex[:12]
    stage = vectors.with_generation(generation)
    published = False
    try:
        stage.upsert_clips(
            ids=[c.clip_id for c in clips],
            embeddings=text_vectors,
            documents=texts,
            metadatas=[
                {
                    "clip_id": c.clip_id,
                    "video_id": video_id,
                    "index_no": c.index_no,
                    "start_ms": c.start_ms,
                    "end_ms": c.end_ms,
                    "core_start_ms": c.core_start_ms,
                    "core_end_ms": c.core_end_ms,
                    "encoder_model": embedder.text_model_name,
                }
                for c in clips
            ],
        )
        if frames:
            stage.upsert_frames(
                ids=[f.keyframe_id for f in frames],
                embeddings=image_vectors,
                documents=[f.description or f.keyframe_id for f in frames],
                metadatas=[
                    {
                        "keyframe_id": f.keyframe_id,
                        "clip_id": f.clip_id or "",
                        "video_id": video_id,
                        "timestamp_ms": f.timestamp_ms,
                        "keyframe_path": f.path,
                        "encoder_model": embedder.image_model_name,
                    }
                    for f in frames
                ],
            )
        if stage.count("clip") != len(clips) or stage.count("frame") != len(frames):
            raise RuntimeError("新索引条数校验失败，保留原索引")
        repo.update_video_meta(video_id, vector_generation=generation)
        published = True
    finally:
        if not published:
            vectors.drop_generation(generation)
    # 活动指针已切换，报告更新失败也不能删除已发布的新索引。
    payload = repo.get_video_report(video_id)
    if payload:
        payload["vector_count"] = len(clips) + len(frames)
        payload.setdefault("profile", {}).update(
            {
                "text_encoder": embedder.text_model_name,
                "image_encoder": embedder.image_model_name,
            }
        )
        try:
            repo.set_video_report(video_id, payload)
        except Exception:
            logger.warning("索引已发布，但报告标记更新失败", exc_info=True)
    previous = video.vector_generation or ""
    if previous and previous != generation:
        try:
            vectors.drop_generation(previous)
        except Exception:
            logger.warning("新索引已发布，旧代清理失败：%s", previous, exc_info=True)
    return {
        "video_id": video_id,
        "原索引代": previous,
        "新索引代": generation,
        "文本向量": len(clips),
        "图像向量": len(frames),
        "文本编码器": embedder.text_model_name,
        "图像编码器": embedder.image_model_name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="重建全部已就绪视频")
    group.add_argument("--video-id", action="append", help="指定视频 ID，可重复传入")
    parser.add_argument("--dry-run", action="store_true", help="只列出切片与关键帧数量")
    parser.add_argument("--batch-size", type=int, help="模型推理批次大小，默认读取配置")
    parser.add_argument(
        "--backup-dir", type=Path, help="备份目录，默认项目 runtime/backups 下"
    )
    parser.add_argument("--output", type=Path, help="重建结果 JSON 路径")
    args = parser.parse_args()
    cfg = load_app_config(
        overrides={"embedding.batch_size": args.batch_size} if args.batch_size else None
    )
    if not args.dry_run:
        try:
            with socket.create_connection(("127.0.0.1", cfg.api.port), timeout=0.5):
                print("请先停止 VideoLens 服务，再执行索引重建。", file=sys.stderr)
                return 2
        except OSError:
            pass
    repo = SqliteRepository(cfg.path("sqlite"))
    store = None
    embedder = None
    try:
        if args.all:
            targets = []
            offset = 0
            while True:
                page = repo.list_videos(limit=100, offset=offset, status="ready")
                targets.extend(v.video_id for v in page)
                if len(page) < 100:
                    break
                offset += len(page)
        else:
            targets = list(dict.fromkeys(args.video_id))
        if args.dry_run:
            print(
                json.dumps(
                    [
                        {
                            "video_id": v,
                            "切片": len(repo.get_clips(v)),
                            "关键帧": len(repo.get_keyframes(v)),
                        }
                        for v in targets
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        backup = args.backup_dir or runtime_path("backups") / datetime.now().strftime(
            "reindex-%Y%m%d-%H%M%S"
        )
        backup.mkdir(parents=True, exist_ok=False)
        with (
            sqlite3.connect(cfg.path("sqlite")) as src,
            sqlite3.connect(backup / "zhiying.db") as dst,
        ):
            src.backup(dst)
        if cfg.path("chroma").exists():
            shutil.copytree(cfg.path("chroma"), backup / "chroma")
        embedder = EmbeddingService.from_config(
            cfg, manager=ModelManager(allow_download=False)
        )
        store = VectorStore.from_config(cfg)
        results = []
        for video_id in targets:
            print("正在编码并重建：" + video_id, flush=True)
            results.append(reindex_video(repo, store, embedder, video_id))
        result = {"备份": str(backup), "重建结果": results}
        output = args.output or runtime_path("state/reindex-result.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        if embedder:
            embedder.unload()
        if store:
            store.close()
        repo.close()


if __name__ == "__main__":
    raise SystemExit(main())
