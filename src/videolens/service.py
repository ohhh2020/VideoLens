"""应用服务：把媒体、模型、索引与持久化任务连接成可恢复的完整流程。

接入端不直接操作数据库或模型。任务阶段成功后才写检查点；发生取消时，
当前阶段不被标为完成，恢复会安全重做该阶段，已完成阶段不重复生成。
"""

from __future__ import annotations

import hashlib
import json
import queue
import shutil
import threading
import time
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from pydantic import ValidationError

from . import media
from .config import SecretStore, Settings, atomic_json, load_settings
from .errors import TaskInterrupted, VideoLensError
from .generation import Composer, parse_object, validate_claims
from .models import ModelGateway
from .retrieval import IndexManager, clip_refs, public_frame, timestamp
from .store import Database

TERMINAL = {"succeeded", "partial", "failed", "cancelled", "paused"}
ACTIVE = {"queued", "running", "pause_requested", "cancel_requested"}


def _public(value):
    """内部字段以下划线开头；接入层永远只拿到白名单形式的公开结构。"""
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items() if not key.startswith("_") and key not in {"path", "config_path", "api_token", "ui_password", "key"}}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return value


class VideoService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.data_dir = Path(self.settings.data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for part in ("videos", "tmp", "exports"):
            (self.data_dir / part).mkdir(exist_ok=True)
        self.store = Database(self.data_dir / "videolens.sqlite3")
        self.secrets = SecretStore()
        self.gateway = ModelGateway(self.settings, self.secrets.get)
        self.index = IndexManager(self.settings, self.store, self.gateway)
        self.composer = Composer(self.settings, self.gateway)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._guard = threading.RLock()
        self._inference = threading.RLock()
        self._sessions: dict[str, threading.Lock] = {}
        self._last_cleanup = 0.0

    def start(self):
        if self._threads:
            return
        self._stop.clear()
        # 上次进程可能在任意阶段退出，只恢复真正未结束的任务。
        for job in reversed(self.store.list_jobs()):
            if job["status"] == "pause_requested":
                self.store.update_job(job["id"], status="paused")
            elif job["status"] == "cancel_requested":
                self.store.update_job(job["id"], status="cancelled")
            elif job["status"] in {"running", "queued"}:
                self.store.update_job(job["id"], status="queued")
                self._queue.put(job["id"])
        for i in range(self.settings.analysis_concurrency):
            worker = threading.Thread(target=self._worker, name=f"videolens-worker-{i}", daemon=True)
            worker.start()
            self._threads.append(worker)

    def close(self):
        self._stop.set()
        for _ in self._threads:
            self._queue.put(None)
        for worker in self._threads:
            worker.join(timeout=10)
        self._threads.clear()
        self.gateway.release()

    def _video(self, video_id):
        value = self.store.get_video(video_id)
        if value is None:
            raise VideoLensError("VIDEO_NOT_FOUND", "视频不存在", 404)
        return value

    def _job(self, identity):
        value = self.store.get_job(identity)
        if value is None:
            raise VideoLensError("TASK_NOT_FOUND", "任务不存在", 404)
        return value

    def _check_capacity(self, incoming=0):
        usage = sum(p.stat().st_size for p in self.data_dir.rglob("*") if p.is_file())
        if usage + incoming > self.settings.data_quota_gb * 1024 ** 3:
            raise VideoLensError("QUOTA_EXCEEDED", "数据目录已达到容量上限，请清理或调整容量设置", 507)
        if shutil.disk_usage(self.data_dir).free < incoming + 256 * 1024 ** 2:
            raise VideoLensError("DISK_FULL", "磁盘可用空间不足，请清理后重试", 507)

    def import_file(self, path: Path, subtitle: Path | None = None):
        path = Path(path).resolve()
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".mkv", ".avi", ".mov"}:
            raise VideoLensError("INVALID_MEDIA", "请选择存在的 MP4、MKV、AVI 或 MOV 文件")
        if path.stat().st_size > self.settings.max_video_bytes:
            raise VideoLensError("MEDIA_TOO_LARGE", "视频文件超过大小限制", 413)
        if subtitle is None:
            candidates = [p for p in path.parent.glob(path.stem + ".*") if p.suffix.lower() in {".srt", ".vtt", ".ass"} and p.is_file()]
            candidates.sort(key=lambda p: (self.settings.subtitle_language not in p.name.lower(), len(p.name), p.name))
            subtitle = candidates[0] if candidates else None
        if subtitle:
            subtitle = Path(subtitle).resolve()
            if not subtitle.is_file() or subtitle.suffix.lower() not in {".srt", ".vtt", ".ass"} or subtitle.stat().st_size > 20_000_000:
                raise VideoLensError("INVALID_SUBTITLE", "字幕不存在、格式不支持或超过 20MB")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        media_hash = digest.hexdigest()
        if subtitle:
            digest.update(subtitle.read_bytes())
        content_hash = digest.hexdigest()
        with self._guard:
            for value in self.store.list_videos():
                if value.get("_content_hash") == content_hash and Path(value.get("_path", "")).is_file():
                    return _public(value)
            self._check_capacity(path.stat().st_size)
            identity = uuid4().hex
            folder = self.data_dir / "videos" / identity / "media"
            folder.mkdir(parents=True)
            target = folder / ("source" + path.suffix.lower())
            try:
                with path.open("rb") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if subtitle:
                    shutil.copyfile(subtitle, target.with_suffix(subtitle.suffix.lower()))
                value = {"id": identity, "title": path.name, "source_type": "file", "status": "imported",
                         "created_at": time.time(), "sha256": media_hash, "warnings": [],
                         "_content_hash": content_hash, "_path": str(target), "_media_created_at": time.time()}
                self.store.save_video(value)
            except BaseException:
                self._remove_owned(folder.parent)
                raise
        return _public(value)

    def import_url(self, url: str):
        parsed = urlparse(url)
        if len(url) > 2048 or parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
            raise VideoLensError("INVALID_URL", "请输入合法的 HTTP/HTTPS 视频链接（最长 2048 字符）")
        if not self.settings.allow_video_download:
            raise VideoLensError("DOWNLOAD_DISABLED", "视频下载功能已关闭")
        identity = uuid4().hex
        value = {"id": identity, "title": f"来自 {parsed.hostname} 的视频", "source_type": "url", "status": "imported",
                 "created_at": time.time(), "warnings": [], "_url": url}
        self.store.save_video(value)
        return _public(value)

    def list_videos(self):
        return [_public(v) for v in self.store.list_videos()]

    def get_video(self, video_id):
        value = self._video(video_id)
        return {**_public(value), "clips": [{**c, "start_time": timestamp(c["start_ms"]), "end_time": timestamp(c["end_ms"])} for c in self.store.clips(video_id)],
                "frames": [public_frame(f) for f in self.store.frames(video_id)]}

    def create_task(self, video_id, kind="analyze"):
        video = self._video(video_id)
        if kind not in {"analyze", "reindex"}:
            raise VideoLensError("INVALID_TASK", "任务类型必须为 analyze 或 reindex")
        if kind == "reindex" and not self.store.clips(video_id):
            raise VideoLensError("NOT_ANALYZED", "请先完成视频分析")
        with self._guard:
            fingerprint = self.settings.fingerprint()
            for old in self.store.list_jobs():
                if old["video_id"] == video_id and old["status"] in ACTIVE | {"paused"}:
                    return _public(old)
                if kind == "analyze" and old["video_id"] == video_id and old["status"] == "succeeded" and old.get("fingerprint") == fingerprint and video.get("report"):
                    return {**_public(old), "reused": True}
            identity = uuid4().hex
            value = {"id": identity, "video_id": video_id, "kind": kind, "status": "queued", "stage": "queued", "progress": 0,
                     "created_at": time.time(), "updated_at": time.time(), "warnings": [], "fingerprint": fingerprint,
                     "_settings": self.settings.model_dump(), "timings": {}}
            self.store.save_job(value)
            self.store.event(identity, "status", _public(value))
            self._queue.put(identity)
            return _public(value)

    def get_task(self, identity):
        return _public(self._job(identity))

    def list_tasks(self):
        return [_public(j) for j in self.store.list_jobs()]

    def control_task(self, identity, action):
        with self._guard:
            job = self._job(identity)
            status = job["status"]
            if action == "pause" and status in {"queued", "running"}:
                status = "paused" if status == "queued" else "pause_requested"
            elif action == "cancel" and status in ACTIVE | {"paused"}:
                status = "cancelled" if status in {"queued", "paused"} else "cancel_requested"
            elif action == "resume" and status in {"paused", "failed"}:
                if any(other["id"] != identity and other["video_id"] == job["video_id"] and other["status"] in ACTIVE | {"paused"} for other in self.store.list_jobs()):
                    raise VideoLensError("TASK_ACTIVE", "该视频已有另一项任务，请先完成或取消", 409)
                status = "queued"
            else:
                raise VideoLensError("INVALID_STATE", "当前状态不能执行该操作", 409)
            result = self.store.update_job(identity, status=status, error=None)
            self.store.event(identity, "status", _public(result))
            if action == "resume":
                self._queue.put(identity)
            return _public(result)

    def task_events(self, identity, after=0):
        self._job(identity)
        return self.store.events(identity, after)

    def _check(self, identity):
        if self._stop.is_set():
            raise TaskInterrupted("shutdown")
        job = self._job(identity)
        if job["status"] in {"pause_requested", "paused"}:
            raise TaskInterrupted("pause")
        if job["status"] in {"cancel_requested", "cancelled"}:
            raise TaskInterrupted("cancel")

    def _worker(self):
        while not self._stop.is_set():
            try:
                identity = self._queue.get(timeout=1)
            except queue.Empty:
                if time.time() - self._last_cleanup > 3600:
                    try:
                        self.cleanup()
                    except Exception:
                        pass
                    self._last_cleanup = time.time()
                continue
            if identity is None:
                break
            try:
                with self._guard:
                    job = self.store.get_job(identity)
                    if not job or job["status"] != "queued":
                        continue
                    self.store.update_job(identity, status="running")
                self._run_task(identity)
            finally:
                self._queue.task_done()

    def _run_task(self, identity):
        job = self._job(identity)
        video = self._video(job["video_id"])
        settings = Settings(**job["_settings"])
        # 设置在提交时固定，模型清单则可能在暂停期间升级。版本不同就撤销旧
        # 检查点，安全重做；不能把旧向量或旧摘要混进新模型处理结果。
        current_fingerprint = settings.fingerprint()
        if job.get("fingerprint") != current_fingerprint:
            self.store.clear_checkpoints(identity, ["acquire", "probe", "transcript", "clips", "frames", "vision", "index", "summary"])
            self.store.update_job(identity, fingerprint=current_fingerprint)
        gateway = ModelGateway(settings, self.secrets.get)
        index = IndexManager(settings, self.store, gateway)
        composer = Composer(settings, gateway)
        base = self.data_dir / "videos" / video["id"]
        work = base / "tasks" / identity / "work"
        work.mkdir(parents=True, exist_ok=True)
        warnings = list(job.get("warnings", []))
        def check():
            self._check(identity)
        timings = job.get("timings", {})

        def stage(name, progress, function):
            check()
            old = self.store.get_checkpoint(identity, name)
            order = ["acquire", "probe", "transcript", "clips", "frames", "vision", "index", "summary"]
            invalid = old is not None and ((name == "acquire" and not Path(old["value"]).is_file()) or
                                         (name in {"frames", "vision"} and any(not Path(f["path"]).is_file() for f in old["value"])))
            if invalid:
                self.store.clear_checkpoints(identity, order[order.index(name):])
                old = None
            if old is not None:
                return old["value"]
            self.store.update_job(identity, stage=name, progress=progress)
            self.store.event(identity, "progress", {"stage": name, "progress": progress, "status": "running"})
            started = time.monotonic()
            value = function()
            check()
            timings[name] = round(time.monotonic() - started, 3)
            self.store.checkpoint(identity, name, {"value": value})
            self.store.update_job(identity, timings=timings, warnings=list(dict.fromkeys(warnings)))
            return value

        def acquire():
            while not self._inference.acquire(timeout=0.2):
                check()

        locked = False
        try:
            if job["kind"] == "reindex":
                acquire()
                locked = True
                warnings.extend(index.build(video["id"], self.store.clips(video["id"]), self.store.frames(video["id"]), check))
            else:
                def obtain():
                    if video.get("_path") and Path(video["_path"]).is_file():
                        return video["_path"]
                    if video.get("_url"):
                        self._check_capacity()
                        return str(media.download(video["_url"], base / "media", settings, check))
                    raise VideoLensError("MEDIA_EXPIRED", "播放副本已清理，请重新导入视频")

                path = Path(stage("acquire", 5, obtain))
                metadata = stage("probe", 10, lambda: media.probe(path, settings, check))
                video.update(metadata, _path=str(path), _media_created_at=video.get("_media_created_at", time.time()), status="analyzing")
                self.store.save_video(video)

                def transcribe():
                    try:
                        subtitles = media.find_subtitles(path, work, settings, check)
                    except media.MediaError:
                        subtitles = []
                        warnings.append("字幕解析失败，尝试语音转写")
                    if subtitles:
                        return subtitles
                    if not metadata["has_audio"]:
                        warnings.append("视频没有音轨，使用画面通道")
                        return []
                    audio = media.extract_audio(path, work / "audio.wav", settings, check)
                    acquired = False
                    try:
                        acquire()
                        acquired = True
                        return gateway.transcribe(audio, check)
                    except TaskInterrupted:
                        raise
                    except Exception:
                        warnings.append("语音转写不可用，继续处理画面；可导入字幕后重试")
                        return []
                    finally:
                        gateway.release()
                        if acquired:
                            self._inference.release()

                segments = stage("transcript", 20, transcribe)
                video["transcript"] = segments
                clips = stage("clips", 35, lambda: media.make_clips(segments, metadata["duration_ms"], settings))
                frames = stage("frames", 45, lambda: media.extract_frames(path, clips, base / "tasks" / identity / "frames", settings, check))
                if metadata["height"] < 720:
                    warnings.append("源视频低于 720p，关键帧保留真实分辨率")
                if any(f["size_bytes"] > settings.frame_max_bytes for f in frames):
                    warnings.append("部分关键帧为保留可读性超过图片大小目标")

                acquire()
                locked = True

                def describe():
                    try:
                        return gateway.describe(frames, check)
                    except TaskInterrupted:
                        raise
                    except Exception:
                        warnings.append("视觉模型未配置或不可用，保留关键帧并采用文本通道")
                        return frames

                frames = stage("vision", 55, describe)
                def prepare_index():
                    notes = index.build(video["id"], clips, frames, check, activate=False)
                    return {"warnings": notes, "meta": index.pending_meta}

                prepared = stage("index", 70, prepare_index)
                warnings.extend(prepared["warnings"])
                gateway.release()
                report = stage("summary", 90, lambda: composer.report(video["title"], clips, frames, check))
                if report["mode"] != "generated":
                    warnings.append("总结采用基础证据摘录或证据不足提示")
                video.update(report=report, warnings=list(dict.fromkeys(warnings)), status="ready")
                self.store.commit_analysis(video, clips, frames, prepared["meta"])
                if index._client is not None or (self.data_dir / "chroma").exists():
                    try:
                        index.prune(video["id"], set((prepared["meta"] or {}).get("collections", {}).values()))
                    except Exception:
                        warnings.append("旧索引清理未完成，可通过重新构建或删除视频重试")
                atomic_json(base / "report.json", _public(report))
                (base / "report.md").write_text(report["markdown"], encoding="utf-8")
                self._remove_owned(work)
            result = self.store.update_job(identity, status="partial" if warnings else "succeeded", stage="completed", progress=100,
                                           warnings=list(dict.fromkeys(warnings)), timings=timings)
            self.store.event(identity, "complete", _public(result))
        except TaskInterrupted as exc:
            status = {"pause": "paused", "cancel": "cancelled", "shutdown": "queued"}[exc.action]
            result = self.store.update_job(identity, status=status, warnings=list(dict.fromkeys(warnings)), timings=timings)
            self.store.event(identity, "status", _public(result))
            if exc.action == "cancel":
                prepared = self.store.get_checkpoint(identity, "index")
                for name in (prepared or {}).get("value", {}).get("meta", {}).get("collections", {}).values() if (prepared or {}).get("value", {}).get("meta") else []:
                    try:
                        index.client().delete_collection(name)
                    except Exception:
                        pass
                self._remove_owned(base / "tasks" / identity)
        except Exception as exc:
            error = {"code": exc.code, "message": exc.message} if isinstance(exc, VideoLensError) else {"code": "ANALYSIS_FAILED", "message": "分析失败，请检查媒体格式、工具依赖与模型诊断后重试"}
            result = self.store.update_job(identity, status="failed", error=error, warnings=list(dict.fromkeys(warnings)), timings=timings)
            self.store.event(identity, "error", _public(result))
        finally:
            gateway.release()
            if locked:
                self._inference.release()

    def search(self, video_id, query="", image_path=None):
        self._video(video_id)
        if len(query) > 500 or (not query.strip() and image_path is None):
            raise VideoLensError("INVALID_QUERY", "请输入 1—500 字的问题，或提供检索图片")
        with self._inference:
            return self.index.search(video_id, query, image_path)

    def create_session(self, video_id):
        self._video(video_id)
        return self.store.create_session(video_id)

    def list_sessions(self, video_id):
        self._video(video_id)
        return self.store.list_sessions(video_id)

    def session_messages(self, session_id):
        if not self.store.get_session(session_id):
            raise VideoLensError("SESSION_NOT_FOUND", "会话不存在", 404)
        return self.store.messages(session_id)

    def _conversation(self, session_id, question):
        if not question.strip() or len(question) > 500:
            raise VideoLensError("INVALID_QUESTION", "问题长度必须为 1—500 个字符")
        session = self.store.get_session(session_id)
        if not session:
            raise VideoLensError("SESSION_NOT_FOUND", "会话不存在", 404)
        history = self.store.messages(session_id)
        recent = history[-self.settings.recent_messages:]
        if session.get("summary"):
            recent = [{"role": "system", "content": "历史摘要：" + session["summary"]}, *recent]
        hits = self.search(session["video_id"], question)
        refs = [ref for hit in hits for ref in hit["refs"]]
        # 检索先确定相关时间段，再引入相应章节摘要；摘要的每条结论仍需携带
        # 它引用的原文，由 Composer 在同一上下文预算内成对选择，不能仅凭摘要回答。
        video = self._video(session["video_id"])
        chapters = video.get("report", {}).get("chapters", [])
        related = []
        by_id = {ref["id"]: ref for ref in refs}
        for index, chapter in enumerate(chapters):
            end_ms = chapters[index + 1]["start_ms"] if index + 1 < len(chapters) else video.get("duration_ms", 2**63)
            if any(chapter["start_ms"] <= ref["start_ms"] < end_ms for ref in refs):
                # 少量章节结论提供背景，保留预算给本轮问题的切片证据和近期对话。
                selected = {**chapter, "claims": chapter.get("claims", [])[:2]}
                related.append(selected)
                for claim in selected["claims"]:
                    for ref in claim["refs"]:
                        by_id.setdefault(ref["id"], ref)
                if len(related) >= 3:
                    break
        return session, history, recent, list(by_id.values()), related

    def _save_answer(self, session, question, answer, history):
        self.store.add_message(session["id"], {"role": "user", "content": question})
        self.store.add_message(session["id"], {"role": "assistant", "content": answer["answer"], "refs": answer["refs"], "mode": answer["mode"]})
        if len(history) > self.settings.recent_messages:
            # 离线可用的提取式历史摘要；保留用户问题与已核验回答，避免引入新事实。
            older = history[:-self.settings.recent_messages]
            session["summary"] = "\n".join(f"{item['role']}：{item['content'][:160]}" for item in older[-12:])[:1800]
            self.store.update_session(session)

    def ask(self, session_id, question):
        lock = self._sessions.setdefault(session_id, threading.Lock())
        with lock, self._inference:
            session, history, recent, refs, chapters = self._conversation(session_id, question)
            def supplement():
                return [r for c in self.store.clips(session["video_id"]) for r in clip_refs(c, self.store.frames(session["video_id"]))]
            try:
                answer = self.composer.compose(question, refs, recent, self.settings.answer_prompt, lambda: None, supplement, chapters=chapters)
                self._save_answer(session, question, answer, history)
                return answer
            finally:
                self.gateway.release()

    def ask_stream(self, session_id, question):
        """delta 来自真实模型流；最终结果经过引用校验，失败不会伪装成生成成功。"""
        import re

        lock = self._sessions.setdefault(session_id, threading.Lock())
        with lock, self._inference:
            session, history, recent, refs, chapters = self._conversation(session_id, question)
            yield {"event": "status", "data": {"message": "已检索证据，正在生成；流式草稿的引用将在完成时检查"}}
            raw, displayed = "", ""
            started = time.monotonic()
            first_token = None
            messages, selected = self.composer._messages(question, refs, recent, self.settings.answer_prompt, chapters)
            try:
                if not selected:
                    answer = {"answer": "当前视频中没有检索到足以回答这个问题的证据。", "claims": [], "refs": [], "mode": "insufficient"}
                else:
                    try:
                        for delta in self.gateway.stream(messages, lambda: None):
                            if first_token is None:
                                first_token = time.monotonic() - started
                            raw += delta
                            # 只展示 JSON 中结论 text 的可读部分；最终结果统一替换草稿。
                            parts = re.findall(r'"text"\s*:\s*"((?:\\.|[^"\\])*)', raw)
                            draft = "\n\n".join(parts).replace("\\n", "\n").replace('\\"', '"')
                            if draft.startswith(displayed) and len(draft) > len(displayed):
                                yield {"event": "delta", "data": {"text": draft[len(displayed):], "provisional": True}}
                                displayed = draft
                        claims = validate_claims(parse_object(raw), selected)
                        used = {r["id"]: r for claim in claims for r in claim["refs"]}
                        answer = {"answer": "\n\n".join(c["text"] + " " + " ".join(f"[{r['start_time']} · {r['id']}]" for r in c["refs"]) for c in claims),
                                  "claims": claims, "refs": list(used.values()), "mode": "generated"}
                    except Exception:
                        yield {"event": "status", "data": {"message": "生成或引用检查未通过，正在修正；最终结果将替换草稿"}}
                        # 流式首次生成已占用一轮，只允许再修正一次；扩展相邻切片证据。
                        selected_ids = {r["clip_id"] for r in refs}
                        clips = self.store.clips(session["video_id"])
                        frames = self.store.frames(session["video_id"])
                        expanded = {r["id"]: r for r in refs}
                        for offset, clip in enumerate(clips):
                            if clip["id"] in selected_ids:
                                for neighbor in clips[max(0, offset - 1):offset + 2]:
                                    expanded.update({r["id"]: r for r in clip_refs(neighbor, frames)})
                        answer = self.composer.compose(question, list(expanded.values()), recent, self.settings.answer_prompt, lambda: None, attempts=1, chapters=chapters)
                answer["timings"] = {"first_model_token_seconds": first_token, "total_seconds": round(time.monotonic() - started, 3)}
                self._save_answer(session, question, answer, history)
                yield {"event": "final", "data": answer}
            finally:
                self.gateway.release()

    def export(self, video_id, fmt):
        video = self.get_video(video_id)
        report = video.get("report")
        if not report:
            raise VideoLensError("NOT_ANALYZED", "视频尚无分析报告", 409)
        if fmt in {"markdown", "md"}:
            return report["markdown"]
        if fmt == "json":
            return json.dumps(video, ensure_ascii=False, indent=2)
        raise VideoLensError("INVALID_FORMAT", "导出格式仅支持 markdown 或 json")

    def media_path(self, video_id, frame_id=None):
        video = self._video(video_id)
        if frame_id:
            item = next((f for f in self.store.frames(video_id) if f["id"] == frame_id), None)
            path = Path(item["path"]) if item else None
        else:
            path = Path(video["_path"]) if video.get("_path") else None
        if not path or not path.is_file() or not path.resolve().is_relative_to(self.data_dir):
            raise VideoLensError("MEDIA_NOT_AVAILABLE", "媒体不存在或已按保留策略清理", 404)
        return path

    def _remove_owned(self, path):
        path = Path(path).resolve()
        if path == self.data_dir or not path.is_relative_to(self.data_dir):
            raise VideoLensError("UNSAFE_PATH", "拒绝清理数据目录之外的路径")
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    def delete_video(self, video_id):
        self._video(video_id)
        with self._guard, self._inference:
            if any(j["video_id"] == video_id and j["status"] in ACTIVE for j in self.store.list_jobs()):
                raise VideoLensError("TASK_ACTIVE", "请先取消该视频正在执行的任务", 409)
            try:
                self.index.delete(video_id)
            except Exception as exc:
                raise VideoLensError("INDEX_DELETE_FAILED", "向量库清理失败，请修复向量库后重试；原始记录已保留", 503) from exc
            self._remove_owned(self.data_dir / "videos" / video_id)
            self.store.delete_video(video_id)
        return {"deleted": video_id}

    def cleanup(self):
        now = time.time()
        cleaned = {"media": 0, "temporary": 0, "videos": 0}
        jobs = self.store.list_jobs()
        for video in self.store.list_videos():
            video_jobs = [j for j in jobs if j["video_id"] == video["id"]]
            if any(j["status"] in ACTIVE | {"paused"} for j in video_jobs):
                continue
            if self.settings.result_ttl_days and now - video["created_at"] > self.settings.result_ttl_days * 86400:
                self.delete_video(video["id"])
                cleaned["videos"] += 1
                continue
            if video.get("report") and video.get("_path") and now - video.get("_media_created_at", now) > self.settings.managed_media_days * 86400:
                self._remove_owned(Path(video["_path"]).parent)
                video.pop("_path", None)
                video["media_expired"] = True
                self.store.save_video(video)
                cleaned["media"] += 1
            for job in video_jobs:
                task_folder = self.data_dir / "videos" / video["id"] / "tasks" / job["id"]
                work = task_folder / "work"
                if work.is_dir() and now - job["updated_at"] > self.settings.failed_temp_hours * 3600:
                    self._remove_owned(work)
                    cleaned["temporary"] += 1
                # 重新分析成功后，旧任务的帧已不是当前证据；失败任务的帧同样属于
                # 可重做中间文件。只保留数据库正在引用的那一代，避免每次重试累积图片。
                # 暂停及运行视频已在上方跳过；以后恢复失败任务时，缺帧检查点会自动重做。
                referenced = any(Path(frame["path"]).resolve().is_relative_to(task_folder.resolve())
                                 for frame in self.store.frames(video["id"]))
                if task_folder.is_dir() and not referenced and now - job["updated_at"] > self.settings.failed_temp_hours * 3600:
                    self._remove_owned(task_folder)
                    cleaned["temporary"] += 1
        return cleaned

    def public_settings(self):
        data = self.settings.model_dump(exclude={"config_path"})
        # 默认路径以相对路径展示，避免公开响应泄露本机用户名和内部绝对路径。
        root = Path(self.settings.config_path).parent if self.settings.config_path else self.data_dir.parent
        for key in ("data_dir", "models_dir", "runtime_dir"):
            path = Path(data[key])
            if path.is_relative_to(root):
                data[key] = str(path.relative_to(root))
            elif path.is_absolute():
                data[key] = "（已配置外部目录，请在本机配置文件修改）"
        for key in ("ffmpeg", "ffprobe", "llama_server_path", "llama_model_path"):
            if data.get(key) and Path(data[key]).is_absolute():
                data[key] = Path(data[key]).name
        return data

    def update_settings(self, patch):
        if any(k in patch for k in ("api_token", "ui_password", "config_path")):
            raise VideoLensError("READONLY_SETTING", "不能通过普通设置修改访问凭据")
        with self._guard, self._inference:
            if any(j["status"] in ACTIVE for j in self.store.list_jobs()):
                raise VideoLensError("TASK_ACTIVE", "修改设置前请暂停或等待当前任务完成", 409)
            current = self.settings.model_dump()
            for key, value in patch.items():
                current[key] = {**current[key], **value} if isinstance(value, dict) and isinstance(current.get(key), dict) else value
            try:
                updated = Settings(**current)
            except ValidationError as exc:
                raise VideoLensError("INVALID_SETTINGS", "配置字段或数值无效，请检查切片范围、路径及模型设置") from exc
            path = Path(self.settings.config_path or self.data_dir.parent / "config.local.json")
            atomic_json(path, updated.model_dump(exclude={"config_path"}))
            restart_fields = {"host", "port", "data_dir", "models_dir", "runtime_dir", "analysis_concurrency"}
            restart = any(getattr(updated, key) != getattr(self.settings, key) for key in restart_fields)
            for key in Settings.model_fields:
                if key not in restart_fields | {"api_token", "ui_password", "config_path"}:
                    setattr(self.settings, key, getattr(updated, key))
            self.gateway.release()
            return {"settings": self.public_settings(), "restart_required": restart}

    def store_key(self, ref, key):
        self.secrets.set(ref, key)

    def check_provider(self, kind):
        if kind not in {"text", "vision", "local"}:
            raise VideoLensError("INVALID_PROVIDER", "未知模型服务类型")
        import httpx

        if kind == "local":
            base, _, key, timeout = self.gateway._local_provider()
        else:
            base, _, key, timeout = self.gateway._provider(kind)
        try:
            with httpx.Client(timeout=min(timeout, 15), trust_env=False) as client:
                response = client.get(base.rstrip("/") + "/models", headers={"Authorization": "Bearer " + key} if key else {})
                response.raise_for_status()
            return {"ok": True, "message": "模型列表接口可连接；生成及视觉能力仍需实际调用验证"}
        except Exception as exc:
            raise VideoLensError("PROVIDER_UNAVAILABLE", "模型服务连接失败，请检查地址、网络和密钥", 503) from exc

    def doctor(self):
        import platform
        import sys

        return {"python": platform.python_version(), "platform": platform.system(), "environment": Path(sys.prefix).name,
                "ffmpeg": bool(shutil.which(self.settings.ffmpeg) or Path(self.settings.ffmpeg).is_file()),
                "ffprobe": bool(shutil.which(self.settings.ffprobe) or Path(self.settings.ffprobe).is_file()),
                "cloud_enabled": self.settings.cloud_enabled, "models": self.gateway.health(),
                "database": "ok", "active_tasks": sum(j["status"] in ACTIVE for j in self.store.list_jobs())}
