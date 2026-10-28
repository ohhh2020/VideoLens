"""文字到 BGE-M3 向量的本地 JSON 映射，原子写入，不混用视频向量库。"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

KINDS = {"interest", "question", "style"}


class MemoryError(ValueError):
    pass


def text_hash(text):
    return hashlib.sha256("".join(text.split()).encode("utf-8")).hexdigest()


def normalized(vector, dim):
    if len(vector) != dim or any(isinstance(v,bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
        raise MemoryError("记忆向量维度或数值无效，禁止降级保存")
    norm = math.sqrt(sum(v*v for v in vector))
    if norm <= 0:
        raise MemoryError("记忆向量为空，禁止降级保存")
    return [float(v/norm) for v in vector]


def cosine(a, b):
    return sum(x*y for x,y in zip(a,b))


class MemoryStore:
    def __init__(self, path, embedder, config):
        self.path, self.embedder, self.config = Path(path), embedder, config
        self.dim = int(embedder.text_dim)
        self.lock = threading.RLock()
        self.data = {"version":1,"embedding_model":"bge-m3","dimension":self.dim,
                     "revision":0,"enabled":True,"auto_learn":True,
                     "items":[],"seen_sources":[],"blocked_hashes":[]}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
                if self.data["version"] != 1 or self.data["embedding_model"] != "bge-m3" or self.data["dimension"] != self.dim:
                    raise MemoryError("记忆文件模型或版本不匹配")
                if not isinstance(self.data["revision"],int) or any(not isinstance(self.data[k],bool) for k in ("enabled","auto_learn")):
                    raise MemoryError("记忆设置或版本号无效")
                for key in ("seen_sources","blocked_hashes"):
                    if not isinstance(self.data[key],list) or any(not isinstance(v,str) for v in self.data[key]):
                        raise MemoryError("记忆历史标识无效")
                ids = set()
                for item in self.data["items"]:
                    self.validate_text(item["text"])
                    item["vector"] = normalized(item["vector"], self.dim)
                    if item["id"] in ids or item["kind"] not in KINDS:
                        raise MemoryError("记忆文件记录无效")
                    ids.add(item["id"])
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise MemoryError(f"记忆文件无法加载，原文件已保留：{self.path}") from exc
        else:
            self._save(self.data)

    def validate_text(self, text):
        if not isinstance(text, str):
            raise MemoryError("记忆必须是一段文字")
        text = " ".join(text.split())
        if not text or len(text) > self.config.max_text_chars:
            raise MemoryError(f"每条记忆需要 1–{self.config.max_text_chars} 个字符")
        return text

    def _save(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=self.path.parent,prefix=".memory-",delete=False)
        try:
            with handle:
                json.dump(data,handle,ensure_ascii=False,indent=2,allow_nan=False)
                handle.write("\n")
            os.replace(handle.name,self.path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise

    def _commit(self, data):
        data["revision"] = self.data["revision"]+1
        self._save(data)
        self.data = data

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.data)

    def public(self):
        data = self.snapshot()
        data["items"] = [{k:v for k,v in item.items() if k!="vector"} for item in data["items"]]
        for key in ("seen_sources","blocked_hashes"):
            data.pop(key,None)
        data["file"] = str(self.path)
        data["max_text_chars"] = self.config.max_text_chars
        return data

    def put(self, text, kind="interest", *, item_id=None):
        text = self.validate_text(text)
        if kind not in KINDS:
            raise MemoryError("记忆类型无效")
        vector = normalized(self.embedder.embed_query(text), self.dim)
        with self.lock:
            data = self.snapshot()
            old = next((i for i in data["items"] if i["id"]==item_id),None)
            if item_id and old is None:
                raise KeyError(item_id)
            if not old and len(data["items"]) >= self.config.max_items:
                raise MemoryError("记忆库已达到容量上限，请先删除不需要的记忆")
            if any(i["id"]!=item_id and text_hash(i["text"])==text_hash(text) for i in data["items"]):
                raise MemoryError("相同的记忆已存在")
            now = datetime.now(timezone.utc).isoformat()
            item = {"id":item_id or "m"+uuid4().hex,"text":text,"vector":vector,
                    "kind":kind,"origin":"manual","evidence":old["evidence"] if old else [],
                    "created_at":old["created_at"] if old else now,"updated_at":now}
            if old:
                data["blocked_hashes"] = list(set(data["blocked_hashes"])|{text_hash(old["text"])})
                data["items"] = [item if i["id"]==item_id else i for i in data["items"]]
            else:
                data["items"].append(item)
            self._commit(data)
            return {k:v for k,v in item.items() if k!="vector"}

    def delete(self, item_id):
        with self.lock:
            data = self.snapshot()
            item = next((i for i in data["items"] if i["id"]==item_id),None)
            if item is None:
                raise KeyError(item_id)
            data["items"] = [i for i in data["items"] if i["id"]!=item_id]
            # 仅保留不可逆摘要和已读历史标识，不保留删掉的文字或向量。
            data["blocked_hashes"] = list(set(data["blocked_hashes"])|{text_hash(item["text"])})
            self._commit(data)

    def options(self, *, enabled, auto_learn):
        with self.lock:
            data = self.snapshot()
            data.update(enabled=bool(enabled),auto_learn=bool(auto_learn))
            self._commit(data)

    def add_learned(self, candidates, sources, expected_revision):
        texts = [self.validate_text(i["text"]) for i in candidates]
        vectors = self.embedder.embed_texts(texts) if texts else []
        if len(vectors) != len(texts):
            raise MemoryError("记忆向量数量与文字不匹配")
        vectors = [normalized(v,self.dim) for v in vectors]
        with self.lock:
            data = self.snapshot()
            if not data["auto_learn"]:
                return 0
            added = 0
            # 用户编辑/删除优先；丢弃在用户操作前开始归纳的整批结果。
            if data["revision"] == expected_revision:
                for candidate,text,vector in zip(candidates,texts,vectors):
                    if text_hash(text) in data["blocked_hashes"] or len(data["items"])>=self.config.max_items:
                        continue
                    if any(cosine(vector,i["vector"])>=self.config.dedupe_similarity for i in data["items"]):
                        continue
                    now = datetime.now(timezone.utc).isoformat()
                    data["items"].append({"id":"m"+uuid4().hex,"text":text,"vector":vector,
                        "kind":candidate["kind"],"origin":"history","evidence":candidate["evidence"],
                        "created_at":now,"updated_at":now})
                    added += 1
            data["seen_sources"] = list(set(data["seen_sources"])|set(sources))
            self._commit(data)
            return added

    def search(self, query, *, report=False):
        data = self.snapshot()
        if not data["enabled"] or not data["items"]:
            return []
        vector = normalized(self.embedder.embed_query(query[:2000]),self.dim)
        # 编码期间用户可能删除或修改记忆，只召回此刻仍有效的记录。
        data = self.snapshot()
        if not data["enabled"]:
            return []
        ranked = []
        threshold = self.config.report_min_similarity if report else self.config.min_similarity
        for item in data["items"]:
            score = cosine(vector,item["vector"])
            # 报告中的风格记忆也可能夹带硬件等话题，不能无条件放行。
            if (not report and item["kind"]=="style") or score>=threshold:
                ranked.append((item,score))
        ranked.sort(key=lambda pair:(pair[0]["kind"]=="style",pair[0]["origin"]=="manual",pair[1]),reverse=True)
        return [{"id":i["id"],"text":i["text"],"origin":i["origin"],"score":round(s,4)} for i,s in ranked[:self.config.top_k]]
