"""偏好记忆的持久化、删除、并发保护、证据约束与真实流水线接入。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.config import load_app_config
from backend.memory.store import MemoryStore, MemoryError
from backend.memory.service import PreferenceMemory
from backend.pipeline.orchestrator import Orchestrator


class Encoder:
    text_dim=4

    def embed_query(self,text):
        return [1,1 if "步骤" in text else 0,1 if "硬件" in text else 0,0]

    def embed_texts(self,texts):
        return [self.embed_query(text) for text in texts]


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.cfg=load_app_config(root=ROOT,ignore_user_settings=True)
        self.encoder=Encoder()
        self.store=MemoryStore(self.root/"preferences.json",self.encoder,self.cfg.memory)
        self.repo=Mock()
        self.repo.list_videos.return_value=[SimpleNamespace(video_id="v1",title="硬件部署",display_name="硬件部署")]
        self.repo.get_video_report.return_value={"summary":"介绍本地 AI 的硬件需求与部署。"}
        self.repo.list_qa.return_value=[{"id":1,"question":"硬件需要什么条件？"}]
        self.llm=Mock()
        self.memory=PreferenceMemory(self.store,self.repo,self.llm,self.cfg.memory)

    def tearDown(self):
        self.memory.close()
        self.tmp.cleanup()

    def extracted(self,text="用户近期更关注本地 AI 部署的硬件需求。",ids=None,kind="interest"):
        self.llm.complete.return_value=SimpleNamespace(text=json.dumps({"memories":[{"text":text,"kind":kind,"evidence_ids":ids or ["video:v1","qa:1"]}]},ensure_ascii=False))

    def test_crud_survives_restart_and_text_vectors_are_deleted_together(self):
        item=self.store.put("用户希望回答优先给出可执行步骤。","style")
        raw=json.loads(self.store.path.read_text(encoding="utf-8"))
        self.assertEqual(len(raw["items"][0]["vector"]),4)
        self.assertNotIn("vector",self.store.public()["items"][0])
        reopened=MemoryStore(self.store.path,self.encoder,self.cfg.memory)
        self.assertEqual(reopened.search("问题")[0]["id"],item["id"])
        updated=self.store.put("用户更关注硬件需求。",item_id=item["id"])
        self.assertEqual(updated["origin"],"manual")
        self.store.delete(item["id"])
        raw=self.store.path.read_text(encoding="utf-8")
        self.assertNotIn("用户更关注硬件需求",raw)
        self.assertNotIn('"vector"',raw)
        self.assertEqual(MemoryStore(self.store.path,self.encoder,self.cfg.memory).public()["items"],[])

    def test_failed_embedding_or_write_keeps_existing_file(self):
        item=self.store.put("用户偏好解释硬件条件。")
        before=self.store.path.read_bytes()
        with patch.object(self.encoder,"embed_query",return_value=[float("nan")]*4):
            with self.assertRaises(MemoryError):self.store.put("修改记忆",item_id=item["id"])
        self.assertEqual(before,self.store.path.read_bytes())
        with patch("backend.memory.store.os.replace",side_effect=OSError("fixture")):
            with self.assertRaises(OSError):self.store.delete(item["id"])
        self.assertEqual(before,self.store.path.read_bytes())
        self.assertEqual(self.store.public()["items"][0]["id"],item["id"])

    def test_deleted_during_retrieval_is_not_sent_to_generation(self):
        item=self.store.put("用户偏好实际操作步骤。","style")
        def encode(text):
            self.store.delete(item["id"])
            return [1,0,0,0]
        with patch.object(self.encoder,"embed_query",side_effect=encode):
            self.assertEqual(self.store.search("问题"),[])

    def test_history_is_not_relearned_after_delete(self):
        self.extracted()
        self.assertEqual(self.memory.refresh_once()["added"],1)
        self.store.delete(self.store.public()["items"][0]["id"])
        self.assertEqual(self.memory.refresh_once()["added"],0)
        self.llm.complete.assert_called_once()
        self.assertEqual(self.store.public()["items"],[])

    def test_manual_change_discards_inflight_auto_result(self):
        snap=self.store.snapshot()
        item=self.store.put("用户希望回答包含具体步骤。","style")
        candidates=[{"text":"用户关注硬件需求。","kind":"interest","evidence":[]}]
        self.assertEqual(self.store.add_learned(candidates,["video:v1"],snap["revision"]),0)
        self.assertEqual([i["id"] for i in self.store.public()["items"]],[item["id"]])
        self.assertIn("video:v1",self.store.snapshot()["seen_sources"])

    def test_unknown_or_insufficient_evidence_cannot_create_memory(self):
        self.extracted(ids=["video:nonexistent"])
        self.assertEqual(self.memory.refresh_once()["added"],0)
        self.repo.list_qa.return_value=[{"id":2,"question":"这个视频讲什么？"}]
        self.extracted(ids=["qa:2"])
        self.assertEqual(self.memory.refresh_once()["added"],0)
        self.repo.list_qa.return_value=[{"id":3,"question":"我更关注硬件需求。"}]
        self.extracted(ids=["qa:3"])
        self.assertEqual(self.memory.refresh_once()["added"],1)

    def test_summary_and_chat_receive_preference_without_changing_evidence(self):
        paragraph="用户希望回答优先给出可执行步骤。"
        self.store.put(paragraph,"style")
        video=SimpleNamespace(title="本地部署",display_name="部署",duration_ms=1000)
        self.repo.find_video.return_value=video
        self.repo.iter_clip_payloads.return_value=[{"clip_id":"c1","text":"演示硬件安装与配置步骤，先检查显卡内存，再安装运行库，最后验证模型推理。"*5,"start_ms":0,"end_ms":1000}]
        self.repo.get_keyframes.return_value=[]
        self.llm.complete.return_value=SimpleNamespace(text="总结",backend="cloud",degraded=False)
        orch=Orchestrator(repository=self.repo,llm=self.llm,memory=self.memory,config=self.cfg)
        self.assertEqual(orch.summarize("v1"),"总结")
        messages=self.llm.complete.call_args.args[0]
        self.assertEqual(messages[0]["role"],"system")
        self.assertIn(paragraph,"\n".join(m["content"] for m in messages if m["role"]=="system"))
        self.assertIn("硬件安装",messages[-1]["content"])
        evidence=[{"role":"user","content":"本轮视频证据[1]"}]
        adapted=orch._preference_messages(evidence,"如何安装")
        self.assertEqual(adapted[-1],evidence[-1])
        self.assertIn("不是视频事实",adapted[0]["content"])
        self.store.options(enabled=False,auto_learn=False)
        self.assertEqual(orch._preference_messages(evidence,"如何安装"),evidence)

    def test_report_excludes_unrelated_preferences_even_when_marked_as_style(self):
        hardware="用户偏好 BIOS 设置、二手硬件组装与 AI 推理优化的操作步骤。"
        film="用户希望电影分析关注人物情感与剧情转折。"
        def vector(text):
            if "BIOS" in text:return [0.4,0.9165,0,0]
            return [1,0,0,0]
        with patch.object(self.encoder,"embed_query",side_effect=vector):
            self.store.put(hardware,"style")
            self.store.put(film,"interest")
            evidence=[{"role":"user","content":"三叶与泷交换身体，通过彗星事件与跨越时间的记忆相互寻找。"}]
            orch=Orchestrator(repository=self.repo,memory=self.memory,config=self.cfg)
            result=orch._preference_messages(evidence,"你的名字：人物情感与剧情",report=True)
            text="\n".join(m["content"] for m in result)
            self.assertIn(film,text)
            self.assertNotIn(hardware,text)
            self.assertEqual(result[-1],evidence[-1])
            self.assertIn("静默忽略",text)
            self.assertIn("不列出与本视频无关",text)
            # 普通问答仍允许用户通用表达风格；报告必须走更严格的相关性门槛。
            self.assertTrue(any(item["text"]==hardware for item in self.store.search("你的名字")))

    def test_report_guard_is_kept_when_memory_disabled_and_query_uses_only_evidence(self):
        from backend.pipeline.stages.reports import _preference_query
        query=_preference_query("你的名字",[{"text":"泷与三叶交换身体"}])
        self.assertNotIn("操作步骤",query)
        self.assertIn("泷与三叶",query)
        self.store.options(enabled=False,auto_learn=False)
        orch=Orchestrator(repository=self.repo,memory=self.memory,config=self.cfg)
        evidence=[{"role":"user","content":"片中证据"}]
        result=orch._preference_messages(evidence,query,report=True)
        self.assertEqual(len(result),2)
        self.assertIn("只能来自本视频",result[0]["content"])
        self.assertEqual(result[-1],evidence[-1])

    def test_invalid_cloud_output_is_visible_and_does_not_mark_sources_seen(self):
        self.llm.complete.return_value=SimpleNamespace(text="not json")
        self.memory.notify()
        self.memory.future.result(timeout=3)
        self.assertIn("未完成",self.memory.status()["last_error"])
        self.assertEqual(self.store.snapshot()["seen_sources"],[])

    def test_memory_web_crud_and_vector_not_exposed(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.web.routes.memory import create_router

        app=FastAPI()
        app.state.orchestrator=SimpleNamespace(_memory=self.memory)
        app.include_router(create_router(),prefix="/internal")
        with TestClient(app) as client:
            response=client.post("/internal/memory/items",json={"text":"用户偏好硬件需求说明。","kind":"interest"})
            self.assertEqual(response.status_code,200,response.text)
            item=response.json()
            self.assertNotIn("vector",item)
            self.assertEqual(client.put("/internal/memory/items/"+item["id"],json={"text":"用户希望看到可执行步骤。","kind":"style"}).status_code,200)
            self.assertEqual(client.get("/internal/memory").json()["items"][0]["kind"],"style")
            self.assertEqual(client.delete("/internal/memory/items/"+item["id"]).status_code,200)
            self.assertEqual(client.delete("/internal/memory/items/"+item["id"]).status_code,404)
            self.assertEqual(client.post("/internal/memory/items",json={"text":""}).status_code,422)

    def test_batch_delete_is_atomic_and_blocks_relearning(self):
        first = self.store.put("用户希望优先说明硬件需求。")
        second = self.store.put("用户偏好精简的步骤说明。")
        keep = self.store.put("用户希望电影分析关注剧情。")
        before = self.store.snapshot()
        raw = self.store.path.read_bytes()
        with self.assertRaises(KeyError):
            self.store.delete_many([first['id'], 'missing'])
        self.assertEqual(self.store.path.read_bytes(), raw)
        with patch.object(self.store, '_save', side_effect=OSError('磁盘写入失败')):
            with self.assertRaises(OSError):
                self.store.delete_many([first['id'], second['id']])
        self.assertEqual(self.store.snapshot(), before)
        self.assertEqual(self.store.path.read_bytes(), raw)
        removed = self.store.delete_many([first['id'], second['id'], first['id']])
        self.assertEqual(set(removed), {first['id'], second['id']})
        after = self.store.snapshot()
        self.assertEqual(after['revision'], before['revision'] + 1)
        self.assertEqual([i['id'] for i in after['items']], [keep['id']])
        from backend.memory.store import text_hash
        self.assertTrue({text_hash(first['text']), text_hash(second['text'])} <= set(after['blocked_hashes']))
        reloaded = MemoryStore(self.store.path, self.encoder, self.cfg.memory)
        self.assertEqual([i['id'] for i in reloaded.snapshot()['items']], [keep['id']])

    def test_batch_delete_http_validation_and_failure_keep_originals(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.web.routes.memory import create_router
        app = FastAPI(); app.state.orchestrator = SimpleNamespace(_memory=self.memory)
        app.include_router(create_router(), prefix='/internal')
        first = self.store.put('用户希望看到硬件对比。')
        second = self.store.put('用户希望简洁说明操作。')
        endpoint = '/internal/memory/items/delete-many'
        with TestClient(app) as client:
            for payload in ({'ids': []}, {'ids': [1]}, {'ids': ['']}, {'ids': [first['id']] * 201}):
                self.assertEqual(client.post(endpoint, json=payload).status_code, 422)
            self.assertEqual(client.post(endpoint, json={'ids': [first['id'], 'missing']}).status_code, 404)
            self.assertEqual(len(self.store.snapshot()['items']), 2)
            with patch.object(self.store, '_save', side_effect=OSError('写入失败')):
                self.assertEqual(client.post(endpoint, json={'ids': [first['id']]}).status_code, 500)
            self.assertEqual(len(self.store.snapshot()['items']), 2)
            result = client.post(endpoint, json={'ids': [first['id'], second['id'], first['id']]})
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.json()['deleted_count'], 2)
            self.assertEqual(self.store.snapshot()['items'], [])


if __name__=="__main__":
    unittest.main(verbosity=2)
