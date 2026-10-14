"""验证分层摘要真正进入请求，并且上下文预算和原始证据映射保持一致。"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from videolens.config import GENERATION_SAFETY_TOKENS, MIN_GENERATION_INPUT_TOKENS, Provider, Settings
from videolens.generation import Composer


def reference(index, text="原始视频证据"):
    return {"id": f"t:c{index:06d}", "clip_id": f"c{index:06d}", "text": text,
            "start_ms": index * 300000, "end_ms": index * 300000 + 20000,
            "start_time": "00:00:00.000", "end_time": "00:00:20.000", "source": "transcript"}


def wire_bytes(messages):
    return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def test_local_output_budget_accepts_exact_boundary_and_rejects_one_over():
    output = 1024 - GENERATION_SAFETY_TOKENS - MIN_GENERATION_INPUT_TOKENS
    assert Settings(local_context_tokens=1024, output_tokens=output).output_tokens == output
    with pytest.raises(ValidationError, match="本地生成模型上下文过小"):
        Settings(local_context_tokens=1024, output_tokens=output + 1)


@pytest.mark.parametrize("role", ["text", "vision"])
def test_each_enabled_cloud_provider_checks_output_budget(role):
    with pytest.raises(ValidationError, match="云端.*上下文过小"):
        Settings(local_enabled=False, cloud_enabled=True, output_tokens=512,
                 **{f"{role}_provider": Provider(enabled=True, context_tokens=1024)})


def test_disabled_contexts_do_not_restrict_enabled_cloud_model():
    cfg = Settings(local_enabled=False, local_context_tokens=1024, cloud_enabled=True,
                   text_provider=Provider(enabled=True, context_tokens=8192), output_tokens=1536)
    item = reference(1, "这是原始证据。" * 80)
    messages, selected = Composer(cfg, None)._messages("概括", [item], [], "")
    assert selected == [item]
    assert wire_bytes(messages) + cfg.output_tokens + GENERATION_SAFETY_TOKENS <= 8192
    # 云端总开关关闭时，尚未启用的云端配置也不能使纯本地配置验证失败。
    assert Settings(cloud_enabled=False, text_provider=Provider(enabled=True, context_tokens=1024))


def test_report_uses_generated_chapter_summaries_with_original_ids():
    class Gateway:
        def __init__(self):
            self.payloads = []

        def generate(self, messages, check):
            check()
            payload = json.loads(messages[-1]["content"])
            self.payloads.append(payload)
            if payload["chapter_summaries"]:
                assert {row["text"] for row in payload["chapter_summaries"]} == {"第一章的归纳结果", "第二章的归纳结果"}
                ids = [row["source_refs"][0] for row in payload["chapter_summaries"]]
                return json.dumps({"claims": [{"text": "根据两章归纳得到综合总结", "refs": ids}]})
            source = payload["evidence"][0]
            text = "第一章的归纳结果" if source["id"] == "t:c000001" else "第二章的归纳结果"
            return json.dumps({"claims": [{"text": text, "refs": [source["id"]]}]})

    gateway = Gateway()
    clips = [{"id": "c000001", "start_ms": 0, "end_ms": 20000, "text": "第一章原文", "segment_ids": []},
             {"id": "c000002", "start_ms": 300000, "end_ms": 320000, "text": "第二章原文", "segment_ids": []}]
    report = Composer(Settings(), gateway).report("测试视频", clips, [], lambda: None)
    assert len(gateway.payloads) == 3
    assert "综合总结" in report["summary"]
    assert {item["id"] for item in report["refs"]} == {"t:c000001", "t:c000002"}
    assert all(chapter["claims"] for chapter in report["chapters"])
    assert not gateway.payloads[0]["chapter_summaries"]
    assert not gateway.payloads[1]["chapter_summaries"]


def test_many_chapters_are_bounded_and_every_summary_has_original_evidence():
    cfg = Settings(local_context_tokens=4096, output_tokens=512)
    refs = [reference(index, f"第{index}章证据：" + "具体事实。" * 10) for index in range(24)]
    chapters = [{"start_ms": ref["start_ms"], "claims": [{"text": "该章节的归纳。" * 100, "refs": [ref]}]} for ref in refs]
    messages, selected = Composer(cfg, None)._messages("总结整片", refs, [], "", chapters)
    payload = json.loads(messages[-1]["content"])
    assert wire_bytes(messages) + cfg.output_tokens + GENERATION_SAFETY_TOKENS <= cfg.local_context_tokens
    assert 1 < len(payload["chapter_summaries"]) < len(chapters)
    selected_ids = {ref["id"] for ref in selected}
    assert all(set(chapter["source_refs"]) <= selected_ids for chapter in payload["chapter_summaries"])
    # 首尾均覆盖，证明预算不足时不会仅取视频开头若干章节。
    assert chapters[0]["start_ms"] in {row["start_ms"] for row in payload["chapter_summaries"]}
    assert chapters[-1]["start_ms"] in {row["start_ms"] for row in payload["chapter_summaries"]}


def test_summary_with_missing_original_id_is_not_sent():
    ref = reference(1)
    chapter = {"start_ms": 0, "claims": [{"text": "无法追溯的摘要", "refs": ["missing-source"]}]}
    messages, selected = Composer(Settings(), None)._messages("总结", [ref], [], "", [chapter])
    assert selected == [ref]
    assert json.loads(messages[-1]["content"])["chapter_summaries"] == []


def test_oversized_fixed_prompt_never_reaches_model():
    class Gateway:
        def generate(self, *args):
            pytest.fail("超出窗口的请求不能发送给模型")

    answer = Composer(Settings(local_context_tokens=2048, output_tokens=256), Gateway()).compose(
        "问题", [reference(1)], [], "很长的用户自定义提示词" * 1000, lambda: None)
    assert answer["mode"] == "extractive"


def test_repair_rebuilds_budgeted_context_and_keeps_chapter_mapping():
    cfg = Settings(local_context_tokens=4096, output_tokens=512)
    ref = reference(1)
    chapter = {"start_ms": 0, "claims": [{"text": "已核验章节归纳", "refs": [ref]}]}

    class Gateway:
        def __init__(self):
            self.calls = 0

        def generate(self, messages, check):
            self.calls += 1
            assert wire_bytes(messages) + cfg.output_tokens + GENERATION_SAFETY_TOKENS <= cfg.local_context_tokens
            assert json.loads(messages[-1]["content"])["chapter_summaries"][0]["source_refs"] == [ref["id"]]
            if self.calls == 1:
                return "不是 JSON"
            return json.dumps({"claims": [{"text": "修正后的总结", "refs": [ref["id"]]}]})

    gateway = Gateway()
    answer = Composer(cfg, gateway).compose("总结", [ref], [], "", lambda: None, chapters=[chapter])
    assert gateway.calls == 2 and answer["mode"] == "generated"
