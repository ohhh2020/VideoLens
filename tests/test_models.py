"""models 层自检：prompts / vl / llm / embed。

纯 stdlib 断言，不依赖 pytest —— 答辩机不装环境也能跑：
    python tests\\t_models_llm.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.models.embed import (  # noqa: E402
    DEFAULT_IMAGE_DIM,
    DEFAULT_TEXT_DIM,
    EmbeddingError,
    EmbeddingService,
    HashingEmbedder,
    detect_device,
)
from core.models.llm import (  # noqa: E402
    KeywordBackend,
    LlmError,
    LlmGateway,
    LlmMessage,
    LlmResponse,
    OpenAICompatibleBackend,
    normalize_messages,
)
from core.models.vl import (  # noqa: E402
    VlError,
    VlResult,
    VlService,
    split_description,
)
from core.prompts import DEFAULT_PROMPTS, PromptRegistry  # noqa: E402

PASSED = 0
FAILED: list[str] = []


def check(name: str, fn) -> None:
    global PASSED
    try:
        fn()
    except AssertionError as exc:
        FAILED.append(f"{name}: {exc}")
        print(f"  FAIL {name} :: {exc}")
    except Exception as exc:  # noqa: BLE001
        FAILED.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"  ERR  {name} :: {type(exc).__name__}: {exc}")
    else:
        PASSED += 1
        print(f"  ok   {name}")


def expect_raises(exc_type, fn) -> None:
    try:
        fn()
    except exc_type:
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"期望 {exc_type.__name__}，实际 {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"期望抛 {exc_type.__name__}，但没有抛")


# =============================================================================
# prompts
# =============================================================================

print("\n[prompts]")

reg = PromptRegistry.load(root=ROOT)


def t_prompts_loaded() -> None:
    keys = reg.keys()
    assert "qa_answer" in keys, keys
    assert "keyframe_describe" in keys, keys
    assert len(reg) >= 7, len(reg)


check("prompts 从 prompts.yaml 加载", t_prompts_loaded)


def t_prompts_render() -> None:
    text = reg.render("qa_answer", context="证据A", query="切片多长", video_title="测试视频")
    assert "证据A" in text
    assert "切片多长" in text
    assert "测试视频" in text
    assert "{context}" not in text


check("render 填充占位符", t_prompts_render)


def t_prompts_missing_placeholder() -> None:
    text = reg.render("qa_answer", query="只给问题")
    assert "{context}" in text, "缺失占位符应原样保留而非空串"


check("render 缺失占位符原样保留", t_prompts_missing_placeholder)


def t_prompts_fallback() -> None:
    missing = PromptRegistry.load(ROOT / "configs" / "__nope__.yaml", use_defaults=True)
    assert len(missing) >= 1
    assert missing.get("qa_answer") == DEFAULT_PROMPTS["qa_answer"]


check("文件缺失回退内置默认", t_prompts_fallback)


def t_prompts_unknown_key() -> None:
    expect_raises(KeyError, lambda: reg.get("__no_such_prompt__"))


check("未知 key 抛 KeyError", t_prompts_unknown_key)


def t_prompts_to_dict() -> None:
    data = reg.to_dict()
    assert isinstance(data, dict) and data
    assert all(isinstance(v, str) for v in data.values())


check("to_dict 返回纯文本映射", t_prompts_to_dict)


# =============================================================================
# split_description
# =============================================================================

print("\n[vl.split_description]")


def t_split_marked() -> None:
    desc, ocr = split_description(
        "一名讲师站在白板前讲解架构。- 文字：知影 ZhiYing\n- 文字：MCP Server"
    )
    assert "讲师" in desc, desc
    assert "MCP Server" in ocr, ocr
    assert "知影" in ocr


check("识别文字标记并拆分", t_split_marked)


def t_split_plain() -> None:
    desc, ocr = split_description("画面是一张深色背景的终端截图，代码正在滚动。")
    assert desc.startswith("画面是"), desc
    assert ocr == "", ocr


check("无标记时全文当描述", t_split_plain)


def t_split_none_ocr() -> None:
    desc, ocr = split_description("一名男子在讲台上演讲。\n文字：无")
    assert "男子" in desc
    assert ocr == "", ocr


check("文字为「无」时置空", t_split_none_ocr)


def t_split_empty() -> None:
    assert split_description("") == ("", "")
    assert split_description("   \n ") == ("", "")


check("空输入返回空对", t_split_empty)


# =============================================================================
# VlService
# =============================================================================

print("\n[vl.VlService]")


class FakeRunner:
    name = "fake-vl"

    def __init__(self, *, fail: int = 0, content: str = "画面是彩条测试图。\n文字：1600x900") -> None:
        self.fail = fail
        self.content = content
        self.calls = 0

    def describe(self, image_path, prompt, *, max_tokens=512, temperature=0.1) -> str:
        self.calls += 1
        if self.calls <= self.fail:
            raise VlError("模拟推理失败")
        return self.content

    def describe_multi(self, image_paths, prompt, *, stamps=None, max_tokens=512,
                       temperature=0.1) -> str:
        """多图一次推理（跨帧连续叙述）。"""
        paths = list(image_paths)
        self.calls += 1
        self.last_multi = (len(paths), list(stamps or []))
        if self.calls <= self.fail:
            raise VlError("模拟推理失败")
        return self.content

    def close(self) -> None:
        return None


tmpdir = Path(tempfile.mkdtemp(prefix="zy-vl-"))
IMG = tmpdir / "frame.jpg"
try:
    from PIL import Image

    Image.new("RGB", (320, 180), (12, 34, 56)).save(IMG, format="JPEG")
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def t_vl_degraded() -> None:
    svc = VlService(None, degraded_reason="权重缺失")
    r = svc.describe(IMG)
    assert r.degraded is True
    assert r.ok is False
    assert "权重缺失" in r.error
    assert svc.available is False


check("无 runner 时降级不抛错", t_vl_degraded)


def t_vl_ok() -> None:
    runner = FakeRunner()
    svc = VlService(runner, prompt_key="keyframe_describe")
    r = svc.describe(tmpdir / "fake.jpg")
    assert r.ok is True, r.error
    assert "彩条" in r.description
    assert "1600x900" in r.ocr_text
    assert r.attempts == 1
    assert r.model == "fake-vl"


check("正常推理拆分描述/OCR", t_vl_ok)


def t_vl_retry() -> None:
    runner = FakeRunner(fail=2)
    svc = VlService(runner, max_retries=2)
    r = svc.describe(tmpdir / "fake.jpg")
    assert r.ok is True, r.error
    assert r.attempts == 3, r.attempts
    assert runner.calls == 3


check("失败重试后成功（attempts=3）", t_vl_retry)


def t_vl_retry_exhausted() -> None:
    runner = FakeRunner(fail=99)
    svc = VlService(runner, max_retries=2)
    r = svc.describe(tmpdir / "fake.jpg")
    assert r.degraded is True
    assert r.attempts == 3
    assert "模拟推理失败" in r.error


check("重试耗尽后降级", t_vl_retry_exhausted)


def t_vl_empty_content() -> None:
    svc = VlService(FakeRunner(content="   "), max_retries=0)
    r = svc.describe(tmpdir / "fake.jpg")
    assert r.degraded is True
    assert "空内容" in r.error


check("空内容视为失败", t_vl_empty_content)


def t_vl_describe_many() -> None:
    svc = VlService(FakeRunner(), max_retries=0)
    results = svc.describe_many([tmpdir / "a.jpg"])
    assert len(results) == 1
    assert results[0].ok


check("describe_many 单帧不合并", t_vl_describe_many)


def t_vl_result_dict() -> None:
    r = VlResult(description="描述", ocr_text="文字", model="m")
    data = r.to_dict()
    assert data["ok"] is True
    assert data["description"] == "描述"
    assert "[画面文字] 文字" in r.combined_text()
    assert r.has_ocr is True


check("VlResult.to_dict / combined_text", t_vl_result_dict)


def t_vl_info() -> None:
    svc = VlService(FakeRunner(), prompt_key="keyframe_describe", max_retries=1)
    info = svc.info()
    assert info["available"] is True
    assert info["model"] == "fake-vl"
    assert info["max_retries"] == 1


check("info() 字段完整", t_vl_info)


def t_vl_render_fallback() -> None:
    svc = VlService(FakeRunner(), prompt_key="__no_such_prompt__")
    text = svc._render("__no_such_prompt__")  # noqa: SLF001
    assert "描述" in text, text


check("未知提示词 key 用内置兜底", t_vl_render_fallback)


# =============================================================================
# llm
# =============================================================================

print("\n[llm]")


def t_message_validate() -> None:
    expect_raises(LlmError, lambda: LlmMessage("tool", "x"))
    assert LlmMessage("user", "hi").to_dict() == {"role": "user", "content": "hi"}


check("LlmMessage 校验 role", t_message_validate)


def t_normalize() -> None:
    msgs = normalize_messages(["纯字符串", {"role": "system", "content": "系统"}])
    assert [m.role for m in msgs] == ["user", "system"]


check("normalize_messages 支持 str/dict", t_normalize)


def t_normalize_parts() -> None:
    msgs = normalize_messages(
        [{"role": "user", "content": [{"type": "text", "text": "片段1"}, {"type": "text", "text": "片段2"}]}]
    )
    assert msgs[0].content == "片段1片段2", msgs[0].content


check("多段 content 拼接", t_normalize_parts)


def t_normalize_empty() -> None:
    expect_raises(LlmError, lambda: normalize_messages([]))


check("空消息列表抛错", t_normalize_empty)


def t_keyword_backend() -> None:
    kb = KeywordBackend()
    r = kb.complete([LlmMessage("user", "视频讲了切片长度默认十到三十秒，关键词是分块。")])
    assert r.degraded is True
    assert r.backend == "keywords"
    assert "内容概览" in r.text, r.text[:80]
    assert r.ok


check("KeywordBackend 永不失败", t_keyword_backend)


def t_keyword_extract() -> None:
    kb = KeywordBackend()
    words = kb.extract_keywords("视频智能分析助手使用 faster-whisper 做语音转写，用 ChromaDB 做向量检索。")
    assert words, words
    assert all(isinstance(w, str) for w in words)


check("extract_keywords 返回非空", t_keyword_extract)


class BoomBackend:
    name = "boom"
    model = "boom-1"

    def available(self) -> bool:
        return True

    def complete(self, messages, **kwargs) -> LlmResponse:
        raise LlmError("模拟后端故障")


class OkBackend:
    name = "ok"
    model = "ok-1"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def available(self) -> bool:
        return True

    def complete(self, messages, **kwargs) -> LlmResponse:
        self.seen = [m.role for m in messages]
        return LlmResponse(text="来自 ok 后端", backend=self.name, model=self.model)


def t_gateway_fallback() -> None:
    good = OkBackend()
    gw = LlmGateway([BoomBackend(), good, KeywordBackend()])
    r = gw.complete([LlmMessage("user", "问题")])
    assert r.text == "来自 ok 后端"
    assert r.degraded is False
    assert any("boom" in a for a in r.attempts), r.attempts
    assert good.seen == ["user"]


check("后端失败自动降级到下一个", t_gateway_fallback)


def t_gateway_all_fail_strict() -> None:
    gw = LlmGateway([BoomBackend(), KeywordBackend()])
    expect_raises(LlmError, lambda: gw.complete([LlmMessage("user", "问题")], require_quality=True))


check("require_quality 时不允许落到关键词", t_gateway_all_fail_strict)


def t_gateway_all_fail_soft() -> None:
    gw = LlmGateway([BoomBackend(), KeywordBackend()])
    r = gw.complete([LlmMessage("user", "视频讲了什么")])
    assert r.degraded is True
    assert r.backend == "keywords"
    assert r.ok


check("全失败时关键词兜底", t_gateway_all_fail_soft)


def t_gateway_names() -> None:
    gw = LlmGateway([OkBackend(), KeywordBackend()])
    assert gw.backend_names == ["ok", "keywords"]
    assert gw.primary == "ok"
    assert gw.info()["mode"] == "auto"


check("backend_names / primary / info", t_gateway_names)


def t_gateway_stream_fallback() -> None:
    gw = LlmGateway([OkBackend()])
    chunks = list(gw.stream([LlmMessage("user", "问题")]))
    assert chunks and chunks[0] == "来自 ok 后端", chunks


check("非流式后端的 stream 退化为整段", t_gateway_stream_fallback)


def t_openai_endpoint() -> None:
    b = OpenAICompatibleBackend("https://api.example.com/v1", "m")
    assert b.endpoint == "https://api.example.com/v1/chat/completions"
    b2 = OpenAICompatibleBackend("https://api.example.com/v1/chat/completions", "m")
    assert b2.endpoint == "https://api.example.com/v1/chat/completions"


check("endpoint 不重复拼接", t_openai_endpoint)


def t_openai_parse() -> None:
    b = OpenAICompatibleBackend("http://x/v1", "m")
    r = b._parse(  # noqa: SLF001
        {
            "model": "m-2024",
            "choices": [{"message": {"content": " 你好 "}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        },
        0.0,
    )
    assert r.text == "你好"
    assert r.total_tokens == 10
    assert r.finish_reason == "stop"
    expect_raises(LlmError, lambda: b._parse({"choices": []}, 0.0))  # noqa: SLF001


check("_parse 解析 usage 与异常结构", t_openai_parse)


def t_openai_unavailable() -> None:
    b = OpenAICompatibleBackend("", "m")
    assert b.available() is False
    expect_raises(LlmError, lambda: b.complete([LlmMessage("user", "x")]))


check("未配置 base_url 时拒绝调用", t_openai_unavailable)


# =============================================================================
# embed
# =============================================================================

print("\n[embed]")


def t_hash_deterministic() -> None:
    h = HashingEmbedder()
    a = h.embed_texts(["切片长度是怎么定的"])[0]
    b = h.embed_texts(["切片长度是怎么定的"])[0]
    assert len(a) == DEFAULT_TEXT_DIM
    assert a == b, "哈希向量必须确定性"


check("哈希文本向量确定性且维度正确", t_hash_deterministic)


def t_hash_normalized() -> None:
    h = HashingEmbedder()
    v = h.embed_texts(["知影 视频 分析"])[0]
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-6, norm


check("哈希向量 L2 归一化", t_hash_normalized)


def t_service_query_dim() -> None:
    svc = EmbeddingService()
    assert svc.degraded is True
    v = svc.embed_query("问题")
    assert len(v) == DEFAULT_TEXT_DIM, len(v)
    assert svc.text_dim == DEFAULT_TEXT_DIM


check("EmbeddingService 默认降级且维度正确", t_service_query_dim)


def t_service_cross_modal() -> None:
    svc = EmbeddingService()
    v = svc.embed_cross_modal("架构图里的融合节点")
    assert len(v) == DEFAULT_IMAGE_DIM, len(v)
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-6


check("跨模态文本编码到图像维度", t_service_cross_modal)


def t_service_images() -> None:
    if not HAS_PIL:
        print("  skip 无 Pillow")
        return
    svc = EmbeddingService()
    vectors = svc.embed_images([IMG])
    assert len(vectors) == 1
    assert len(vectors[0]) == DEFAULT_IMAGE_DIM, len(vectors[0])


check("图像编码返回图像维度向量", t_service_images)


def t_service_custom_dim() -> None:
    svc = EmbeddingService(HashingEmbedder(text_dim=128, image_dim=64))
    assert len(svc.embed_query("x")) == 128
    assert len(svc.embed_cross_modal("x")) == 64


check("自定义维度生效", t_service_custom_dim)


def t_service_batch_consistency() -> None:
    svc = EmbeddingService()
    single = svc.embed_query("切片长度")
    batch = svc.embed_texts(["切片长度", "另一段"])[0]
    assert single == batch, "单条与批量编码必须一致"


check("单条与批量编码一致", t_service_batch_consistency)


def t_service_empty() -> None:
    svc = EmbeddingService()
    assert svc.embed_texts([]) == []
    assert svc.embed_images([]) == []


check("空输入返回空列表", t_service_empty)


def t_service_dim_mismatch() -> None:
    class BadEmbedder:
        name = "bad"
        dim = 16

        def embed_texts(self, texts):
            return [[0.0] * 8 for _ in texts]

    svc = EmbeddingService(BadEmbedder(), HashingEmbedder())
    expect_raises(EmbeddingError, lambda: svc.embed_texts(["x"]))


check("维度不一致时报错而不是静默", t_service_dim_mismatch)


def t_service_from_config_no_manager() -> None:
    svc = EmbeddingService.from_config()
    assert svc.degraded is True
    assert "ModelManager" in svc.degraded_reason
    info = svc.info()
    assert info["text_dim"] == DEFAULT_TEXT_DIM
    assert "hashing" in info["text_model"] or info["text_model"] == "hashing"


check("from_config 无 manager 时降级", t_service_from_config_no_manager)


def t_service_from_config_strict() -> None:
    expect_raises(EmbeddingError, lambda: EmbeddingService.from_config(allow_degraded=False))


check("allow_degraded=False 时抛错", t_service_from_config_strict)


def t_detect_device() -> None:
    assert detect_device("cpu") == "cpu"
    assert detect_device("auto") in {"cpu", "cuda", "mps"}


check("detect_device 返回合法设备", t_detect_device)


def t_service_truncate() -> None:
    svc = EmbeddingService(max_text_chars=4)
    assert svc._truncate("一二三四五六") == "一二三四"  # noqa: SLF001


check("文本按 max_text_chars 截断", t_service_truncate)


def t_hash_cross_texts() -> None:
    h = HashingEmbedder(text_dim=32, image_dim=8)
    vectors = h.embed_cross_texts(["问题"])
    assert len(vectors[0]) == 8, len(vectors[0])


check("HashingEmbedder.embed_cross_texts 走图像维度", t_hash_cross_texts)


# =============================================================================
# D1：ASR 转写级 CPU 回退（cuBLAS 惰性加载，加载期探测不到，转写才炸）
# =============================================================================

print("\n[asr-d1] 转写级 CPU 回退")

from types import SimpleNamespace as _SN  # noqa: E402

from core.models.asr import AsrConfig, AsrEngine  # noqa: E402


def t_asr_cpu_fallback() -> None:
    engine = AsrEngine(AsrConfig(device="cuda"))
    loads = {"n": 0}

    class _CudaModel:
        def transcribe(self, *a, **k):
            raise RuntimeError(
                "Library cublas64_12.dll is not found or cannot be loaded"
            )

    class _CpuModel:
        def transcribe(self, *a, **k):
            seg = _SN(text="CPU 转写成功的内容。", start=0.0, end=2.0, words=[])
            info = _SN(language="zh", language_probability=1.0, duration=2.0)
            return iter([seg]), info

    cuda_model, cpu_model = _CudaModel(), _CpuModel()

    def fake_load(*a, **k):
        loads["n"] += 1
        if engine._force_cpu:
            engine._model, engine._loaded_key = cpu_model, ("p", "cpu", "int8")
            return cpu_model
        engine._model, engine._loaded_key = cuda_model, ("p", "cuda", "float16")
        return cuda_model

    engine.load = fake_load  # type: ignore[method-assign]
    audio = Path(tmpdir) / "a.wav"
    audio.write_bytes(b"RIFF----")
    tr = engine._from_whisper(audio, video_id="vd1", duration_ms=2000, warnings=[])
    assert tr.segments and tr.segments[0].text.startswith("CPU 转写成功"), tr.to_dict()
    assert engine._force_cpu is True
    assert engine.active_device[0] == "cpu", engine.active_device
    assert any("回退" in w for w in tr.warnings), tr.warnings
    assert loads["n"] == 2, f"应加载两次（cuda→cpu），实际 {loads['n']}"


check("CUDA 转写炸→卸载重载 CPU 重试成功", t_asr_cpu_fallback)


def t_asr_force_cpu_plan() -> None:
    engine = AsrEngine(AsrConfig(device="cuda"))
    engine._force_cpu = True
    plan = engine._device_plan()
    assert plan and all(d == "cpu" for d, _ in plan), plan


check("force_cpu 后设备计划不再含 CUDA", t_asr_force_cpu_plan)


# =============================================================================
# 汇总
# =============================================================================

import shutil  # noqa: E402

shutil.rmtree(tmpdir, ignore_errors=True)

print("\n" + "=" * 60)
if FAILED:
    print(f"FAILED {len(FAILED)} / {PASSED + len(FAILED)}")
    for item in FAILED:
        print("  -", item)
    sys.exit(1)
print(f"ALL PASSED {PASSED} checks")
