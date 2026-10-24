"""启动预检自检 —— preflight 曾因引用已删除的配置 schema（llm.backend /
ui.port）而整段静默失效，这是防复发的回归测试。

覆盖：
  1. verdict 分级（READY/DEGRADED/BROKEN）
  2. check_config 的 --config 显式路径（不得污染 ZHIYING_CONFIG）
  3. check_models 的 LLM 检查走 llm.mode（auto/cloud/local），不再读 llm.backend
  4. check_ports 读 gradio.port（旧 schema 是 ui.port）
  5. render / --json 输出可解析

跑法::

    python tests/test_preflight.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORK = ROOT / ".tmp_test" / "preflight"
PASS = 0
FAIL = 0


def check(label: str, cond: bool, extra: object = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label} {extra}")


def _load_preflight():
    spec = importlib.util.spec_from_file_location(
        "preflight_under_test", ROOT / "scripts" / "preflight.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # 必须先注册再 exec：dataclasses 解析字符串注解时要用 sys.modules[__module__]
    sys.modules["preflight_under_test"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    pf = _load_preflight()
    check("模块可加载", pf is not None)

    print("\n=== 1. verdict 分级 ===")
    rep = pf.Report()
    rep.add("A", pf.OK)
    check("全绿 = READY", rep.verdict == pf.READY, rep.verdict)
    rep.add("B", pf.WARN)
    check("有 WARN = DEGRADED", rep.verdict == pf.DEGRADED, rep.verdict)
    rep.add("C", pf.FAIL)
    check("有 FAIL = BROKEN", rep.verdict == pf.BROKEN, rep.verdict)

    print("\n=== 2. check_config 显式路径 ===")
    saved = os.environ.pop("ZHIYING_CONFIG", None)
    try:
        rep = pf.Report()
        cfg = pf.check_config(rep, config_path=str(ROOT / "configs" / "app_config.yaml"))
        got = rep.get("CONFIG")
        check("CONFIG 检查存在", got is not None)
        check("默认配置可加载", cfg is not None and got is not None and got.state == pf.OK,
              got.state if got else None)
        check("不污染 ZHIYING_CONFIG", "ZHIYING_CONFIG" not in os.environ)

        rep = pf.Report()
        bad = pf.check_config(rep, config_path=str(WORK / "nope.yaml"))
        got = rep.get("CONFIG")
        # 指定文件不存在时 load_app_config 会告警并回落默认值（不是 FAIL），
        # 但路径非法/解析炸掉必须是 FAIL。这里用一段非法 YAML 验证 FAIL 路径。
        bad_yaml = WORK / "broken.yaml"
        WORK.mkdir(parents=True, exist_ok=True)
        bad_yaml.write_text("storage:\n  sqlite: [unclosed", encoding="utf-8")
        rep = pf.Report()
        pf.check_config(rep, config_path=str(bad_yaml))
        got = rep.get("CONFIG")
        check("非法 YAML → CONFIG FAIL", got is not None and got.state == pf.FAIL, got)
    finally:
        if saved is not None:
            os.environ["ZHIYING_CONFIG"] = saved

    print("\n=== 3. LLM 检查走 llm.mode（新 schema）===")
    mgr_ok = SimpleNamespace(status_report=lambda: ())

    def fake_get_model_manager(*_a, **_k):  # noqa: ANN002, ANN003
        return mgr_ok

    import core.models.manager as mgr_mod

    real_gmm = mgr_mod.get_model_manager
    mgr_mod.get_model_manager = fake_get_model_manager
    try:
        # local 模式 + 权重缺失 + 无 Ollama → 应该给出 MISS（而不是永远 OK）
        rep = pf.Report()
        cfg = SimpleNamespace(
            asr=SimpleNamespace(model="m-asr"),
            embedding=SimpleNamespace(text_model="m-emb", image_model="m-img"),
            vl=SimpleNamespace(model="m-vl"),
            llm=SimpleNamespace(
                mode="local",
                local=SimpleNamespace(model="qwen2.5-7b-gguf"),
                cloud=SimpleNamespace(
                    base_url="", resolved_key=lambda: "",
                ),
            ),
        )
        pf.check_models(rep, cfg)
        got = rep.get("LLM")
        check("LLM 检查存在", got is not None)
        check("local 无权重无 Ollama → MISS/WARN", got is not None and got.state in (pf.MISS, pf.WARN),
              f"{got.state} | {got.detail if got else ''}")
        check("不出现 'keyword（无需模型）'", got is not None and "无需模型" not in got.detail,
              got.detail if got else "")

        # cloud 模式 + 密钥齐全 → OK
        rep = pf.Report()
        cfg.llm = SimpleNamespace(
            mode="cloud",
            local=SimpleNamespace(model="qwen2.5-7b-gguf"),
            cloud=SimpleNamespace(base_url="https://api.example.com/v1", resolved_key=lambda: "sk-x"),
        )
        pf.check_models(rep, cfg)
        got = rep.get("LLM")
        check("cloud 配齐 → OK", got is not None and got.state == pf.OK, got)

        # auto 模式：有密钥 → cloud 优先
        rep = pf.Report()
        cfg.llm = SimpleNamespace(
            mode="auto",
            local=SimpleNamespace(model="qwen2.5-7b-gguf"),
            cloud=SimpleNamespace(base_url="https://api.example.com/v1", resolved_key=lambda: "sk-x"),
        )
        pf.check_models(rep, cfg)
        got = rep.get("LLM")
        check("auto 有密钥 → OK 且报 cloud", got is not None and got.state == pf.OK
              and "cloud" in got.detail, got)
    finally:
        mgr_mod.get_model_manager = real_gmm

    print("\n=== 4. 端口检查读 gradio.port ===")
    rep = pf.Report()
    pf.check_ports(rep, SimpleNamespace(api=SimpleNamespace(port=59999),
                                         gradio=SimpleNamespace(port=59998)))
    keys = {c.key for c in rep.checks}
    check("PORT 检查存在", "PORT" in keys, keys)
    check("gradio.port 被读取（旧 ui.port 已删）",
          not hasattr(pf, "check_ports_old") and "PORT" in keys)

    print("\n=== 5. 输出可解析 ===")
    rep = pf.Report()
    rep.add("PYTHON", pf.OK, "3.11")
    text = pf.render(rep, quiet=True)
    check("机器行含 ###PRECHECK###", "###PRECHECK###" in text and "PYTHON=OK|3.11" in text)
    check("机器行含 VERDICT", "VERDICT=READY" in text)
    rc = pf.main(["--json", "--skip-ports"])
    check("--json 退出码合法", rc in (0, 1, 2), rc)

    print(f"\nPASS={PASS}  FAIL={FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
