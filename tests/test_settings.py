"""界面设置层（``core/user_settings.py``）与配置覆盖链的自检。

跑法::

    python tests/test_settings.py

覆盖重点：白名单（防远程乱写配置）、类型强制、密钥打码、原子落盘、
四层优先级（默认 < yaml < user_settings < 环境变量）、以及云端视觉后端
是否真的被装配出来。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_app_config, user_settings_path  # noqa: E402
from core.user_settings import (  # noqa: E402
    SETTABLE_KEYS,
    SettingsError,
    current_settings,
    describe_backends,
    format_backends,
    save_settings,
    settings_file,
    validate_settings,
)

PASS = 0
FAIL = 0


def ok(condition: bool, message: str) -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS {message}")
    else:
        FAIL += 1
        print(f"  FAIL {message}")


def raises(func, message: str) -> None:
    try:
        func()
    except SettingsError:
        ok(True, message)
    except Exception as exc:  # noqa: BLE001
        ok(False, f"{message}（抛出 {type(exc).__name__} 而非 SettingsError）")
    else:
        ok(False, f"{message}（没有抛错）")


def main() -> int:
    print("=" * 70)
    print("1. 白名单与类型强制")
    print("=" * 70)
    clean = validate_settings({
        "llm.mode": "cloud",
        "llm.cloud.timeout_s": "45",
        "llm.cloud.temperature": "0.7",
        "vl.cloud_enabled": "true",
        "models.device": "cuda",
    })
    ok(clean["llm.cloud.timeout_s"] == 45 and isinstance(clean["llm.cloud.timeout_s"], int),
       "字符串 '45' 强转为 int")
    ok(abs(clean["llm.cloud.temperature"] - 0.7) < 1e-9, "字符串 '0.7' 强转为 float")
    ok(clean["vl.cloud_enabled"] is True, "字符串 'true' 强转为 bool")
    ok(clean["models.device"] == "cuda", "枚举值原样保留")

    ok(validate_settings({"vl.cloud_enabled": "0"})["vl.cloud_enabled"] is False,
       "字符串 '0' 强转为 False")
    ok(validate_settings({"vl.cloud_enabled": ""})["vl.cloud_enabled"] is False,
       "空串强转为 False")

    raises(lambda: validate_settings({"llm.cloud.max_tokens": "很多"}),
           "非数字字符串转 int 被拒绝")
    raises(lambda: validate_settings({"vl.cloud_enabled": "也许"}),
           "非布尔字符串转 bool 被拒绝")
    raises(lambda: validate_settings({"llm.mode": "hybrid"}),
           "llm.mode 越界被拒绝")
    raises(lambda: validate_settings({"paths.sqlite": "D:/evil.db"}),
           "白名单外的键被拒绝（不能远程改写数据库路径）")
    raises(lambda: validate_settings({"llm.cloud.base_url": "x", "vl.model": "y"}),
           "整批里只要一个非法就全部拒绝")

    print()
    print("=" * 70)
    print("2. 落盘、合并与读取")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = save_settings({"llm.mode": "cloud", "llm.cloud.model": "deepseek-chat"},
                             root=root)
        ok(path.is_file(), f"文件已创建 {path.name}")
        ok(path.parent.name == "data", "落在 data/ 目录下")
        payload = json.loads(path.read_text(encoding="utf-8"))
        ok(payload["llm.mode"] == "cloud", "内容正确")
        ok("\n" in path.read_text(encoding="utf-8"), "写的是带缩进的多行 JSON")

        save_settings({"vl.cloud_model": "qwen-vl-max"}, root=root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        ok(payload["llm.mode"] == "cloud" and payload["vl.cloud_model"] == "qwen-vl-max",
           "merge=True 保留旧键并新增")

        save_settings({"llm.mode": "local"}, root=root, merge=False)
        payload = json.loads(path.read_text(encoding="utf-8"))
        ok(list(payload) == ["llm.mode"], "merge=False 整份替换")

        ok(settings_file(root) == path, "settings_file() 与 save_settings() 路径一致")

    print()
    print("=" * 70)
    print("3. 密钥打码与清除")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        secret = "sk-abcdefghijklmnop1234"
        save_settings({"llm.cloud.api_key": secret}, root=root)

        shown = current_settings(root)
        ok(shown["llm.cloud.api_key"] != secret, "读出来时已打码")
        ok(shown["llm.cloud.api_key"].startswith("sk-a"), "保留前 4 位便于辨认")
        ok(shown["llm.cloud.api_key"].endswith("1234"), "保留后 4 位便于辨认")
        ok(shown["llm.cloud.api_key"].count("*") >= 6, "中间用星号遮盖")

        raw = json.loads(settings_file(root).read_text(encoding="utf-8"))
        ok(raw["llm.cloud.api_key"] == secret, "磁盘上仍是明文（本地单机工具）")

        save_settings({"llm.cloud.api_key": ""}, root=root)
        raw = json.loads(settings_file(root).read_text(encoding="utf-8"))
        ok("llm.cloud.api_key" not in raw, "传空串表示清除该项")

        save_settings({"llm.cloud.api_key": "short"}, root=root)
        ok(current_settings(root)["llm.cloud.api_key"] == "*****",
           "短密钥整体打码（不露字符）")

    print()
    print("=" * 70)
    print("4. 损坏文件的容错")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = settings_file(root)
        path.parent.mkdir(parents=True)
        path.write_text("{这不是 JSON", encoding="utf-8")
        ok(current_settings(root) == {}, "坏 JSON 读成空设置而不是抛异常")
        try:
            save_settings({"llm.mode": "local"}, root=root)
            ok(True, "坏文件仍可被新写入覆盖")
        except Exception as exc:  # noqa: BLE001
            ok(False, f"坏文件仍可被新写入覆盖（{type(exc).__name__}）")

        path.write_text("[1, 2, 3]", encoding="utf-8")
        ok(current_settings(root) == {}, "顶层不是对象时也当空设置")

    print()
    print("=" * 70)
    print("5. 配置优先级：默认 < yaml < user_settings < 环境变量")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "configs").mkdir(parents=True)
        (root / "configs" / "app_config.yaml").write_text(
            "app:\n  name: 知影\nllm:\n  mode: local\n  cloud:\n    model: yaml-model\n",
            encoding="utf-8",
        )
        save_settings({"llm.mode": "cloud", "llm.cloud.model": "json-model"}, root=root)

        cfg = load_app_config(root=root, env={}, strict=False)
        ok(cfg.llm.mode == "cloud", "user_settings 压过 yaml")
        ok(cfg.llm.cloud.model == "json-model", "cloud.model 也压过 yaml")

        cfg_env = load_app_config(
            root=root, env={"ZH_YING__LLM__MODE": "local"}, strict=False
        )
        ok(cfg_env.llm.mode == "local", "环境变量反过来压过 user_settings")

        cfg_override = load_app_config(
            root=root, env={}, strict=False, overrides={"llm.mode": "auto"}
        )
        ok(cfg_override.llm.mode == "auto", "显式 overrides 优先级最高")

        ok(user_settings_path(root) == settings_file(root), "两个入口指向同一路径")

    print()
    print("=" * 70)
    print("6. 云端视觉后端装配")
    print("=" * 70)
    from core.models.vl import OpenAiVisionRunner, VlService  # noqa: E402

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "configs").mkdir(parents=True)
        (root / "configs" / "app_config.yaml").write_text(
            "app:\n  name: 知影\n", encoding="utf-8"
        )
        save_settings({
            "vl.cloud_enabled": True,
            "vl.cloud_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "vl.cloud_model": "qwen-vl-plus",
            "vl.cloud_api_key": "sk-test-key-1234",
        }, root=root)
        cfg = load_app_config(root=root, env={}, strict=False)
        ok(cfg.vl.cloud_enabled is True, "配置里读到 cloud_enabled")

        service = VlService.from_config(cfg, manager=None)
        ok(isinstance(service._runner, OpenAiVisionRunner),  # noqa: SLF001
           "云端开启时装配的是 OpenAiVisionRunner")
        ok(service.available is True, "云端后端标记为可用")
        ok(service.model_name == "qwen-vl-plus", "模型名透传")
        ok(service._sequence_frames == cfg.vl.cloud_sequence_frames,  # noqa: SLF001
           "多帧上限用云端值（默认 8）")
        ok(service._runner.endpoint.endswith("/chat/completions"),  # noqa: SLF001
           "端点拼接正确")
        ok(service._runner.api_key == "sk-test-key-1234",  # noqa: SLF001
           "密钥从设置里取到")

        save_settings({"vl.cloud_enabled": False}, root=root)
        cfg2 = load_app_config(root=root, env={}, strict=False)
        service2 = VlService.from_config(cfg2, manager=None)
        ok(not isinstance(service2._runner, OpenAiVisionRunner),  # noqa: SLF001
           "关掉云端后退回本地链路")
        ok(service2.available is False, "没有 ModelManager 时降级")
        ok("云端视觉未启用" in service2.degraded_reason,
           "降级原因里点明云端也没用上")

        save_settings({"vl.cloud_enabled": True, "vl.cloud_base_url": ""}, root=root)
        cfg3 = load_app_config(root=root, env={}, strict=False)
        service3 = VlService.from_config(cfg3, manager=None)
        ok(not isinstance(service3._runner, OpenAiVisionRunner),  # noqa: SLF001
           "开了云端但地址为空时不装配（不抛异常）")

    print()
    print("=" * 70)
    print("7. 后端状态展示")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "configs").mkdir(parents=True)
        (root / "configs" / "app_config.yaml").write_text(
            "app:\n  name: 知影\n", encoding="utf-8"
        )
        save_settings({
            "llm.mode": "cloud",
            "llm.cloud.base_url": "https://api.deepseek.com/v1",
            "llm.cloud.model": "deepseek-chat",
            "llm.cloud.api_key": "sk-real-key-9999",
            "vl.cloud_enabled": True,
            "vl.cloud_base_url": "https://example.com/v1",
            "vl.cloud_model": "qwen-vl-plus",
        }, root=root)
        cfg = load_app_config(root=root, env={}, strict=False)
        statuses = describe_backends(cfg)
        by_name = {s.name: s for s in statuses}
        ok(len(statuses) == 4, "四个通道都列出")
        ok(by_name["云端大模型"].enabled is True, "云端大模型：有密钥即启用")
        ok(by_name["本地大模型"].enabled is False, "mode=cloud 时本地大模型不算启用")
        ok(by_name["云端视觉"].enabled is True, "云端视觉：地址与模型齐即启用")
        ok(by_name["云端视觉"].warning != "" and "密钥" in by_name["云端视觉"].warning,
           "没填密钥时给出提示（本地 vLLM 可忽略）")
        ok(by_name["本地视觉"].enabled is False, "云端接管后本地视觉不算启用")
        ok("已启用" in by_name["云端大模型"].line(), "行内文案可读")

        text = format_backends(cfg)
        ok("2/4" in text, "汇总行统计可用通道数")
        ok("deepseek-chat" in text and "qwen-vl-plus" in text, "模型名出现在报告里")

        save_settings({"llm.cloud.api_key": ""}, root=root)
        cfg2 = load_app_config(root=root, env={}, strict=False)
        cloud_llm = {s.name: s for s in describe_backends(cfg2)}["云端大模型"]
        ok(cloud_llm.enabled is False, "清掉密钥后云端大模型变为未启用")
        ok("没有密钥" in cloud_llm.warning, "警告里说明缺密钥")

    print()
    print("=" * 70)
    print("8. 密钥不外泄")
    print("=" * 70)
    ok(all(not key.startswith("api_key") for key in SETTABLE_KEYS),
       "白名单键名都是点号路径")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_app_config(root=Path(tmp), env={}, strict=False)
        text = format_backends(cfg)
        ok("sk-" not in text, "后端状态报告里不含密钥片段")

    print()
    print("=" * 70)
    print(f"结果：PASS={PASS} FAIL={FAIL}")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
