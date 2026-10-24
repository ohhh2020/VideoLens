"""本地自检：不依赖 pytest，直接断言 ModelManager 的关键契约。"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\VideoLens")

from core.models.manager import (
    ModelDownloadError,
    ModelManager,
    ModelManagerError,
    ModelState,
    UnknownModelError,
)

CFG = Path(r"D:\VideoLens\.tmp_test\models.test.yaml")
mgr = ModelManager(config_path=CFG, show_progress=False)
ok: list[str] = []
fail: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    (ok if cond else fail).append(f"{name} {extra}")
    print(("PASS " if cond else "FAIL ") + name + (f" | {extra}" if extra else ""))


# 1) 幂等 + 路径正确
p1 = mgr.ensure_model("tiny-hf")
p2 = mgr.ensure_model("tiny-hf")
check("ensure_model 幂等", p1 == p2 == mgr.cache_root / "tiny-hf", str(p1))

# 2) 状态 PRESENT 且带来源信息
st = mgr.status("tiny-hf")
check("status PRESENT", st.state is ModelState.PRESENT and st.source == "huggingface",
      f"{st.state.value}/{st.source}/{st.file_count} files/{st.size_bytes}B")

# 3) 未知模型
try:
    mgr.get_spec("nope")
    check("UnknownModelError", False)
except UnknownModelError as exc:
    check("UnknownModelError", "未配置" in str(exc), str(exc)[:60])

# 4) 非法模型名（防路径穿越）
for bad in ("../etc", "a/b", ".hidden", ""):
    try:
        mgr.get_spec(bad)
        check(f"拒绝非法名 {bad!r}", False)
    except ModelManagerError:
        check(f"拒绝非法名 {bad!r}", True)

# 5) INCOMPLETE：删掉关键文件后能被识别
victim = p1 / "config.json"
backup = victim.read_bytes()
victim.unlink()
st2 = mgr.status("tiny-hf")
check("识别 INCOMPLETE", st2.state is ModelState.INCOMPLETE and "config.json" in st2.missing_files,
      st2.message)
victim.write_bytes(backup)
check("恢复后 PRESENT", mgr.status("tiny-hf").state is ModelState.PRESENT)

# 6) 失败路径：两个源都不存在 -> ModelDownloadError，且记录两个源
started = time.monotonic()
try:
    mgr.ensure_model("broken-repo")
    check("broken-repo 抛错", False)
except ModelDownloadError as exc:
    elapsed = time.monotonic() - started
    check("broken-repo 抛 ModelDownloadError", True, f"{len(exc.errors)} 源失败, {elapsed:.1f}s")
    check("错误信息含两个源", len(exc.errors) == 2, " || ".join(e[:50] for e in exc.errors))
except Exception as exc:  # noqa: BLE001
    check("broken-repo 抛 ModelDownloadError", False, f"{type(exc).__name__}: {exc}")

# 7) 强制指定不存在的源 -> ConfigError
try:
    mgr.ensure_model("tiny-hf", source="modelscope")
    check("强制未配置源报错", False)
except ModelManagerError as exc:
    check("强制未配置源报错", "没有配置下载源" in str(exc), str(exc)[:60])

# 8) 失败后不留垃圾：暂存区应为空
leftovers = [p.name for p in mgr.config.cache.partial_root.iterdir()] if mgr.config.cache.partial_root.is_dir() else []
check("失败后暂存区干净", leftovers == [], str(leftovers))

# 9) prune 能清理过期暂存目录
stale = mgr.config.cache.partial_root / "stale-model-1"
stale.mkdir(parents=True, exist_ok=True)
(stale / "junk.bin").write_bytes(b"x" * 16)
old = time.time() - 7200
import os
os.utime(stale, (old, old))
check("prune 清理过期暂存", mgr.prune_partials(older_than_hours=1) == 1)

# 10) remove 幂等
check("remove 首次 True", mgr.remove("tiny-hf") is True)
check("remove 二次 False", mgr.remove("tiny-hf") is False)
check("删除后状态回退 MISSING", mgr.status("tiny-hf").state is ModelState.MISSING)

# 11) 配置快照与设备探测
check("配置解析", mgr.config.version == 1 and len(mgr.config.models) == 3, str(mgr.list_models()))
check("设备探测", mgr.device_hint in {"cpu", "cuda", "mps"}, mgr.device_hint)

# 12) describe() 可 JSON 序列化
import json
check("describe 可序列化", bool(json.dumps(mgr.describe("tiny-hf"), ensure_ascii=False)))

# 13) 全局配置能正常构造（真实 models.yaml）
real = ModelManager()
check("真实配置可构造", len(real.list_models(required_only=True)) == 5,
      ",".join(real.list_models(required_only=True)))

# 14) 模型级 allow_patterns 必须被解析（回归：曾经整段被静默丢弃 → 拉全仓库）
spec_7b = real.get_spec("qwen2.5-7b-gguf")
check("模型级 allow_patterns 已解析", bool(spec_7b.allow_patterns), str(spec_7b.allow_patterns))
spec_bge = real.get_spec("bge-m3")
check("无 allow 时保持为空", spec_bge.allow_patterns == (), str(spec_bge.allow_patterns))

# 15) 生效规则：源级覆盖模型级；ignore 取并集
from core.models.manager import SourceSpec, _effective_allow_patterns, _effective_ignore_patterns
override = SourceSpec(repo_id="x/y", allow_patterns=("*.safetensors",))
check("源级 allow 覆盖模型级", _effective_allow_patterns(spec_7b, override) == ("*.safetensors",))
fallback = SourceSpec(repo_id="x/y")
check("源级空则回退模型级", _effective_allow_patterns(spec_7b, fallback) == spec_7b.allow_patterns,
      str(_effective_allow_patterns(spec_7b, fallback)))
merged_ignore = _effective_ignore_patterns(spec_7b, SourceSpec(repo_id="x/y", ignore_patterns=("*.md", "*.bin")))
check("ignore 取并集且去重", merged_ignore == ("*.md", "*.bin", "*.png", ".gitattributes"), str(merged_ignore))

# 16) 端到端：tiny-ms 走 modelscope 且只落盘 1 个 json（白名单生效）
ms_dir = mgr.ensure_model("tiny-ms", source="modelscope")
landed = sorted(p.name for p in ms_dir.iterdir() if p.is_file() and p.name != "manifest.json")
check("modelscope 白名单生效", landed == ["configuration.json"], str(landed))

print(f"\n通过 {len(ok)} / {len(ok) + len(fail)}")
if fail:
    print("失败项:")
    for item in fail:
        print("  - " + item)
sys.exit(0 if not fail else 1)
