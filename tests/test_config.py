"""core/config.py 自检。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import (  # noqa: E402
    AppConfig,
    ConfigFileError,
    ConfigValueError,
    UnknownConfigKeyError,
    detect_project_root,
    get_app_config,
    load_app_config,
    set_app_config,
)

ok: list[str] = []
fail: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        ok.append(name)
        print(f"PASS {name}" + (f" | {detail}" if detail else ""))
    else:
        fail.append(name)
        print(f"FAIL {name}" + (f" | {detail}" if detail else ""))


def expect_error(name: str, fn, exc_type, contains: str = "") -> None:
    try:
        fn()
    except exc_type as exc:
        text = str(exc)
        if contains and contains not in text:
            check(name, False, f"错误信息未包含 {contains!r}: {text[:120]}")
        else:
            check(name, True, text.splitlines()[0][:90])
        return
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"抛出了 {type(exc).__name__}: {exc}")
        return
    check(name, False, "未抛出异常")


ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# 1) 默认加载
# --------------------------------------------------------------------------- #
cfg = load_app_config(env={}, ignore_user_settings=True)
check("加载成功", isinstance(cfg, AppConfig))
check("项目根", cfg.root_dir == ROOT, str(cfg.root_dir))
check("配置源文件", cfg.source_file == ROOT / "configs" / "app_config.yaml")
check("chunk 默认", (cfg.chunk.min_ms, cfg.chunk.max_ms) == (10_000, 30_000))

# YAML 与 dataclass 默认值完全一致 —— 证明两侧字段一一对应、无遗漏无多余
diff = cfg.diff_from_defaults()
check("YAML 与默认值一致", diff == {}, json.dumps(diff, ensure_ascii=False)[:160])

# --------------------------------------------------------------------------- #
# 2) 所有段都能序列化
# --------------------------------------------------------------------------- #
payload = cfg.to_dict()
check(
    "to_dict 无运行时分部", "root_dir" not in payload and "source_file" not in payload
)
check(
    "20 个顶层段（含网页显示与记忆配置）",
    len(payload) == 20 and "ui" in payload and "memory" in payload and "gradio" not in payload,
    f"{sorted(payload)}",
)
try:
    blob = json.dumps(payload, ensure_ascii=False)
    check("to_dict JSON 可序列化", len(blob) > 2000, f"{len(blob)} chars")
except TypeError as exc:
    check("to_dict JSON 可序列化", False, str(exc))

# --------------------------------------------------------------------------- #
# 3) 路径解析
# --------------------------------------------------------------------------- #
check(
    "path(frames) 绝对",
    cfg.path("frames")
    == __import__("backend.runtime", fromlist=["runtime_path"]).runtime_path(
        "data/frames"
    ),
)
check("path(sqlite) 指向文件", cfg.path("sqlite").name == "zhiying.db")
expect_error("path 未知 key", lambda: cfg.path("nope"), UnknownConfigKeyError, "nope")
check("all_paths 条目数", len(cfg.all_paths()) == len(cfg.paths.keys()))

with tempfile.TemporaryDirectory() as tmp:
    tmp_root = Path(tmp)
    alt = load_app_config(
        config_path=ROOT / "configs" / "app_config.yaml",
        env={},
        root=tmp_root,
    )
    check(
        "显式 root 生效",
        alt.path("frames").is_relative_to(tmp_root),
        str(alt.path("frames")),
    )
    created = alt.ensure_dirs()
    check(
        "ensure_dirs 建目录",
        all(p.is_dir() or p.parent.is_dir() for p in created.values()),
    )
    check("ensure_dirs 建 sqlite 父目录", alt.path("sqlite").parent.is_dir())

# --------------------------------------------------------------------------- #
# 4) 环境变量覆盖
# --------------------------------------------------------------------------- #
env_cfg = load_app_config(env={"ZH_YING__CHUNK__MIN_MS": "12000"})
check("env 覆盖 int", env_cfg.chunk.min_ms == 12_000, str(env_cfg.chunk.min_ms))

env_cfg = load_app_config(env={"ZH_YING__RETRIEVAL__WEIGHTS__TEXT": "2.5"})
check(
    "env 覆盖嵌套 dict",
    env_cfg.retrieval.weights["text"] == 2.5,
    str(dict(env_cfg.retrieval.weights)),
)

env_cfg = load_app_config(env={"ZH_YING__ASR__VAD_FILTER": "false"})
check("env 覆盖 bool(false)", env_cfg.asr.vad_filter is False)

env_cfg = load_app_config(env={"ZH_YING__ASR__SUBTITLE_PRIORITY": "vtt,ass"})
check(
    "env 覆盖 tuple",
    env_cfg.asr.subtitle_priority == ("vtt", "ass"),
    str(env_cfg.asr.subtitle_priority),
)

env_cfg = load_app_config(env={"ZH_YING__ASR__LANGUAGE": "zh"})
check("env 覆盖 Optional[str]", env_cfg.asr.language == "zh")

env_cfg = load_app_config(env={"ZH_YING__MEDIA__SCENE_DETECT__THRESHOLD": "35.5"})
check(
    "env 覆盖三层嵌套 float",
    env_cfg.media.scene_detect.threshold == 35.5,
    str(env_cfg.media.scene_detect.threshold),
)

env_cfg = load_app_config(env={"ZH_YING__PATHS__DATA": "D:/ziy-data"})
check(
    "env 覆盖绝对路径原样保留",
    str(env_cfg.path("data")) == "D:\\ziy-data",
    str(env_cfg.path("data")),
)

expect_error(
    "不接受旧本地 LLM 配置",
    lambda: load_app_config(env={"ZH_YING__LLM__LOCAL__N_THREADS": "8"}),
    UnknownConfigKeyError,
    "local",
)

# 不相关的环境变量被忽略
env_cfg = load_app_config(env={"PATH": "/x", "ZH_YING_CHUNK_MIN_MS": "1", "HOME": "/y"})
check("忽略无关变量与单下划线形式", env_cfg.chunk.min_ms == 10_000)

# --------------------------------------------------------------------------- #
# 5) 错误处理
# --------------------------------------------------------------------------- #
with tempfile.TemporaryDirectory() as tmp:
    bad = Path(tmp) / "bad.yaml"
    bad.write_text("chunk:\n  min_ms: 10000\n  typo_key: 1\n", encoding="utf-8")
    expect_error(
        "未知键报错",
        lambda: load_app_config(bad, env={}),
        UnknownConfigKeyError,
        "typo_key",
    )

    bad.write_text("chunk:\n  min_ms: not_a_number\n", encoding="utf-8")
    expect_error(
        "类型错误报错",
        lambda: load_app_config(bad, env={}),
        ConfigValueError,
        "chunk.min_ms",
    )

    bad.write_text("chunk:\n  min_ms: 31000\n", encoding="utf-8")
    expect_error(
        "跨字段校验失败",
        lambda: load_app_config(bad, env={}),
        ConfigValueError,
        "chunk.min_ms",
    )

    bad.write_text("retrieval:\n  rrf_k: 0\n", encoding="utf-8")
    expect_error(
        "非正值校验", lambda: load_app_config(bad, env={}), ConfigValueError, "rrf_k"
    )

    bad.write_text("storage:\n  chroma:\n    clip_collection: ab\n", encoding="utf-8")
    expect_error(
        "Chroma 集合名过短",
        lambda: load_app_config(bad, env={}),
        ConfigValueError,
        "storage.chroma",
    )

    bad.write_text("media:\n  audio_sample_rate: 44100\n", encoding="utf-8")
    expect_error(
        "强制 16kHz",
        lambda: load_app_config(bad, env={}),
        ConfigValueError,
        "audio_sample_rate",
    )

    bad.write_text("chunk: [1, 2, 3]\n", encoding="utf-8")
    expect_error(
        "段类型错误", lambda: load_app_config(bad, env={}), ConfigValueError, "chunk"
    )

    bad.write_text("chunk: {min_ms: 10000}\n", encoding="utf-8")
    loose = load_app_config(bad, env={}, strict=False)
    check("strict=False 容忍未知键", loose.chunk.min_ms == 10_000)

    bad.write_text("chunk: [unclosed\n", encoding="utf-8")
    expect_error(
        "YAML 语法错误", lambda: load_app_config(bad, env={}), ConfigFileError, "YAML"
    )

    bad.write_text("", encoding="utf-8")
    check("空文件全默认", load_app_config(bad, env={}).chunk.min_ms == 10_000)

# 文件不存在 → 默认值 + 告警
missing = load_app_config(Path("definitely/not/here.yaml"), env={})
check("文件缺失降级默认", missing.chunk.max_ms == 30_000)

# env 覆盖出来的非法值也要被校验
expect_error(
    "env 非法值被校验",
    lambda: load_app_config(env={"ZH_YING__CHUNK__MAX_MS": "9000"}),
    ConfigValueError,
    "chunk.min_ms",
)

expect_error(
    "env 未知段报错",
    lambda: load_app_config(env={"ZH_YING__NOPE__X": "1"}),
    UnknownConfigKeyError,
    "nope",
)

# --------------------------------------------------------------------------- #
# 6) .env 加载
# --------------------------------------------------------------------------- #
with tempfile.TemporaryDirectory() as tmp:
    dot = Path(tmp) / ".env"
    dot.write_text(
        "# 注释行\n"
        "ZH_YING__CHUNK__TARGET_MS=15000\n"
        'ZH_YING__ASR__LANGUAGE="en"\n'
        "OPENAI_API_KEY=sk-test\n",
        encoding="utf-8",
    )
    env_cfg = load_app_config(dotenv=[dot], env={})
    check(".env 覆盖生效", env_cfg.chunk.target_ms == 15_000)
    check(".env 去引号", env_cfg.asr.language == "en")
    # 已存在的变量不被覆盖
    env_cfg = load_app_config(dotenv=[dot], env={"ZH_YING__CHUNK__TARGET_MS": "18000"})
    check(".env 不覆盖已有变量", env_cfg.chunk.target_ms == 18_000)

# --------------------------------------------------------------------------- #
# 7) overrides 参数（最高优先级）
# --------------------------------------------------------------------------- #
ov = load_app_config(
    env={"ZH_YING__CHUNK__MIN_MS": "12000"}, overrides={"chunk.min_ms": 14_000}
)
check("overrides 优先级最高", ov.chunk.min_ms == 14_000)

# --------------------------------------------------------------------------- #
# 8) 业务配置桥接
# --------------------------------------------------------------------------- #
flat = cfg.media.scene_detect.flattened()
check(
    "scene flattened 含 keyframe 字段",
    "max_jpeg_bytes" in flat and "blur_threshold" in flat,
)
check(
    "scene flattened 无嵌套",
    not any(
        isinstance(v, object) and hasattr(v, "__dataclass_fields__")
        for v in flat.values()
    ),
    str(sorted(flat))[:120],
)
chunk_kwargs = cfg.chunk.as_chunk_config()
check(
    "chunk kwargs 映射到 ChunkConfig 字段名",
    set(chunk_kwargs)
    == {
        "min_ms",
        "max_ms",
        "target_ms",
        "overlap_ms",
        "silence_threshold_ms",
        "hard_max_ms",
        "weight_tail_penalty",
        "max_chars",
    },
    str(sorted(chunk_kwargs)),
)


def _chunk_ok(kwargs):
    try:
        from backend.segmentation.chunker import ChunkConfig as _CC

        built = _CC(**kwargs)
        return (
            built.min_ms == cfg.chunk.min_ms and built.max_chars == cfg.chunk.max_chars
        )
    except Exception as exc:  # noqa: BLE001
        print("    ChunkConfig 展开失败:", exc)
        return False


check("chunk kwargs 可直接展开", _chunk_ok(chunk_kwargs))
check(
    "chunk 下掉未接线的死配置",
    not hasattr(cfg.chunk, "weight_target") and not hasattr(cfg.chunk, "sentence_end"),
)

# --------------------------------------------------------------------------- #
# 9) 单例
# --------------------------------------------------------------------------- #
a = get_app_config(env={})
b = get_app_config()
check("单例复用", a is b)
custom = load_app_config(env={"ZH_YING__CHUNK__MIN_MS": "11000"})
set_app_config(custom)
check("set_app_config 注入", get_app_config().chunk.min_ms == 11_000)
set_app_config(a)

# --------------------------------------------------------------------------- #
# 10) 项目根探测
# --------------------------------------------------------------------------- #
check("detect_project_root", detect_project_root(ROOT / "core") == ROOT)
env_cfg = load_app_config(env={"ZH_YING_PROJECT_ROOT": str(ROOT)})
check("env 指定项目根", env_cfg.root_dir == ROOT)

# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
print()
if fail:
    print(f"失败 {len(fail)} 项:")
    for name in fail:
        print(f"  - {name}")
    sys.exit(1)
print(f"通过 {len(ok)} / {len(ok)}")
