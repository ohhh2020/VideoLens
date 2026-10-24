#!/usr/bin/env python
"""知影自检总入口 —— 零第三方依赖，只用标准库。

设计要点：
1. **不依赖 pytest**。答辩机只要装了项目运行时就能跑（`python tests/run_all.py`），
   不需要额外安装测试框架；每个测试文件内部是 `check(name, cond, detail)` 式的
   极简断言累积器，退出码非 0 即失败。
2. **按依赖从底向上排序**：配置 → 媒体 → 模型 → 检索 → 存储 → 流水线 → 端到端。
   前面的失败不影响后面的执行（每个文件独立子进程），最后统一汇总。
3. **超时可配**：媒体测试要真实调用 ffmpeg 生成合成片，端到端测试要跑完整流水线，
   整体耗时约 1-3 分钟；`--timeout` 可单独放宽。
4. **输出 UTF-8**：Windows 控制台默认 cp936，中文断言信息会乱码；子进程强制
   `PYTHONIOENCODING=utf-8`，本文件自身也尝试重置 stdout。

用法::

    python tests/run_all.py                # 跑全部
    python tests/run_all.py -k media       # 只跑名字含 media 的
    python tests/run_all.py --list         # 列出用例
    python tests/run_all.py -v             # 显示每个文件的完整输出
    python tests/run_all.py --timeout 600  # 单文件超时（秒）
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess  # noqa: S404 - 只跑本仓库内的测试脚本
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent

#: (文件名, 分类, 说明) —— 顺序即执行顺序，从底层依赖向上。
SUITES: tuple[tuple[str, str, str], ...] = (
    ("test_config.py", "配置", "app_config.yaml 加载、env 覆盖、类型转换、跨段校验"),
    ("test_model_manager.py", "模型", "ModelManager 下载/校验/清单/清理"),
    ("test_models.py", "模型", "ASR/VL/Embedding/LLM 四个适配器 + 提示词"),
    ("test_media_parse.py", "媒体", "字幕解析、来源识别、转写清洗"),
    ("test_media.py", "媒体", "场景检测与关键帧抽取（需要 ffmpeg）"),
    ("test_chunker.py", "切片", "自适应语音切片（静音/标点/超长兜底）"),
    ("test_vector.py", "检索", "Chroma 向量库封装（需要 chromadb）"),
    ("test_retrieval.py", "检索", "三通道 RRF 混合检索"),
    ("test_storage.py", "存储", "SQLite 仓储、容量限制、任务恢复"),
    ("test_pipeline.py", "流水线", "全链路编排 + 问答 + 降级路径"),
    ("test_chapters.py", "流水线", "章节分组 + 标题渲染 + 位置铺开取样"),
    ("test_claims.py", "流水线", "回答证据校验：悬空编号/错配/孤立句/补出处"),
    ("test_settings.py", "设置", "界面设置白名单、密钥打码、配置优先级、云端视觉装配"),
    ("test_gc.py", "运维", "TTL 清理计划、临时文件判定、保留策略执行"),
    ("test_eval.py", "评测", "CER/WER 计算口径、数字归一化、参考文本读取"),
    ("test_preflight.py", "运维", "启动预检：verdict 分级、--config、llm.mode/gradio.port 新 schema"),
    ("test_api.py", "接口", "HTTP/SSE/MCP 端点、鉴权、错误信封"),
    ("test_cli.py", "客户端", "命令解析、本地/远程后端、输出渲染"),
    ("test_e2e.py", "端到端", "media↔chunker 联动"),
)


def _enable_utf8() -> None:
    """尽量把 stdout 切到 UTF-8，失败就保持原样（不影响测试结果）。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - 老环境或已被重定向
            pass


def _run_one(path: Path, timeout: float, verbose: bool) -> tuple[bool, float, str]:
    """跑一个测试文件，返回 (是否通过, 耗时秒, 输出摘要)。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # 让测试文件能直接 `import core.*`，同时不污染子进程 cwd。
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(ROOT)]
    )

    started = time.monotonic()
    try:
        proc = subprocess.run(  # noqa: S603 - 参数来自仓库内固定清单
            [sys.executable, str(path)],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, time.monotonic() - started, f"超时（>{timeout:.0f}s）"

    elapsed = time.monotonic() - started
    text = (proc.stdout or b"").decode("utf-8", "replace")
    err = (proc.stderr or b"").decode("utf-8", "replace")
    ok = proc.returncode == 0
    detail = text if verbose else _summarize(text, err, ok)
    return ok, elapsed, detail


def _summarize(stdout: str, stderr: str, ok: bool) -> str:
    """失败时抓 FAIL 行与 Traceback 尾部，成功时抓 PASS=/统计行。"""
    lines = stdout.splitlines()
    if ok:
        hits = [ln.strip() for ln in lines if "PASS=" in ln or "ALL PASS" in ln]
        return hits[-1] if hits else "通过"
    fails = [ln.strip() for ln in lines if ln.lstrip().startswith("FAIL")]
    tail = [ln for ln in (stdout + "\n" + stderr).splitlines() if ln.strip()][-12:]
    parts = []
    if fails:
        parts.append(f"{len(fails)} 处失败：")
        parts.extend(f"  {ln}" for ln in fails[:8])
    parts.append("--- 输出尾部 ---")
    parts.extend(tail)
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    _enable_utf8()
    parser = argparse.ArgumentParser(
        prog="run_all.py", description="知影自检总入口（零第三方依赖）"
    )
    parser.add_argument("-k", "--filter", default="", help="只跑文件名含该子串的用例")
    parser.add_argument("--list", action="store_true", help="只列出用例，不执行")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示完整输出")
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="单文件超时秒数（默认 300）"
    )
    parser.add_argument(
        "--keep-going", action="store_true", default=True, help="单个失败继续跑（默认开启）"
    )
    args = parser.parse_args(argv)

    cases = [(n, c, d) for n, c, d in SUITES if not args.filter or args.filter in n]
    missing = [n for n, _c, _d in cases if not (TESTS_DIR / n).is_file()]
    cases = [(n, c, d) for n, c, d in cases if (TESTS_DIR / n).is_file()]

    if args.list:
        for name, cat, desc in cases:
            print(f"{cat:6s} {name:24s} {desc}")
        for name in missing:
            print(f"{'缺失':6s} {name:24s} （文件不存在）")
        return 0 if not missing else 1

    if not cases:
        print(f"没有匹配 {args.filter!r} 的用例")
        return 1

    print("=" * 78)
    print(f"知影自检 · {len(cases)} 个用例 · Python {sys.version.split()[0]}")
    print(f"项目根: {ROOT}")
    print("=" * 78)

    results: list[tuple[str, bool, float, str]] = []
    total_started = time.monotonic()
    for name, cat, desc in cases:
        print(f"\n[{cat}] {name} …", flush=True)
        ok, elapsed, detail = _run_one(TESTS_DIR / name, args.timeout, args.verbose)
        results.append((name, ok, elapsed, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {elapsed:.1f}s")
        if detail:
            for line in detail.splitlines():
                print(f"  {line}")

    total = time.monotonic() - total_started
    passed = sum(1 for _n, ok, _e, _d in results if ok)
    failed = [n for n, ok, _e, _d in results if not ok]

    # Node 侧（Obsidian 插件的纯逻辑）不是 Python 套件，单独跑：有 node 就验，
    # 没有就明说跳过 —— 不能让"没装 node"变成红色失败。
    node = os.environ.get("ZHIYING_NODE") or shutil.which("node")
    node_probe = TESTS_DIR / "probe_obsidian_plugin.js"
    node_status = ""
    if node and node_probe.is_file() and not args.filter:
        print("\n[插件] probe_obsidian_plugin.js …", flush=True)
        try:
            proc = subprocess.run(  # noqa: S603 - 固定路径 + 本仓库脚本
                [node, str(node_probe)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=args.timeout,
            )
            node_ok = proc.returncode == 0
            node_status = "PASS" if node_ok else "FAIL"
            if not node_ok:
                failed.append("probe_obsidian_plugin.js")
                for line in (proc.stdout or "").splitlines():
                    if "FAIL" in line:
                        print(f"  {line.strip()}")
        except (OSError, subprocess.SubprocessError) as exc:
            node_status = f"SKIP（{exc}）"
    elif not args.filter:
        node_status = "SKIP（未找到 node）"

    print("\n" + "=" * 78)
    total_cases = len(results) + (1 if node_status.startswith(("PASS", "FAIL")) else 0)
    if node_status.startswith("PASS"):
        passed += 1
    print(f"汇总: {passed}/{total_cases} 通过，用时 {total:.1f}s")
    for name, ok, elapsed, _detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:24s} {elapsed:6.1f}s")
    if node_status:
        print(f"  {node_status:5s} probe_obsidian_plugin.js")
    if failed:
        print("\n失败用例: " + ", ".join(failed))
    if missing:
        print("缺失文件: " + ", ".join(missing))
    print("=" * 78)
    return 0 if not failed and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
