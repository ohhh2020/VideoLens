"""三个必需模型的下载、验证、状态与清理入口。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

# 允许直接以 `python scripts/download_models.py` 运行（脚本目录不在 sys.path 中）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.models.manager import (  # noqa: E402 - 必须在上面的 sys.path 注入之后
    ModelManager,
    ModelManagerError,
    ModelSpec,
    ModelState,
    ModelStatus,
    configure_logging,
)

_EXIT_OK = 0
_EXIT_FAILED = 1
_EXIT_USAGE = 2

_STATE_LABEL = {
    ModelState.PRESENT: "present",
    ModelState.MISSING: "missing",
    ModelState.INCOMPLETE: "incomplete",
    ModelState.CORRUPT: "corrupt",
}


# --------------------------------------------------------------------------- #
# 展示辅助
# --------------------------------------------------------------------------- #
def _format_bytes(num: float) -> str:
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < step or unit == "TB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= step
    return f"{num:.1f} TB"


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    """定宽表格（纯 ASCII 表头，避免中文字符宽度对齐问题）。"""
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def _spec_row(spec: ModelSpec) -> list[str]:
    sources = ", ".join(
        f"{k}:{v.repo_id}@{v.revision}" for k, v in spec.sources.items()
    )
    return [
        spec.name,
        spec.kind,
        "yes" if spec.required else "no",
        f"{spec.size_hint_bytes / 1024**3:.1f}G" if spec.size_hint_bytes else "-",
        ",".join(spec.allow_patterns) or "*",
        sources,
    ]


def _status_row(status: ModelStatus) -> list[str]:
    return [
        status.name,
        status.kind,
        "yes" if status.required else "no",
        _STATE_LABEL[status.state],
        _format_bytes(status.size_bytes) if status.size_bytes else "-",
        status.source or "-",
        status.message or "",
    ]


def _status_payload(status: ModelStatus) -> dict[str, Any]:
    return {
        "name": status.name,
        "kind": status.kind,
        "required": status.required,
        "state": status.state.value,
        "path": str(status.path),
        "size_bytes": status.size_bytes,
        "file_count": status.file_count,
        "source": status.source,
        "revision": status.revision,
        "missing_files": list(status.missing_files),
        "message": status.message,
    }


# --------------------------------------------------------------------------- #
# 动作
# --------------------------------------------------------------------------- #
def _resolve_targets(
    manager: ModelManager, args: argparse.Namespace
) -> tuple[str, ...]:
    if args.models:
        requested = [
            item.strip()
            for item in args.models.replace(",", " ").split()
            if item.strip()
        ]
        unknown = [name for name in requested if name not in manager.list_models()]
        if unknown:
            print(f"[ERROR] 未配置的模型: {', '.join(unknown)}", file=sys.stderr)
            print(f"        可选: {', '.join(manager.list_models())}", file=sys.stderr)
            raise SystemExit(_EXIT_USAGE)
        return tuple(requested)
    return manager.list_models()


def action_list(manager: ModelManager, args: argparse.Namespace) -> int:
    specs = list(manager.config.models.values())
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "name": s.name,
                        "kind": s.kind,
                        "required": s.required,
                        "license": s.license,
                        "description": s.description,
                        "size_hint_gb": round(s.size_hint_bytes / 1024**3, 2),
                        "required_files": list(s.required_files),
                        "allow_patterns": list(s.allow_patterns),
                        "ignore_patterns": list(s.ignore_patterns),
                        "sources": {
                            k: {"repo_id": v.repo_id, "revision": v.revision}
                            for k, v in s.sources.items()
                        },
                    }
                    for s in specs
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return _EXIT_OK
    print(f"配置文件 : {manager.config.config_path}")
    print(f"缓存目录 : {manager.cache_root}")
    print(f"下载顺序 : {' -> '.join(manager.config.network.sources_order)}")
    print(
        f"HF 镜像  : {manager.config.network.hf_endpoint or 'https://huggingface.co'}"
    )
    print(f"设备探测 : {manager.device_hint}")
    print()
    _print_table(
        ["MODEL", "KIND", "REQ", "SIZE", "ALLOW", "SOURCES"],
        [_spec_row(spec) for spec in specs],
    )
    return _EXIT_OK


def action_status(manager: ModelManager, args: argparse.Namespace) -> int:
    targets = _resolve_targets(manager, args)
    statuses = [manager.status(name, verify_hash=args.verify) for name in targets]
    if args.json:
        print(
            json.dumps(
                [_status_payload(s) for s in statuses], ensure_ascii=False, indent=2
            )
        )
    else:
        _print_table(
            ["MODEL", "KIND", "REQ", "STATE", "SIZE", "SOURCE", "NOTE"],
            [_status_row(status) for status in statuses],
        )
        ready = sum(1 for s in statuses if s.available)
        total_gb = sum(s.size_bytes for s in statuses) / 1024**3
        print()
        print(f"就绪 {ready}/{len(statuses)}，本地占用 {total_gb:.2f} GB")
    if args.verify:
        # --verify 时，必需的模型处于 CORRUPT 视为失败
        corrupt = [s.name for s in statuses if not s.available]
        return _EXIT_FAILED if corrupt else _EXIT_OK
    return _EXIT_OK


def action_download(manager: ModelManager, args: argparse.Namespace) -> int:
    targets = _resolve_targets(manager, args)
    logger = logging.getLogger("zhiying.models")
    logger.info("开始处理 %d 个模型: %s", len(targets), ", ".join(targets))

    ok: dict[str, Path] = {}
    failed: dict[str, str] = {}
    for index, name in enumerate(targets, start=1):
        print(f"\n[{index}/{len(targets)}] {name}", flush=True)
        try:
            path = manager.ensure_model(
                name,
                force=args.force,
                verify_hash=args.verify,
                source=args.source,
                show_progress=not args.quiet,
            )
            ok[name] = path
            print(f"    OK  -> {path}", flush=True)
        except ModelManagerError as exc:
            failed[name] = str(exc)
            print(f"    FAIL -> {exc}", flush=True)
            if not args.keep_going:
                break
        except KeyboardInterrupt:
            print("\n已取消（已下载内容保留在暂存区，可续传）", file=sys.stderr)
            return _EXIT_FAILED
        except Exception as exc:  # noqa: BLE001 - CLI 边界，任何异常都不该抛栈给用户
            failed[name] = f"{type(exc).__name__}: {exc}"
            print(f"    FAIL -> {type(exc).__name__}: {exc}", flush=True)
            if not args.keep_going:
                break

    print()
    print(f"成功 {len(ok)}，失败 {len(failed)}")
    for name, reason in failed.items():
        print(f"  - {name}: {reason}", file=sys.stderr)

    required_failed = [
        name
        for name in failed
        if manager.config.models.get(name) is not None
        and manager.config.models[name].required
    ]
    if args.json:
        print(
            json.dumps(
                {
                    "ok": {k: str(v) for k, v in ok.items()},
                    "failed": failed,
                    "required_failed": required_failed,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return _EXIT_FAILED if failed else _EXIT_OK


def action_remove(manager: ModelManager, args: argparse.Namespace) -> int:
    targets = _resolve_targets(manager, args)
    removed = 0
    for name in targets:
        if manager.remove(name, manifest_only=args.manifest_only):
            print(f"removed: {name}")
            removed += 1
        else:
            print(f"skipped: {name}（本地不存在）")
    print(f"共删除 {removed} 个")
    return _EXIT_OK


def action_prune(manager: ModelManager, args: argparse.Namespace) -> int:
    count = manager.prune_partials(older_than_hours=args.prune_hours)
    print(f"已清理 {count} 个过期下载暂存目录")
    return _EXIT_OK


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_models.py",
        description="知影模型权重预下载与体检工具（ModelScope 优先，自动回退 HuggingFace）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None, help="models.yaml 路径")
    parser.add_argument(
        "--cache-root", type=Path, default=None, help="模型缓存根目录（覆盖配置）"
    )
    parser.add_argument(
        "--models", type=str, default=None, help="模型名列表，逗号或空格分隔"
    )
    parser.add_argument("--all", action="store_true", help="处理全部模型（默认行为）")
    parser.add_argument(
        "--source",
        choices=("modelscope", "huggingface"),
        default=None,
        help="强制指定下载源；缺省按配置顺序回退",
    )
    parser.add_argument(
        "--force", action="store_true", help="忽略本地已有目录，强制重新下载"
    )
    parser.add_argument(
        "--verify", action="store_true", help="执行 sha256 校验（仅当配置了 sha256）"
    )
    parser.add_argument(
        "--list", action="store_true", help="打印模型清单后退出（不联网）"
    )
    parser.add_argument(
        "--status", action="store_true", help="打印本地状态后退出（不联网）"
    )
    parser.add_argument("--remove", action="store_true", help="删除指定模型的本地文件")
    parser.add_argument(
        "--manifest-only", action="store_true", help="配合 --remove：只删 manifest"
    )
    parser.add_argument("--prune", action="store_true", help="清理过期下载暂存目录")
    parser.add_argument(
        "--prune-hours", type=int, default=None, help="暂存目录保留时长（默认取配置）"
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        default=True,
        help="单个模型失败后继续处理其余模型（默认开启）",
    )
    parser.add_argument(
        "--fail-fast",
        dest="keep_going",
        action="store_false",
        help="任一模型失败即停止",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    parser.add_argument("--quiet", action="store_true", help="不显示进度条")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="日志级别（-v INFO, -vv DEBUG）",
    )
    parser.add_argument(
        "--log-file", type=Path, default=None, help="额外写入轮转日志文件"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    configure_logging(level=level, log_file=args.log_file, force=True)

    try:
        manager = ModelManager(
            config_path=args.config,
            cache_root=args.cache_root,
            show_progress=not args.quiet,
        )
    except ModelManagerError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return _EXIT_USAGE

    if args.prune:
        return action_prune(manager, args)
    if args.remove:
        return action_remove(manager, args)
    if args.list:
        return action_list(manager, args)
    if args.status:
        return action_status(manager, args)
    return action_download(manager, args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n中断", file=sys.stderr)
        sys.exit(130)
