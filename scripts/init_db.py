"""初始化知影的数据目录与 SQLite 库：``python scripts/init_db.py``。

为什么要有这个脚本：``Orchestrator`` 第一次 analyze 时会自动建库，但"想让
环境准备好"和"想跑一次真实分析"是两件事 —— 答辩现场按这脚本跑一遍，能
立刻看出目录权限、schema 版本、容量上限是否正常，而不用等 30 分钟转写。

做四件事
--------
1. 按 ``configs/app_config.yaml`` 建齐数据目录（data/videos、audio、frames…）
2. 建库并写 ``PRAGMA user_version``（幂等，重复跑只检查不重建）
3. 打印每张表的行数与容量上限（对照需求文档表3）
4. ``--reset`` 时清空数据（**只删库文件与向量目录，不删用户视频**）

用法::

    python scripts/init_db.py              # 建库 + 报告
    python scripts/init_db.py --check      # 只检查，缺什么报什么
    python scripts/init_db.py --reset --yes   # 清库重建（危险，需二次确认）
    python scripts/init_db.py --json       # 机器可读
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

__all__ = ["main", "init_storage", "storage_report"]

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


def _human_bytes(size: int | float) -> str:
    value = float(size or 0)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    """等宽表格；中文按 2 列宽，否则 CJK 与 ASCII 混排会错位。"""

    def width(text: str) -> int:
        return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)

    cells = [[str(c) for c in row] for row in rows]
    widths = [width(h) for h in headers]
    for row in cells:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], width(cell))

    def pad(text: str, target: int) -> str:
        return text + " " * max(0, target - width(text))

    out = ["  ".join(pad(h, widths[i]) for i, h in enumerate(headers)).rstrip()]
    out.append("  ".join("-" * w for w in widths))
    for row in cells:
        out.append("  ".join(pad(c, widths[i]) for i, c in enumerate(row)).rstrip())
    return "\n".join(out)


def init_storage(config: Any, *, create_dirs: bool = True) -> dict[str, Any]:
    """建目录 + 建库。返回一份可打印的报告（不抛异常，问题写进 ``problems``）。"""
    report: dict[str, Any] = {"created_dirs": [], "problems": [], "database": ""}

    if create_dirs:
        try:
            config.ensure_dirs()
        except Exception as exc:  # noqa: BLE001
            report["problems"].append(f"建目录失败: {exc}")
        for key in ("data", "videos", "audio", "frames", "temp", "logs"):
            try:
                path = Path(config.path(key))
                path.mkdir(parents=True, exist_ok=True)
                report["created_dirs"].append(str(path))
            except Exception as exc:  # noqa: BLE001
                report["problems"].append(f"目录不可用 {key}: {exc}")

    try:
        from core.storage.repository import SqliteRepository  # noqa: PLC0415

        repo = SqliteRepository.from_config(config)
        report["database"] = str(getattr(repo, "db_path", config.path("sqlite")))
        report["schema_version"] = repo.schema_version()
        report["tables"] = list(repo.table_names())
        stats = repo.stats()
        report["stats"] = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
        repo.close()
    except Exception as exc:  # noqa: BLE001
        report["problems"].append(f"建库失败: {exc}")
    return report


def storage_report(config: Any) -> str:
    """人类可读的存储报告（表 / 当前 / 上限 / 占用）。"""
    from core.storage.repository import SqliteRepository  # noqa: PLC0415

    repo = SqliteRepository.from_config(config)
    try:
        stats = repo.stats()
        limits = dict(getattr(stats, "limits", {}) or {})
        rows: list[list[Any]] = []
        for key, label in (
            ("videos", "视频"),
            ("clips", "切片"),
            ("transcript_segments", "转录片段"),
            ("keyframes", "关键帧"),
        ):
            current = int(getattr(stats, key, 0) or 0)
            limit = int(limits.get(f"max_{key}", 0) or 0)
            ratio = f"{current / limit * 100:.1f}%" if limit else "-"
            rows.append([label, current, limit or "-", ratio])
        lines = [
            f"数据库: {getattr(repo, 'db_path', config.path('sqlite'))}",
            f"schema: v{repo.schema_version()}",
            "",
            _table(["表", "当前", "上限", "占用"], rows),
            "",
            f"关键帧占用: {_human_bytes(getattr(stats, 'keyframe_bytes', 0))}"
            f" / {_human_bytes(limits.get('max_keyframe_bytes', 0))}",
            f"数据库大小: {_human_bytes(getattr(stats, 'db_bytes', 0))}",
            f"任务: 排队 {int(getattr(stats, 'tasks_pending', 0) or 0)}"
            f" / 运行 {int(getattr(stats, 'tasks_running', 0) or 0)}",
        ]
        return "\n".join(lines)
    finally:
        repo.close()


def _reset(config: Any) -> list[str]:
    """删库文件与向量目录。**不动** videos/audio（那是用户的素材）。"""
    removed: list[str] = []
    db_path = Path(config.path("sqlite"))
    for candidate in (db_path, db_path.with_suffix(db_path.suffix + "-wal"),
                      db_path.with_suffix(db_path.suffix + "-shm")):
        if candidate.exists():
            candidate.unlink()
            removed.append(str(candidate))
    chroma_dir = Path(config.path("chroma"))
    if chroma_dir.exists() and chroma_dir.is_dir():
        shutil.rmtree(chroma_dir, ignore_errors=True)
        removed.append(str(chroma_dir))
    return removed


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(
        prog="init_db",
        description="初始化知影的数据目录与 SQLite 库（幂等）",
    )
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--check", action="store_true", help="只检查，不创建任何东西")
    parser.add_argument("--reset", action="store_true", help="清空数据库与向量库后重建")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过 --reset 的确认")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    from core.config import load_app_config  # noqa: PLC0415

    try:
        config = load_app_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"错误: 配置加载失败: {exc}", file=sys.stderr)
        return EXIT_FAIL

    try:
        config.validate()
    except Exception as exc:  # noqa: BLE001
        print(f"错误: 配置校验失败: {exc}", file=sys.stderr)
        return EXIT_FAIL

    if args.reset:
        db_path = Path(config.path("sqlite"))
        if not args.yes:
            answer = input(
                f"将删除 {db_path} 与向量目录 {config.path('chroma')}"
                f"（用户视频文件保留）。确认？[y/N] "
            ).strip().lower()
            if answer not in {"y", "yes"}:
                print("已取消")
                return EXIT_OK
        removed = _reset(config)
        if not args.json:
            for path in removed:
                print(f"已删除 {path}")
            print("")

    if args.check:
        report = init_storage(config, create_dirs=False)
    else:
        report = init_storage(config, create_dirs=True)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"项目根: {config.root_dir}")
        print(f"数据目录: {config.paths.data}")
        print(f"模型目录: {config.paths.models}")
        print(f"数据库: {report.get('database') or '-'}"
              f"（schema v{report.get('schema_version', 0)}）")
        print(f"表: {', '.join(report.get('tables') or []) or '-'}")
        print("")
        try:
            print(storage_report(config))
        except Exception as exc:  # noqa: BLE001
            print(f"（存储报告不可用: {exc}）")

    problems = report.get("problems") or []
    if problems:
        for problem in problems:
            print(f"错误: {problem}", file=sys.stderr)
        return EXIT_FAIL
    if not args.json:
        print("")
        print("✓ 存储已就绪")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
