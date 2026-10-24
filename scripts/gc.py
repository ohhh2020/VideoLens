"""知影 ZhiYing —— 过期数据回收。

按 ``storage.retention`` 的配置清理四类东西：

1. **临时文件** —— ``data/temp`` 下超过 ``temp_hours`` 的下载残片、字幕中转。
2. **孤儿关键帧** —— 磁盘上有、库里已无对应记录的 JPG（分析失败或旧版本遗留）。
3. **过期视频** —— 分析完成超过 ``video_ttl_days`` 天的整条记录：元数据、
   切片、关键帧、音频、向量、报告一起删。
4. **容量回收**（``--enforce-limits``）—— 触及需求文档表 3 的红线时，从最久
   完成的分析开始删，直到降到 80% 以下，而不是让下一个任务直接失败。

服务进程不做定时删除（用户可能正在看某个旧视频），所以这里是独立的运维脚本，
由启动器或计划任务按需调用：

    python scripts/gc.py --stats              # 只看占用，不动数据
    python scripts/gc.py --dry-run            # 列出会删什么
    python scripts/gc.py                      # 按 app_config.yaml 执行
    python scripts/gc.py --ttl-days 7         # 临时覆盖保留天数
    python scripts/gc.py --enforce-limits     # 容量红线回收

退出码：0 正常、1 有清理动作失败、2 参数或配置错误。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import AppConfig, ConfigError, get_app_config  # noqa: E402
from core.pipeline.orchestrator import Orchestrator  # noqa: E402
from core.storage.repository import StorageError  # noqa: E402

logger = logging.getLogger("zhiying.gc")

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

RECLAIM_TARGET = 0.80  # 容量回收降到 80% 水位即停，别把库删空

#: ``StorageStats`` 里参与容量判定的表，与 ``repository._LIMIT_MAP`` 一致。
_LIMIT_TABLES = ("videos", "clips", "transcript_segments", "keyframes")
_TABLE_OF = {
    "max_videos": "videos",
    "max_clips": "clips",
    "max_transcript_segments": "transcript_segments",
    "max_keyframes": "keyframes",
}


# -----------------------------------------------------------------------------
# 数据结构
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class Plan:
    """一次回收的完整计划。先算清楚再动手，``--dry-run`` 就是只算不动。"""

    temp: list[Path] = field(default_factory=list)
    orphans: list[Path] = field(default_factory=list)
    videos: list[Any] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def temp_bytes(self) -> int:
        return _size_of(self.temp)

    @property
    def orphan_bytes(self) -> int:
        return _size_of(self.orphans)

    @property
    def video_bytes(self) -> int:
        return sum(int(getattr(v, "size_bytes", 0) or 0) for v in self.videos)

    @property
    def total_bytes(self) -> int:
        return self.temp_bytes + self.orphan_bytes + self.video_bytes

    @property
    def empty(self) -> bool:
        return not (self.temp or self.orphans or self.videos)


@dataclass(slots=True)
class Outcome:
    temp: int = 0
    orphans: int = 0
    videos: int = 0
    freed: int = 0
    failed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed


# -----------------------------------------------------------------------------
# 工具
# -----------------------------------------------------------------------------


def _size_of(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _width(text: str) -> int:
    """终端显示宽度：中日韩全角字符占 2 列。"""
    return sum(2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def _rpad(text: str, width: int) -> str:
    return " " * max(0, width - _width(text)) + text


def _human(size: int) -> str:
    value = float(max(0, int(size)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def _iso_before(*, days: float = 0.0, hours: float = 0.0) -> str:
    """N 天（或小时）之前的 UTC ISO 时间串，与 ``videos.finished_at`` 同格式。"""
    delta = timedelta(days=days, hours=hours)
    return (datetime.now(timezone.utc) - delta).isoformat(timespec="milliseconds")


def _stale_files(root: Path, cutoff_epoch: float) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff_epoch:
                out.append(path)
        except OSError:
            continue
    return out


# -----------------------------------------------------------------------------
# 计划
# -----------------------------------------------------------------------------


def build_plan(
    config: AppConfig,
    orch: Orchestrator,
    *,
    ttl_days: int | None = None,
    temp_hours: int | None = None,
    orphan_days: int | None = None,
    enforce_limits: bool = False,
) -> Plan:
    """算出这次要删什么。所有判断只读不写。"""
    retention = config.storage.retention
    plan = Plan()

    hours = retention.temp_hours if temp_hours is None else temp_hours
    if hours > 0:
        cutoff = time.time() - hours * 3600
        plan.temp = _stale_files(Path(config.path("temp")), cutoff)
    else:
        plan.notes.append("临时文件：temp_hours=0，跳过")

    days = retention.orphan_frames_days if orphan_days is None else orphan_days
    if days > 0:
        cutoff = time.time() - days * 86400
        try:
            candidates = [Path(p) for p in orch.repository.orphan_keyframe_paths()]
        except StorageError as exc:
            plan.notes.append(f"孤儿关键帧：查询失败（{exc}），跳过")
            candidates = []
        plan.orphans = [p for p in candidates if p.is_file() and p.stat().st_mtime < cutoff]
    else:
        plan.notes.append("孤儿关键帧：orphan_frames_days=0，跳过")

    ttl = retention.video_ttl_days if ttl_days is None else ttl_days
    if ttl > 0:
        plan.videos = orch.repository.expired_videos(finished_before=_iso_before(days=ttl))
        if plan.videos:
            plan.notes.append(f"视频 TTL：完成超过 {ttl} 天的记录 {len(plan.videos)} 条")
    else:
        plan.notes.append("视频 TTL：video_ttl_days=0，跳过")

    if enforce_limits:
        extra = _over_limit_videos(config, orch)
        seen = {v.video_id for v in plan.videos}
        fresh = [v for v in extra if v.video_id not in seen]
        if fresh:
            plan.videos.extend(fresh)
            plan.notes.append(f"容量回收：追加最久完成的 {len(fresh)} 条")

    return plan


def _over_limit_videos(config: AppConfig, orch: Orchestrator) -> list[Any]:
    """容量触顶时按最久完成顺序补删，直到预估降到 ``RECLAIM_TARGET`` 水位。"""
    if not config.storage.retention.auto_prune:
        return []
    try:
        stats = orch.repository.stats()
    except StorageError as exc:
        logger.warning("读取容量水位失败: %s", exc)
        return []
    worst = max((stats.usage(name) for name in _LIMIT_TABLES), default=0.0)
    if worst <= RECLAIM_TARGET:
        return []
    # 需要腾出的比例 → 换算成"删掉最老的一批"，按完成时间升序取够为止。
    need = worst - RECLAIM_TARGET
    pool = orch.repository.expired_videos(finished_before=_iso_before(days=0), limit=10_000)
    if not pool:
        return []
    take = max(1, int(len(pool) * need / max(worst, 1e-6)))
    logger.info("容量水位 %.0f%%，计划回收 %d 条（%d 条可选）", worst * 100, take, len(pool))
    return pool[:take]


# -----------------------------------------------------------------------------
# 执行
# -----------------------------------------------------------------------------


def apply_plan(plan: Plan, orch: Orchestrator) -> Outcome:
    """按计划删除。单项失败不中断，全部记进 ``Outcome.failed``。"""
    out = Outcome()
    for path in plan.temp:
        try:
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            out.temp += 1
            out.freed += size
        except OSError as exc:
            out.failed.append(f"临时文件 {path.name}: {exc}")
    for path in plan.orphans:
        try:
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            out.orphans += 1
            out.freed += size
        except OSError as exc:
            out.failed.append(f"孤儿帧 {path.name}: {exc}")
    for video in plan.videos:
        video_id = str(getattr(video, "video_id", ""))
        try:
            result = orch.delete(video_id)
        except Exception as exc:  # noqa: BLE001 - 逐条报告，不因一条失败全停
            out.failed.append(f"视频 {video_id}: {exc}")
            continue
        if result.get("db_error"):
            out.failed.append(f"视频 {video_id}: {result['db_error']}")
            continue
        out.videos += 1
        out.freed += int(getattr(video, "size_bytes", 0) or 0)
    return out


# -----------------------------------------------------------------------------
# 报告
# -----------------------------------------------------------------------------


def render_stats(config: AppConfig, orch: Orchestrator) -> str:
    lines = ["知影 数据占用", ""]
    try:
        stats = orch.repository.stats()
    except StorageError as exc:
        lines.append(f"  数据库统计失败：{exc}")
    else:
        rows = (
            ("视频", stats.videos, "max_videos"),
            ("切片", stats.clips, "max_clips"),
            ("转写片段", stats.transcript_segments, "max_transcript_segments"),
            ("关键帧", stats.keyframes, "max_keyframes"),
        )
        for label, count, key in rows:
            limit = int(stats.limits.get(key, 0) or 0)
            cap = f" / {limit}" if limit else ""
            ratio = f"  {stats.usage(_TABLE_OF[key]) * 100:>5.1f}%" if limit else ""
            lines.append(f"  {_pad(label, 10)} {_rpad(str(count), 8)}{cap}{ratio}")
        lines.append(f"  {_pad('关键帧体积', 10)} {_rpad(_human(stats.keyframe_bytes), 8)}")
        lines.append(f"  {_pad('数据库体积', 10)} {_rpad(_human(stats.db_bytes), 8)}")
    lines.append("")
    for tag in ("videos", "audio", "frames", "temp", "chroma", "exports"):
        root = Path(config.path(tag))
        if not root.exists():
            continue
        size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        count = sum(1 for p in root.rglob("*") if p.is_file())
        lines.append(f"  {_pad(tag, 10)} {_rpad(str(count), 7)} 文件  {_rpad(_human(size), 10)}")
    return "\n".join(lines)


def render_plan(plan: Plan, *, dry_run: bool) -> str:
    head = "知影 回收计划（预览，未改动任何数据）" if dry_run else "知影 回收计划"
    lines = [head, ""]
    if plan.temp:
        lines.append(f"  临时文件      {len(plan.temp):>4} 个   {_human(plan.temp_bytes)}")
    if plan.orphans:
        lines.append(f"  孤儿关键帧    {len(plan.orphans):>4} 张   {_human(plan.orphan_bytes)}")
    if plan.videos:
        lines.append(f"  过期视频      {len(plan.videos):>4} 条   {_human(plan.video_bytes)}")
        # 预览阶段就把标题列出来：删之前总得知道删的是哪几条。
        for video in plan.videos[:3]:
            name = (video.title or video.display_name or "").strip() or "(无标题)"
            lines.append(f"    · {_pad(name, 24)}  {video.video_id}  {video.finished_at}")
        if len(plan.videos) > 3:
            lines.append(f"    · 其余 {len(plan.videos) - 3} 条略")
    if plan.empty:
        lines.append("  没有需要清理的数据")
    else:
        lines.append("")
        lines.append(f"  合计可释放约 {_human(plan.total_bytes)}")
    for note in plan.notes:
        lines.append(f"  · {note}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gc",
        description="知影：按 storage.retention 清理过期数据",
    )
    parser.add_argument("--dry-run", action="store_true", help="只列出计划，不删除")
    parser.add_argument("--stats", action="store_true", help="只显示当前占用")
    parser.add_argument("--ttl-days", type=int, default=None, help="覆盖视频保留天数")
    parser.add_argument("--temp-hours", type=int, default=None, help="覆盖临时文件保留小时数")
    parser.add_argument("--orphan-days", type=int, default=None, help="覆盖孤儿关键帧保留天数")
    parser.add_argument(
        "--enforce-limits", action="store_true", help="容量触顶时按最久完成回收"
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="日志详细度")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.WARNING if args.verbose == 0 else logging.INFO
    if args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    try:
        config = get_app_config()
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        orch = Orchestrator.from_config(config, load_models=False)
    except Exception as exc:  # noqa: BLE001 - 存储层起不来就没法清理
        print(f"初始化失败：{exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        if args.stats:
            print(render_stats(config, orch))
            return EXIT_OK

        plan = build_plan(
            config,
            orch,
            ttl_days=args.ttl_days,
            temp_hours=args.temp_hours,
            orphan_days=args.orphan_days,
            enforce_limits=args.enforce_limits,
        )
        if args.json:
            payload = {
                "dry_run": bool(args.dry_run),
                "temp": [str(p) for p in plan.temp],
                "orphans": [str(p) for p in plan.orphans],
                "videos": [str(v.video_id) for v in plan.videos],
                "bytes": plan.total_bytes,
                "notes": plan.notes,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(render_plan(plan, dry_run=args.dry_run))

        if args.dry_run or plan.empty:
            return EXIT_OK

        outcome = apply_plan(plan, orch)
        if args.json:
            print(
                json.dumps(
                    {
                        "removed": {
                            "temp": outcome.temp,
                            "orphans": outcome.orphans,
                            "videos": outcome.videos,
                        },
                        "freed_bytes": outcome.freed,
                        "failed": outcome.failed,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print()
            print(
                f"已清理：临时 {outcome.temp} / 孤儿帧 {outcome.orphans} / "
                f"视频 {outcome.videos}，释放 {_human(outcome.freed)}"
            )
            for item in outcome.failed:
                print(f"  失败：{item}", file=sys.stderr)
        return EXIT_OK if outcome.ok else EXIT_FAIL
    finally:
        orch.close()


if __name__ == "__main__":
    raise SystemExit(main())
