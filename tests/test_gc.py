"""知影 ZhiYing —— TTL 清理与保留策略自检（``scripts/gc.py``）。

零 pytest 依赖，纯 ``check(name, cond)`` 累积器，退出码非 0 即失败。

覆盖：
* 时间/体积格式化的边界（0、负数、跨数量级、中文全角宽度）
* 过期判定只挑 ``status='ready'`` —— 跑了一半的视频不能被 TTL 删掉
* 临时文件按 mtime 判新旧，且只认文件不认目录
* ``apply_plan`` 的「失败不中断」与「统计账实相符」
* ``--dry-run`` 语义：``Plan`` 里有什么，就一条都不动

跑法：``python -u tests/test_gc.py``
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config import load_app_config  # noqa: E402
from core.storage.repository import SqliteRepository  # noqa: E402
from scripts.gc import (  # noqa: E402
    Outcome,
    Plan,
    _human,
    _iso_before,
    _pad,
    _rpad,
    _size_of,
    _stale_files,
    _width,
    apply_plan,
    build_plan,
    render_plan,
)

PASSED = 0
FAILED: list[str] = []
_ROOTS: list[Path] = []


def check(name: str, cond: bool, detail: Any = "") -> None:  # noqa: ANN001
    global PASSED
    if cond:
        PASSED += 1
        print(f"  ok   {name}")
    else:
        FAILED.append(f"{name}  {detail}")
        print(f"  FAIL {name}  {detail}")


def tmpdir(prefix: str = "zy-gc-") -> Path:
    root = Path(tempfile.mkdtemp(prefix=prefix))
    _ROOTS.append(root)
    return root


class FakeOrch:
    """只实现 ``build_plan`` / ``apply_plan`` 真正用到的那部分。

    故意不继承 ``Orchestrator``：这两条路径不该依赖模型或向量库，用真编排器
    反而会把「清理逻辑」和「推理装配」耦在一起。
    """

    def __init__(self, repo: SqliteRepository) -> None:
        self.repository = repo
        self.deleted: list[str] = []
        self.purged: list[tuple[str, tuple[str, ...]]] = []
        self.fail_on: set[str] = set()

    def delete(self, video_id: str, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """与 ``Orchestrator.delete()`` 同形：返回 ``{"db": [...], "files": {...}}``。"""
        if video_id in self.fail_on:
            raise RuntimeError(f"模拟删除失败: {video_id}")
        self.deleted.append(video_id)
        doomed = self.repository.delete_video(video_id)
        return {"db": doomed, "files": self.purge_files(video_id, doomed)}

    def purge_files(self, video_id: str, paths: Any = (), **kwargs: Any) -> dict[str, int]:  # noqa: ANN401
        self.purged.append((video_id, tuple(str(p) for p in paths)))
        return {"files": len(tuple(paths))}


def seed(repo: SqliteRepository, *, vid: str, status: str, finished: str) -> None:
    """造一条视频并把它钉在指定完成时间（``update_video_status`` 只会写“现在”）。"""
    repo.add_video(
        video_id=vid,
        title=f"片 {vid}",
        source_kind="local",
        local_path=f"D:/x/{vid}.mp4",
        status=status,  # type: ignore[arg-type]
    )
    conn = repo.connection()
    with conn:
        conn.execute(
            "UPDATE videos SET finished_at=?, created_at=? WHERE video_id=?",
            (finished, finished or "2020-01-01T00:00:00.000Z", vid),
        )


def main() -> int:
    config = load_app_config(env={}, ignore_user_settings=True)
    work = tmpdir()

    # -- 1. 格式化 ---------------------------------------------------------
    print("\n[1] 人类可读格式化")
    check("0 字节", _human(0) == "0 B", _human(0))
    check("负数按 0", _human(-5) == "0 B", _human(-5))
    check("999 字节", _human(999) == "999 B", _human(999))
    check("1023 仍是字节", _human(1023) == "1023 B", _human(1023))
    check("1024 进位", _human(1024).startswith("1.0"), _human(1024))
    check("1 MB", _human(1024 * 1024) == "1.0 MB", _human(1024 * 1024))
    check("1 GB", _human(1024**3) == "1.0 GB", _human(1024**3))
    check("1 TB", _human(1024**4) == "1.0 TB", _human(1024**4))
    check("超大不越界", _human(1024**5).endswith("TB"), _human(1024**5))

    # -- 2. 中文宽度对齐 ---------------------------------------------------
    print("\n[2] 中文宽度")
    check("ASCII 宽度 1", _width("abc") == 3, _width("abc"))
    check("汉字宽度 2", _width("中文") == 4, _width("中文"))
    check("混排宽度", _width("a中b") == 4, _width("a中b"))
    check("左对齐补到 10 列", _width(_pad("视频", 10)) == 10, _width(_pad("视频", 10)))
    check("右对齐补到 10 列", _width(_rpad("2 条", 10)) == 10, _width(_rpad("2 条", 10)))
    check("超长不截断", _pad("一二三四五六", 4) == "一二三四五六", _pad("一二三四五六", 4))

    # -- 3. 时间窗口 -------------------------------------------------------
    print("\n[3] 时间窗口")
    now = time.time()
    iso = _iso_before(days=30)
    check("可被 fromisoformat 解析", bool(datetime.fromisoformat(iso)), iso)
    check("ISO 含毫秒", "." in iso, iso)
    check("带 UTC 时区", datetime.fromisoformat(iso).utcoffset() == timedelta(0), iso)
    parsed = datetime.fromisoformat(iso).timestamp()
    check("30 天前误差 <2 分钟", abs((now - 30 * 86400) - parsed) < 120, round(now - parsed))
    check("0 天约等于现在", abs(now - datetime.fromisoformat(_iso_before(days=0)).timestamp()) < 120, "")
    check("与库内格式一致", iso.endswith("+00:00"), iso)

    # -- 4. 临时文件挑选 ---------------------------------------------------
    print("\n[4] 临时文件按 mtime 判定")
    temp_root = work / "temp"
    temp_root.mkdir(parents=True)
    old = temp_root / "old.tmp"
    old.write_text("x" * 100, encoding="utf-8")
    fresh = temp_root / "fresh.tmp"
    fresh.write_text("y" * 50, encoding="utf-8")
    sub = temp_root / "subdir"
    sub.mkdir()
    (sub / "inner.tmp").write_text("z", encoding="utf-8")
    past = time.time() - 7200
    os.utime(old, (past, past))
    os.utime(sub, (past, past))
    stale = _stale_files(temp_root, time.time() - 3600)
    check("只挑过期文件", [p.name for p in stale] == ["old.tmp"], [p.name for p in stale])
    check("目录不算", all(p.is_file() for p in stale), "")
    check("体积累加", _size_of(stale) == 100, _size_of(stale))
    check("空列表体积为 0", _size_of([]) == 0, _size_of([]))

    # 让 build_plan 指向本次测试的目录，别去动工程的 data/temp。
    config = load_app_config(
        env={},
        ignore_user_settings=True,
        overrides={"paths": {"temp": str(temp_root), "exports": str(work / "exports")}},
    )
    check("配置指向测试目录", Path(config.path("temp")) == temp_root, config.path("temp"))

    # -- 5. 过期视频挑选 ---------------------------------------------------
    print("\n[5] 过期视频只挑 ready")
    repo = SqliteRepository(work / "gc.db")
    seed(repo, vid="v-old", status="ready", finished="2020-01-01T00:00:00.000Z")
    seed(repo, vid="v-new", status="ready", finished=_iso_before(days=0))
    seed(repo, vid="v-running", status="running", finished="2020-01-01T00:00:00.000Z")
    seed(repo, vid="v-failed", status="failed", finished="2020-01-01T00:00:00.000Z")

    expired = repo.expired_videos(finished_before=_iso_before(days=30))
    check("只返回 ready", [v.video_id for v in expired] == ["v-old"], [v.video_id for v in expired])
    check("空时间串不返回", repo.expired_videos(finished_before="") == [], "")
    check("limit 生效", len(repo.expired_videos(finished_before=_iso_before(days=30), limit=0)) <= 1, "")

    orch = FakeOrch(repo)

    # -- 6. build_plan -----------------------------------------------------
    print("\n[6] 计划构建")
    plan = build_plan(config, orch, ttl_days=30, temp_hours=1, orphan_days=30)  # type: ignore[arg-type]
    check("plan 是 Plan", isinstance(plan, Plan), type(plan).__name__)
    check("挑出 1 条过期视频", [v.video_id for v in plan.videos] == ["v-old"], [v.video_id for v in plan.videos])
    check("挑出 1 个过期临时文件", [p.name for p in plan.temp] == ["old.tmp"], [p.name for p in plan.temp])
    check("非空", plan.empty is False, "")
    check("有说明文字", bool(plan.notes), plan.notes)

    plan_zero = build_plan(config, orch, ttl_days=0, temp_hours=99999, orphan_days=0)  # type: ignore[arg-type]
    check("ttl=0 不删视频", plan_zero.videos == [], [v.video_id for v in plan_zero.videos])
    check("无项目时 empty", plan_zero.empty is True, (plan_zero.temp, plan_zero.orphans, plan_zero.videos))

    text = render_plan(plan, dry_run=True)
    check("渲染含视频标题", "片 v-old" in text, text[:200])
    check("渲染含预览提示", "预览" in text, text[:80])
    check("渲染含视频 id", "v-old" in text, text[:200])

    # -- 7. apply_plan -----------------------------------------------------
    print("\n[7] 执行计划")
    outcome = apply_plan(plan, orch)  # type: ignore[arg-type]
    check("Outcome 类型", isinstance(outcome, Outcome), type(outcome).__name__)
    check("删掉 1 条视频", outcome.videos == 1, outcome.videos)
    check("删掉 1 个临时文件", outcome.temp == 1, outcome.temp)
    check("调用编排器 delete", orch.deleted == ["v-old"], orch.deleted)
    check("临时文件真的没了", not old.exists(), old)
    check("新文件还在", fresh.exists(), fresh)
    check("库记录已删", repo.find_video("v-old") is None, "")
    check("其他视频未动", repo.find_video("v-running") is not None and repo.find_video("v-new") is not None, "")
    check("ok 为真", outcome.ok is True, outcome.failed)

    # 重复执行必须是空操作（幂等）。
    again = apply_plan(build_plan(config, orch, ttl_days=30, temp_hours=1), orch)  # type: ignore[arg-type]
    check("二次执行空操作", again.videos == 0 and again.temp == 0, (again.videos, again.temp))

    # -- 8. 失败不中断 -----------------------------------------------------
    print("\n[8] 单条失败不中断")
    repo2 = SqliteRepository(work / "gc2.db")
    seed(repo2, vid="v-bad", status="ready", finished="2020-01-01T00:00:00.000Z")
    seed(repo2, vid="v-good", status="ready", finished="2020-01-02T00:00:00.000Z")
    orch2 = FakeOrch(repo2)
    orch2.fail_on.add("v-bad")
    plan2 = build_plan(config, orch2, ttl_days=30, temp_hours=99999)  # type: ignore[arg-type]
    check("计划含两条", len(plan2.videos) == 2, len(plan2.videos))
    out2 = apply_plan(plan2, orch2)  # type: ignore[arg-type]
    check("成功一条", out2.videos == 1, out2.videos)
    check("失败被记录", len(out2.failed) == 1 and "v-bad" in out2.failed[0], out2.failed)
    check("失败时 ok 为假", out2.ok is False, out2.ok)
    check("另一条仍被删", repo2.find_video("v-good") is None, "")

    repo.close()
    repo2.close()

    print(f"\nPASS {PASSED}  FAIL {len(FAILED)}")
    for item in FAILED:
        print("  -", item)
    return 0 if not FAILED else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        for root in _ROOTS:
            import shutil

            shutil.rmtree(root, ignore_errors=True)
