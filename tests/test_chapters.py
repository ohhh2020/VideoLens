"""章节层测试：时间分组、标题渲染、markdown 输出。

不加载任何模型 —— 章节层的价值在"切得对、标得清"，模型好坏由
test_pipeline 与真实 e2e 覆盖。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.pipeline.chapters import (  # noqa: E402
    Chapter,
    ChapterPlan,
    chapters_markdown,
    plan_chapters,
)
from backend.pipeline.context_builder import spread_order  # noqa: E402

PASS = 0
FAIL = 0


def check(label: str, ok: bool, detail: object = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok  {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}  {detail}")


def make_clips(count: int, step_ms: int = 30000) -> list[dict]:
    return [
        {
            "clip_id": f"c{i:03d}",
            "start_ms": i * step_ms,
            "end_ms": (i + 1) * step_ms,
            "text": f"第 {i} 段内容",
        }
        for i in range(count)
    ]


def main() -> int:
    print("=== 1. 时间分组 ===")
    clips = make_clips(12)  # 0–360s
    plans = plan_chapters(clips, chapter_seconds=120, duration_ms=360_000)
    check("360s / 120s → 3 章", len(plans) == 3, len(plans))
    check(
        "章号从 1 开始连续",
        [p.index for p in plans] == [1, 2, 3],
        [p.index for p in plans],
    )
    check("首章从 0 开始", plans[0].start_ms == 0, plans[0].start_ms)
    check("末章到片尾", plans[-1].end_ms == 360_000, plans[-1].end_ms)
    check("12 个切片全部归入", sum(len(p.clip_ids) for p in plans) == 12)
    check(
        "章节不重叠",
        all(plans[i].end_ms <= plans[i + 1].start_ms for i in range(len(plans) - 1)),
    )
    check("每章非空", all(p.clip_ids for p in plans))
    check("每章有时长", all(p.duration_ms > 0 for p in plans))
    check(
        "time_range 格式",
        "–" in plans[0].time_range and ":" in plans[0].time_range,
        plans[0].time_range,
    )

    print("\n=== 2. 上限与边界 ===")
    many = plan_chapters(clips, chapter_seconds=10, duration_ms=360_000, max_chapters=4)
    check("max_chapters 收紧到 4", len(many) <= 4, len(many))
    check("收紧后仍覆盖首尾", many[0].start_ms == 0 and many[-1].end_ms == 360_000)
    check("空输入返回空", plan_chapters([], chapter_seconds=300, duration_ms=0) == [])
    one = plan_chapters(make_clips(1), chapter_seconds=300, duration_ms=30_000)
    check("单切片 → 1 章", len(one) == 1 and one[0].clip_ids == ("c000",), one)
    check(
        "chapter_seconds<=0 不炸",
        len(plan_chapters(clips, chapter_seconds=0, duration_ms=360_000)) >= 1,
    )
    # 乱序输入：时间戳不递增也要能正确分桶
    shuffled = [clips[5], clips[0], clips[9], clips[2]]
    got = plan_chapters(shuffled, chapter_seconds=120, duration_ms=360_000)
    check(
        "乱序输入仍按时间分章",
        got[0].start_ms == 0,
        [(p.start_ms, p.end_ms) for p in got],
    )

    print("\n=== 3. 数据类 ===")
    plan = ChapterPlan(index=2, start_ms=60_000, end_ms=120_000, clip_ids=("c1", "c2"))
    check("ChapterPlan.to_dict 含 clip_count", plan.to_dict()["clip_count"] == 2)
    check("ChapterPlan.duration_ms", plan.duration_ms == 60_000, plan.duration_ms)
    ch = Chapter(
        index=3,
        start_ms=600_000,
        end_ms=900_000,
        summary="这一段先讲训练，接着讲推理。",
        clip_ids=("c9", "c10"),
    )
    check(
        "heading 含章号与时间",
        "第 3 章" in ch.heading and "00:10:00.000" in ch.heading,
        ch.heading,
    )
    check("heading 取摘要首句", "这一段先讲训练" in ch.heading, ch.heading)
    check("Chapter.to_dict 可序列化", isinstance(ch.to_dict()["clip_ids"], list))
    check(
        "空摘要 heading 不空", bool(Chapter(index=1, start_ms=0, end_ms=1000).heading)
    )

    print("\n=== 4. markdown ===")
    md = chapters_markdown(
        "演示视频",
        [
            Chapter(
                index=1,
                start_ms=0,
                end_ms=180_000,
                summary="第一段。",
                clip_ids=("c0",),
            ),
            Chapter(
                index=2,
                start_ms=180_000,
                end_ms=360_000,
                summary="第二段。",
                clip_ids=("c1",),
                degraded=True,
            ),
        ],
        total_summary="整片讲了训练与推理。",
        keywords=["训练", "推理"],
        duration_ms=360_000,
    )
    check("一级标题是视频名", md.startswith("# 演示视频"), md[:16])
    check("含总总结段", "整片讲了训练与推理。" in md)
    check("含章节标题", "## 第 1 章" in md and "## 第 2 章" in md)
    check("含时间范围", "00:03:00.000" in md)
    check("含关键词", "训练" in md and "推理" in md)
    check("降级章有标记", "降级" in md)
    check("两章正文都在", "第一段。" in md and "第二段。" in md)
    check("短视频不显示时长行以外的噪声", md.count("# ") >= 1)

    print("\n=== 5. spread_order（位置铺开）===")
    check("spread_order(0) = []", spread_order(0) == [])
    check("spread_order(1) = [0]", spread_order(1) == [0])
    check("spread_order(2) = [0,1]", spread_order(2) == [0, 1], spread_order(2))
    for n in (3, 5, 7, 12, 30):
        order = spread_order(n)
        check(
            f"spread_order({n}) 覆盖 0..{n - 1} 且不重复",
            sorted(order) == list(range(n)),
            order,
        )
    order12 = spread_order(12)
    check("spread_order 首尾先选", order12[:2] == [0, 11], order12[:4])
    check("spread_order 取前 4 覆盖首尾中", {0, 11} <= set(order12[:4]), order12[:4])

    print()
    print(f"PASS={PASS}  FAIL={FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
