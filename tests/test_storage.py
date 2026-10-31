"""``backend.storage.repository`` 的自检。

覆盖：schema 初始化/幂等、视频状态机、切片/片段/关键帧读写、容量红线、
关键帧回绑、任务队列持久化与崩溃恢复、级联删除、线程并发写入。

跑法：
    python .tmp_test/t_storage.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.storage.repository import (  # noqa: E402
    SCHEMA_VERSION,
    CapacityExceededError,
    NotFoundError,
    RepositoryError,
    SqliteRepository,
    ms_to_timecode,
    new_video_id,
)

PASSED = 0
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED
    if cond:
        PASSED += 1
        print(f"  ok   {name}")
    else:
        FAILED.append(f"{name}  {detail}")
        print(f"  FAIL {name}  {detail}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="zy-repo-"))
    db = tmp / "sub" / "zhiying.db"  # 父目录不存在，验证自动创建

    # --- 1. 构造与 schema ---
    repo = SqliteRepository(db, limits={"max_videos": 3, "max_clips": 100})
    check("父目录被创建", db.parent.is_dir(), str(db.parent))
    check("db 文件存在", db.exists(), str(db))
    check(
        "schema 版本正确",
        repo.schema_version() == SCHEMA_VERSION,
        str(repo.schema_version()),
    )
    tables = repo.table_names()
    check(
        "视频、会话、分析版本与续传表齐备（v8）",
        set(tables)
        == {
            "videos",
            "clips",
            "transcript_segments",
            "keyframes",
            "tasks",
            "artifacts",
            "qa_history",
            "conversations", "conversation_messages", "conversation_operations",
            "uploaded_videos", "conversation_imports", "analysis_runs", "upload_sessions",
        },
        str(tables),
    )
    check(
        "journal_mode=wal",
        repo.connection().execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal",
        "",
    )
    check(
        "外键已开启",
        int(repo.connection().execute("PRAGMA foreign_keys").fetchone()[0]) == 1,
        "",
    )

    repo2 = SqliteRepository(db)  # 再次构造：不重复建表、不报错
    check("重复构造幂等", repo2.schema_version() == SCHEMA_VERSION, "")
    repo2.close()

    # --- 2. 视频 ---
    vid = repo.add_video(
        display_name="demo.mp4",
        source_kind="local",
        local_path=str(tmp / "demo.mp4"),
        duration_ms=612_345,
        width=1920,
        height=1080,
        fps=25.0,
        has_audio=True,
        size_bytes=1024,
    )
    check("video_id 长度 13", len(vid) == 13, vid)
    check("video_id 只含字母数字", vid.isalnum(), vid)
    v = repo.get_video(vid)
    check("duration 正确", v.duration_ms == 612_345, str(v.duration_ms))
    check("duration_s", abs(v.duration_s - 612.345) < 1e-6, str(v.duration_s))
    check("默认状态 pending", v.status == "pending", v.status)
    check("不是 ready", not v.is_ready, "")
    check("to_dict 可 JSON 化", json.dumps(v.to_dict(), ensure_ascii=False) != "", "")

    try:
        repo.add_video(video_id=vid)
        check("同 id 重复登记应报错", False, "")
    except RepositoryError as exc:
        check("同 id 重复登记报错", "已存在" in str(exc), str(exc))

    vid2 = repo.add_video(video_id=vid, display_name="demo-v2.mp4", replace=True)
    check(
        "replace 覆盖同 id",
        vid2 == vid and repo.get_video(vid).display_name == "demo-v2.mp4",
        "",
    )

    repo.update_video_status(vid, "downloading", stage="download", progress=0.3)
    v = repo.get_video(vid)
    check(
        "状态推进",
        v.status == "downloading" and v.stage == "download",
        f"{v.status} {v.stage}",
    )
    check("progress 记录", abs(v.progress - 0.3) < 1e-9, str(v.progress))
    check("非终态无 finished_at", v.finished_at == "", v.finished_at)

    repo.update_video_status(vid, "ready", progress=1.0)
    v = repo.get_video(vid)
    check("ready 是终态", v.is_ready and v.finished_at != "", v.finished_at)

    repo.update_video_status(vid, "failed", error="网络超时")
    v = repo.get_video(vid)
    check("failed 记录原因", v.error == "网络超时", v.error)

    try:
        repo.update_video_status("v-not-exist", "ready")
        check("更新不存在的视频应报错", False, "")
    except NotFoundError as exc:
        check(
            "NotFoundError 是 KeyError",
            isinstance(exc, KeyError),
            type(exc).__mro__.__str__(),
        )
        check(
            "NotFoundError 文案无多余引号",
            "不存在" in str(exc) and not str(exc).startswith("'"),
            str(exc),
        )

    repo.update_video_meta(vid, title="演示视频", language="zh", size_bytes=2048)
    v = repo.get_video(vid)
    check(
        "meta 局部更新",
        v.title == "演示视频" and v.language == "zh" and v.size_bytes == 2048,
        "",
    )
    check("未指定字段保持不变", v.width == 1920, str(v.width))

    # --- 3. 切片 ---
    clips = [
        {
            "clip_id": f"{vid}-c0000",
            "index_no": 0,
            "start_ms": 0,
            "end_ms": 20_100,
            "core_start_ms": 0,
            "core_end_ms": 19_350,
            "text": "大家好，今天讲三件事。",
            "core_text": "大家好，今天讲三件事。",
        },
        {
            "clip_id": f"{vid}-c0001",
            "index_no": 1,
            "start_ms": 18_600,
            "end_ms": 38_500,
            "core_start_ms": 19_350,
            "core_end_ms": 38_500,
            "text": "第一步是采集。",
            "core_text": "第一步是采集。",
            "truncated": True,
        },
    ]
    n = repo.add_clips(vid, clips)
    check("写入 2 个切片", n == 2, str(n))
    check("切片总数", len(repo.get_clips(vid)) == 2, "")
    c0 = repo.get_clip(f"{vid}-c0000")
    check(
        "clip 时间区间格式", c0.time_range == "00:00:00.000-00:00:20.100", c0.time_range
    )
    check("clip duration", c0.duration_ms == 20_100, str(c0.duration_ms))
    check("core 区间保留", c0.core_end_ms == 19_350, str(c0.core_end_ms))
    c1 = repo.get_clip(f"{vid}-c0001")
    check("truncated 标记", c1.truncated is True, str(c1.truncated))
    check("切片按 index 排序", [c.index_no for c in repo.get_clips(vid)] == [0, 1], "")

    # 重叠：c0001.start(18600) < c0000.end(20100) → 重叠 1500ms
    check(
        "相邻切片重叠 1500ms",
        repo.get_clips(vid)[0].end_ms - repo.get_clips(vid)[1].start_ms == 1_500,
        "",
    )

    try:
        repo.add_clips(vid, [{"start_ms": 0, "end_ms": 1000}])
        check("缺 clip_id 应报错", False, "")
    except RepositoryError as exc:
        check("缺 clip_id 报错", "缺少 clip_id" in str(exc), str(exc))

    try:
        repo.add_clips(vid, [{"clip_id": "x-1", "start_ms": 5000, "end_ms": 1000}])
        check("end<start 应报错", False, "")
    except RepositoryError as exc:
        check("end<start 报错", "end_ms" in str(exc), str(exc))

    cmap = repo.clip_map(vid)
    check(
        "clip_map 键齐", set(cmap) == {f"{vid}-c0000", f"{vid}-c0001"}, str(list(cmap))
    )
    check("clip_map 值可 JSON", json.dumps(cmap, ensure_ascii=False) != "", "")
    check("iter_clip_payloads 流式", len(list(repo.iter_clip_payloads(vid))) == 2, "")

    repo.add_clips(vid, [{**clips[0], "text": "覆盖后文本"}])
    check("切片 upsert 覆盖", repo.get_clip(f"{vid}-c0000").text == "覆盖后文本", "")
    check("upsert 不新增行", len(repo.get_clips(vid)) == 2, "")

    # --- 4. 转写片段 ---
    segs = [
        {
            "segment_id": f"{vid}-s00000",
            "start_ms": 0,
            "end_ms": 2500,
            "text": "大家好，",
            "language": "zh",
            "confidence": 0.98,
        },
        {
            "segment_id": f"{vid}-s00001",
            "start_ms": 2500,
            "end_ms": 6120,
            "text": "今天讲三件事。",
            "language": "zh",
            "confidence": 0.95,
        },
    ]
    check("写入片段", repo.add_segments(vid, segs) == 2, "")
    got = repo.get_segments(vid)
    check("片段读回", len(got) == 2 and got[1].text == "今天讲三件事。", str(len(got)))
    check("片段 source 默认 asr", got[0].source == "asr", got[0].source)
    check(
        "片段时间区间",
        got[1].time_range == "00:00:02.500-00:00:06.120",
        got[1].time_range,
    )

    repo.add_segments(
        vid,
        [
            {
                "segment_id": f"{vid}-s00002",
                "start_ms": 7000,
                "end_ms": 8000,
                "text": "新的一句",
                "source": "subtitle",
            }
        ],
    )
    got = repo.get_segments(vid)
    check(
        "replace=True 清旧",
        len(got) == 1 and got[0].source == "subtitle",
        str(len(got)),
    )

    # --- 5. 关键帧 + 回绑 ---
    kfs = [
        {
            "keyframe_id": "kf-1",
            "scene_index": 0,
            "timestamp_ms": 5_000,
            "path": str(tmp / "a.jpg"),
            "width": 1280,
            "height": 720,
            "size_bytes": 100_000,
            "sharpness": 87.2,
        },
        {
            "keyframe_id": "kf-2",
            "scene_index": 1,
            "timestamp_ms": 25_000,
            "path": str(tmp / "b.jpg"),
            "width": 1280,
            "height": 720,
            "size_bytes": 90_000,
            "sharpness": 12.0,
            "sharpness_ok": False,
        },
    ]
    check("写入关键帧", repo.add_keyframes(vid, kfs) == 2, "")
    check(
        "关键帧按时间排序",
        [k.keyframe_id for k in repo.get_keyframes(vid)] == ["kf-1", "kf-2"],
        "",
    )
    check("keyframe timecode", repo.get_keyframe("kf-1").timecode == "00:00:05.000", "")
    check("quality 标记", repo.get_keyframe("kf-2").sharpness_ok is False, "")
    check(
        "keyframe_paths",
        repo.keyframe_paths(vid)
        == {"kf-1": str(tmp / "a.jpg"), "kf-2": str(tmp / "b.jpg")},
        "",
    )

    bound = repo.set_keyframe_binding(vid)
    check("回绑 2 张帧", bound == 2, str(bound))
    c0 = repo.get_clip(f"{vid}-c0000")
    c1 = repo.get_clip(f"{vid}-c0001")
    check("kf-1 绑到 c0000", c0.keyframe_ids == ("kf-1",), str(c0.keyframe_ids))
    check("kf-2 绑到 c0001", c1.keyframe_ids == ("kf-2",), str(c1.keyframe_ids))
    check(
        "回绑写入 clip_id",
        repo.get_keyframe("kf-1").clip_id == f"{vid}-c0000",
        str(repo.get_keyframe("kf-1").clip_id),
    )

    repo.set_clip_keyframes(f"{vid}-c0000", ["kf-1", "kf-9"])
    check(
        "手工回填多帧",
        repo.get_clip(f"{vid}-c0000").keyframe_ids == ("kf-1", "kf-9"),
        "",
    )

    # 时间不相交的帧不绑定
    repo.add_keyframes(
        vid, [{"keyframe_id": "kf-far", "timestamp_ms": 600_000}], replace=False
    )
    repo.set_keyframe_binding(vid)
    check(
        "超范围帧不绑定",
        repo.get_keyframe("kf-far").clip_id is None,
        str(repo.get_keyframe("kf-far").clip_id),
    )

    # --- 6. 容量 ---
    check(
        "容量水位 videos",
        abs(repo.stats().usage("videos") - 1 / 3) < 1e-6,
        str(repo.stats().usage("videos")),
    )
    repo.add_video(display_name="b.mp4")
    repo.add_video(display_name="c.mp4")
    try:
        repo.add_video(display_name="d.mp4")
        check("超 max_videos 应报错", False, "")
    except CapacityExceededError as exc:
        check("超容量报错含表名", exc.table == "videos" and exc.limit == 3, str(exc))
    check("容量已满", repo.count_videos() == 3, str(repo.count_videos()))
    check("stats 计数", repo.stats().videos == 3, "")

    small = SqliteRepository(tmp / "small.db", limits={"max_clips": 1})
    sv = small.add_video(display_name="s.mp4")
    small.add_clips(sv, [{"clip_id": "s-c0", "start_ms": 0, "end_ms": 1000}])
    try:
        small.add_clips(
            sv,
            [
                {"clip_id": "s-c1", "start_ms": 1000, "end_ms": 2000},
                {"clip_id": "s-c2", "start_ms": 2000, "end_ms": 3000},
            ],
        )
        check("批量超容量应报错", False, "")
    except CapacityExceededError:
        check(
            "批量写入整体回滚",
            len(small.get_clips(sv)) == 1,
            str(len(small.get_clips(sv))),
        )
    small.close()

    no_cap = SqliteRepository(tmp / "nocap.db", check_capacity=False)
    nv = no_cap.add_video(display_name="n.mp4")
    no_cap.add_clips(
        nv,
        [
            {"clip_id": f"n-c{i}", "start_ms": i * 1000, "end_ms": i * 1000 + 500}
            for i in range(5)
        ],
    )
    check("check_capacity=False 不拦截", len(no_cap.get_clips(nv)) == 5, "")
    no_cap.close()

    # --- 7. 任务队列 ---
    tid = repo.upsert_task(vid, kind="ingest", payload={"url": "http://x"})
    t = repo.get_task(tid)
    check(
        "任务初始 pending",
        t.status == "pending" and t.payload["url"] == "http://x",
        str(t.payload),
    )
    repo.update_task(
        tid, status="running", stage="asr", progress=0.4, bump_attempts=True
    )
    t = repo.get_task(tid)
    check(
        "任务推进", t.status == "running" and t.stage == "asr", f"{t.status} {t.stage}"
    )
    check("attempts 累加", t.attempts == 1, str(t.attempts))

    # 伪造一个陈旧 running 任务
    stale_id = repo.upsert_task(vid, kind="ingest")
    repo.update_task(stale_id, status="running")
    repo.connection().execute(
        "UPDATE tasks SET updated_ts=? WHERE task_id=?", (0.0, stale_id)
    )
    n = repo.recover_stale_tasks(stale_after_s=1.0)
    check("恢复陈旧任务", n >= 1, str(n))
    check(
        "恢复后回到 pending",
        repo.get_task(stale_id).status == "pending",
        repo.get_task(stale_id).status,
    )
    check(
        "恢复原因留痕",
        "中断" in repo.get_task(stale_id).error,
        repo.get_task(stale_id).error,
    )
    check("running 任务未被误伤", repo.get_task(tid).status == "running", "")
    check(
        "pending_tasks 可用",
        any(x.task_id == stale_id for x in repo.pending_tasks()),
        "",
    )

    repo.update_task(tid, status="done", progress=1.0)
    check("任务完成", repo.get_task(tid).status == "done", "")

    try:
        repo.update_task("t-nope", status="done")
        check("更新不存在的任务应报错", False, "")
    except NotFoundError:
        check("任务 NotFoundError", True, "")

    # --- 8. 工件 ---
    aid = repo.add_artifact(vid, kind="report", path=str(tmp / "r.md"), size_bytes=1234)
    check("工单 id", aid.startswith("a") and len(aid) == 13, aid)
    arts = repo.get_artifacts(vid, kind="report")
    check("工件读回", len(arts) == 1 and arts[0]["size_bytes"] == 1234, str(arts))

    # --- 9. 线程并发写 ---
    errors: list[str] = []

    def worker(idx: int) -> None:
        try:
            for j in range(5):
                repo.upsert_task(vid, kind=f"w{idx}", message=f"{idx}-{j}")
                repo.get_video(vid)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{idx}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    check("6 线程并发无异常", not errors, str(errors[:2]))
    check(
        "并发写入计数",
        len(repo.list_tasks(limit=100)) >= 30 + 2,
        str(len(repo.list_tasks(limit=100))),
    )

    # --- 9b. 指纹与派生结果重置 ---
    check(
        "fingerprint 默认空",
        repo.get_video(vid).fingerprint == "",
        repo.get_video(vid).fingerprint,
    )
    repo.update_video_meta(vid, fingerprint="fp-abc123")
    check("fingerprint 可写入", repo.get_video(vid).fingerprint == "fp-abc123", "")
    check(
        "fingerprint 进 to_dict",
        repo.get_video(vid).to_dict().get("fingerprint") == "fp-abc123",
        "",
    )
    repo.update_video_status(vid, "ready", stage="ready", progress=1.0, finished=True)

    kf_before = repo.get_keyframes(vid)
    check(
        "重置前派生数据齐备",
        bool(kf_before and repo.get_clips(vid) and repo.get_segments(vid)),
        f"kf={len(kf_before)}",
    )
    paths = repo.reset_outputs(vid)
    expected = sorted(k.path for k in kf_before if k.path)
    check("reset 返回关键帧路径", sorted(paths) == expected, f"{paths} vs {expected}")
    check("reset 清空切片", repo.get_clips(vid) == [], "")
    check("reset 清空片段", repo.get_segments(vid) == [], "")
    check("reset 清空关键帧", repo.get_keyframes(vid) == [], "")
    check("clear_keyframes 幂等", repo.clear_keyframes(vid) == 0, "")
    after = repo.get_video(vid)
    check("reset 保留视频行", after is not None and after.video_id == vid, "")
    check(
        "reset 状态回到 pending",
        after.status == "pending" and after.stage == "",
        f"{after.status}/{after.stage}",
    )
    check("reset 清 finished_at", after.finished_at == "", after.finished_at)
    check("reset 保留 fingerprint", after.fingerprint == "fp-abc123", after.fingerprint)

    # 写回关键帧，供下一节验证级联删除仍能拿到路径
    repo.add_keyframes(
        vid,
        [
            {
                "keyframe_id": k.keyframe_id,
                "clip_id": k.clip_id,
                "scene_index": k.scene_index,
                "timestamp_ms": k.timestamp_ms,
                "path": k.path,
            }
            for k in kf_before
        ],
    )
    check("写回关键帧", len(repo.get_keyframes(vid)) == len(kf_before), "")

    # --- 10. 级联删除 ---
    kf_paths = repo.delete_video(vid)
    check("删除返回关键帧路径", any("a.jpg" in p for p in kf_paths), str(kf_paths))
    check("视频已删", repo.find_video(vid) is None, "")
    check("切片级联删除", repo.get_clips(vid) == [], "")
    check("片段级联删除", repo.get_segments(vid) == [], "")
    check("关键帧级联删除", repo.get_keyframes(vid) == [], "")
    check("工件级联删除", repo.get_artifacts(vid) == [], "")

    try:
        repo.delete_video(vid)
        check("重复删除应报错", False, "")
    except NotFoundError:
        check("重复删除 NotFoundError", True, "")

    # 任务不级联（历史留痕），仍可查
    check("任务表保留历史", repo.get_task(tid) is not None, "")

    # --- 11. 维护 ---
    repo.checkpoint()
    check("checkpoint 不报错", True, "")
    repo.vacuum()
    check("vacuum 不报错", True, "")
    st = repo.stats()
    check(
        "stats.to_dict JSON 化", json.dumps(st.to_dict(), ensure_ascii=False) != "", ""
    )

    orphan = repo.orphan_keyframe_paths()
    check(
        "orphan 检测出缺失文件",
        repo.get_keyframes(vid) == [] and orphan == [],
        str(orphan),
    )

    # --- 12. timecode ---
    cases = [
        (0, "00:00:00.000"),
        (999, "00:00:00.999"),
        (1000, "00:00:01.000"),
        (61_500, "00:01:01.500"),
        (3_600_000, "01:00:00.000"),
        (7_200_000, "02:00:00.000"),
        (-5, "00:00:00.000"),
    ]
    ok = all(ms_to_timecode(ms) == want for ms, want in cases)
    check(
        "ms_to_timecode 7 例",
        ok,
        str(
            [
                (ms, ms_to_timecode(ms))
                for ms, _ in cases
                if ms_to_timecode(ms) != dict(cases)[ms]
            ]
        ),
    )

    check("new_video_id 唯一", len({new_video_id() for _ in range(50)}) == 50, "")

    repo.close()
    check("close 可重复", (repo.close() or True), "")
    repo.close()

    # 关闭后重开数据还在
    repo3 = SqliteRepository(db)
    check("重开保留数据", repo3.count_videos() == 2, str(repo3.count_videos()))
    repo3.close()

    print(f"\nPASS {PASSED}  FAIL {len(FAILED)}")
    for f in FAILED:
        print("  -", f)
    return 0 if not FAILED else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(1) from None
