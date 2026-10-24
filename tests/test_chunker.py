"""chunker 模块自检：覆盖正常语音流与各类边界情况。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.rag.chunker import (  # noqa: E402
    ChunkConfig,
    Chunker,
    ChunkError,
    RawSegment,
    Word,
    bind_keyframes,
    segments_from_whisper,
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


def _rejects(fn, exc: type[BaseException] = ValueError) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def segs(spec: list[tuple[int, int, str]]) -> list[dict]:
    return [
        {"id": i, "start": s / 1000.0, "end": e / 1000.0, "text": t}
        for i, (s, e, t) in enumerate(spec)
    ]


chunker = Chunker()

# ---------------------------------------------------------------- 1) 正常流
# 说话 12s（两句，句末有 0.9s 静音），再 10s（两句），再 9s
normal = segs([
    (0, 6_000, "大家好，今天我们来讲一下本地视频分析系统的架构设计。"),
    (6_900, 12_000, "首先看媒体处理层，它负责把视频拆成切片。"),
    (12_900, 18_000, "第二层是模型推理层，包括语音识别和视觉理解。"),
    (18_000, 22_000, "模型权重全部由管理器自动下载。"),
    (23_000, 28_000, "第三层是向量检索，支持文本和图像双模态。"),
    (28_000, 32_000, "检索结果会带上毫秒级时间戳。"),
])
clips = chunker.chunk("v1", normal)
check("正常流产出切片", 2 <= len(clips) <= 6, f"{len(clips)} 片 {[c.core_duration_ms for c in clips]}")
check("切片时长 10~30s", all(10_000 <= c.core_duration_ms <= 30_000 for c in clips),
      str([c.core_duration_ms for c in clips]))
check("切片无空洞且连续", all(clips[i].core_end_ms == clips[i + 1].core_start_ms
                          for i in range(len(clips) - 1)),
      str([(c.core_start_ms, c.core_end_ms) for c in clips]))
check("首尾覆盖转录", clips[0].core_start_ms == 0 and clips[-1].core_end_ms >= 32_000,
      f"{clips[0].core_start_ms}~{clips[-1].core_end_ms}")
check("clip_id 规则", all(c.clip_id == f"v1-c{i:04d}" for i, c in enumerate(clips)),
      clips[0].clip_id)
check("溯源字段齐全", all(c.video_id == "v1" and c.start_time and c.end_time for c in clips))
check("时间格式 HH:MM:SS.mmm", clips[0].start_time.count(":") == 2 and "." in clips[0].start_time,
      clips[0].start_time)
check("文本非空", all(c.text.strip() for c in clips))

# ---------------------------------------------------- 2) 1.5s 边界重叠
check("相邻切片重叠 1.5s", all(
    abs(clips[i].end_ms - clips[i + 1].start_ms - 1_500) <= 60 for i in range(len(clips) - 1)
), str([clips[i].end_ms - clips[i + 1].start_ms for i in range(len(clips) - 1)]))
check("重叠不越界", clips[0].start_ms <= clips[0].core_start_ms and clips[-1].end_ms >= clips[-1].core_end_ms)
check("外扩区间包含核心区间", all(c.start_ms <= c.core_start_ms and c.end_ms >= c.core_end_ms for c in clips))

# ---------------------------------------- 3) 静音点优先（>0.6s 才算切点）
# 段间只有 0.3s 的间隔不应成为切点，0.9s 应成为切点
tight = segs([(0, 5_000, "第一句话说完了。"), (5_300, 10_000, "第二句话接着来了。"),
              (10_900, 16_000, "第三句话隔得比较久。"), (16_000, 21_000, "第四句紧贴着说。")])
c3 = chunker.chunk("v3", tight)
check("小间隔不切/大间隔可切", len(c3) <= 2, f"{len(c3)} 片 {[(c.core_start_ms, c.core_end_ms) for c in c3]}")
cuts = [c.core_end_ms for c in c3[:-1]]
# 唯一的强静音点在 10_000→10_900（0.9s）：切点必须落在它附近，
# 而不是落在小间隔（0.3s）或紧贴边界上
check("切点落在静音附近", all(abs(cut - 10_450) <= 2_500 for cut in cuts), str(cuts))

# ----------------------------------------------- 4) 超长无标点连续语音（边界核心）
long_one = segs([(0, 75_000, "这是一段完全没有任何标点符号的连续中文语音我们需要测试强制切分能力")])
c4 = chunker.chunk("v4", long_one)
check("超长无标点被切分", len(c4) >= 3, f"{len(c4)} 片 {[c.core_duration_ms for c in c4]}")
check("超长切片不超硬上限 30s", all(c.core_duration_ms <= 30_000 for c in c4),
      str([c.core_duration_ms for c in c4]))
check("超长切片不丢时长", abs(sum(c.core_duration_ms for c in c4) - 75_000) <= 100,
      f"合计 {sum(c.core_duration_ms for c in c4)}ms")
check("超长切片文本拼接完整", "".join(c.core_text for c in c4).replace(" ", "") == long_one[0]["text"],
      f"{len(''.join(c.core_text for c in c4))} vs {len(long_one[0]['text'])}")

# ------------------------------------------ 5) 单段 120s 英语（跨段硬切）
long_en = segs([(i * 12_000, (i + 1) * 12_000, f"This is sentence number {i} without any punctuation at all")
                for i in range(10)])
c5 = chunker.chunk("v5", long_en)
check("长英语流切片合规", all(c.core_duration_ms <= 30_000 for c in c5) and len(c5) >= 5,
      f"{len(c5)} 片 {[c.core_duration_ms for c in c5]}")

# ------------------------------------------------ 6) 极短输入与空输入
c6 = chunker.chunk("v6", segs([(0, 800, "嗯。")]))
check("极短输入仍产出 1 片", len(c6) == 1 and c6[0].core_duration_ms == 800, f"{len(c6)}/{c6[0].core_duration_ms if c6 else 0}")
c7 = chunker.chunk("v7", [])
check("空输入返回空列表", c7 == [])
c8 = chunker.chunk("v8", segs([(0, 3_000, "   ")]))
check("纯空白文本返回空列表", c8 == [], f"{len(c8)}")

# ------------------------------------------- 7) 微段合并（"嗯""对"并入邻段）
micro = segs([(0, 8_000, "我们来看第一个模块的设计思路。"), (8_200, 8_600, "嗯。"),
              (8_800, 16_000, "它负责视频的解码和元数据提取。"),
              (16_200, 16_700, "对。"), (17_000, 24_000, "第二个模块是模型推理层。")])
c9 = chunker.chunk("v9", micro)
check("微段被并入切片", all(c.core_duration_ms >= 3_000 or len(c9) == 1 for c in c9),
      f"{[c.core_duration_ms for c in c9]}")
check("微段文本未丢失", any("嗯" in c.text or "对" in c.text for c in c9),
      str([c.text[:12] for c in c9]))

# ------------------------------------------------- 8) 乱序 / 重叠输入鲁棒性
messy = segs([(12_000, 18_000, "第三段在文件里排在前面。"), (0, 6_000, "第一段其实在最后。"),
              (6_000, 11_500, "第二段有重叠。"), (11_000, 12_500, "这条与上一条重叠。")])
c10 = chunker.chunk("v10", messy)
check("乱序输入被排序", all(c10[i].core_start_ms < c10[i + 1].core_start_ms for i in range(len(c10) - 1)),
      str([(c.core_start_ms, c.core_end_ms) for c in c10]))
check("乱序输入不崩", len(c10) >= 1, f"{len(c10)} 片")

# ----------------------------------------------- 9) 非法输入被过滤
bad = segs([(0, 5_000, "正常一段。"), (9_000, 6_000, "结束早于开始。"), (-500, 2_000, "负时间。")])
c11 = chunker.chunk("v11", bad)
check("非法时间段被丢弃", len(c11) == 1, f"{len(c11)} 片")

# ------------------------------------------------ 10) word 级时间戳支持
words = [Word(start_ms=i * 400, end_ms=(i + 1) * 400, text=f"词{i}") for i in range(60)]
ws = [RawSegment(index=0, start_ms=0, end_ms=24_000, text="".join(w.text for w in words), words=tuple(words))]
c12 = chunker.chunk("v12", ws)
check("word 级时间戳可用", len(c12) >= 1 and c12[0].core_duration_ms > 0,
      f"{len(c12)} 片 {[c.core_duration_ms for c in c12]}")

# -------------------------------------------- 11) keyframe 绑定与反查
kf_map = {"v13-c0000": ["kf-00000", "kf-00001"]}
c13 = chunker.chunk("v13", normal, keyframe_map=kf_map)
check("keyframe_map 命中", any(c.keyframe_ids for c in c13),
      str([c.keyframe_ids for c in c13][:2]))
bound = bind_keyframes(c13, [("kf-a", c13[0].core_start_ms + 100), ("kf-b", c13[-1].core_end_ms - 100)])
check("bind_keyframes 生效", sum(len(b.keyframe_ids) for b in bound) == 2,
      str([b.keyframe_ids for b in bound if b.keyframe_ids]))

# ------------------------------------------------ 12) 配置与校验
check("非法配置被拒", _rejects(lambda: ChunkConfig(min_ms=30_000, max_ms=10_000).validate()))
check("重叠不得超最短切片", _rejects(lambda: ChunkConfig(min_ms=1_000, overlap_ms=1_500).validate()))
check("空 video_id 报错", _rejects(lambda: chunker.chunk("  ", normal), ChunkError))
cfg = ChunkConfig(overlap_ms=0)
c14 = Chunker(cfg).chunk("v14", normal)
check("关闭重叠生效", all(c.overlap_prev_ms == 0 and c.overlap_next_ms == 0 for c in c14),
      str([c.overlap_next_ms for c in c14]))
cfg2 = ChunkConfig(max_ms=15_000, target_ms=12_000)
c15 = Chunker(cfg2).chunk("v15", normal)
check("自定义上限生效", all(c.core_duration_ms <= 15_000 for c in c15), f"{[c.core_duration_ms for c in c15]}")

# ---------------------------------------------- 13) 序列化与 segments_from_whisper
row = c13[0].to_dict()
check("to_dict 可 JSON 序列化", bool(json.dumps(row, ensure_ascii=False)))
check("to_dict 含溯源键", all(k in row for k in ("clip_id", "video_id", "start_time", "end_time", "start_ms", "end_ms", "keyframe_ids")),
      str(sorted(row)[:10]))
check("to_json 批量导出", len(chunker.to_json(c13)) == len(c13))
conv = segments_from_whisper([{"start": 0.0, "end": 1.5, "text": " 你好 "}])
check("whisper 转换", len(conv) == 1 and conv[0].text == "你好" and conv[0].end_ms == 1_500,
      f"{conv[0]}")

# --------------------------------------------- 14) 长文本截断保护
cfg3 = ChunkConfig(max_chars=20)
c16 = Chunker(cfg3).chunk("v16", segs([(0, 20_000, "很长的一段话。" * 20)]))
check("max_chars 截断生效", all(len(c.core_text) <= 20 for c in c16) and any(c.truncated for c in c16),
      f"{[len(c.core_text) for c in c16]} truncated={[c.truncated for c in c16]}")

print(f"\n通过 {len(ok)} / {len(ok) + len(fail)}")
if fail:
    print("失败项:")
    for item in fail:
        print("  - " + item)
sys.exit(0 if not fail else 1)
