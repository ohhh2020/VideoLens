"""知影 ZhiYing —— 转写精度评测自检（``scripts/eval_asr.py`` 纯逻辑部分）。

零 pytest 依赖，``check(name, cond)`` 累积器，退出码非 0 即失败。

**不加载 ASR 模型**：``run_asr`` 需要真实权重，留给
``python scripts/eval_asr.py --audio ... --reference ...`` 手工跑。这里只验证
"算分是否算对"，那才是容易写错的部分 —— 模型准不准是模型的事，分数对不对是
代码的事。

覆盖：
* 中文数字与阿拉伯数字归一化（否则「2048」对「两千零四十八」白扣分）
* 标点、全半角、大小写不影响分数
* CER 的替换/删除/插入三种错误的计数口径
* 句级对齐不会因为中间插了一句就整体错位
* 参考文本的编码兜底（UTF-8 / UTF-8-BOM / GB18030）与字幕格式
* 达标判定与最差句排序

跑法：``python -u tests/test_eval.py``
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval_asr import (  # noqa: E402
    Errors,
    Report,
    SentenceScore,
    _fold,
    align,
    is_cjk_text,
    load_reference,
    render,
    score_sentences,
    split_sentences,
    tokenize,
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


def tmpdir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="zy-eval-"))
    _ROOTS.append(root)
    return root


def main() -> int:
    # -- 1. 语言判定 -------------------------------------------------------
    print("\n[1] 语言判定")
    check("纯中文是 CJK", is_cjk_text("知影视频分析助手") is True, "")
    check("纯英文不是 CJK", is_cjk_text("hello world") is False, "")
    check("中英混排按中文算", is_cjk_text("这个 API 很好用") is True, "")
    # 空文本按中文处理是有意的：项目面向中文视频，宁可用严格指标 CER。
    check("空串默认按中文", is_cjk_text("") is True, "")
    check("数字不算 CJK", is_cjk_text("12345") is False, "")

    # -- 2. 归一化 ---------------------------------------------------------
    print("\n[2] 归一化")
    check("大小写折叠", _fold("Hello WORLD") == "helloworld", _fold("Hello WORLD"))
    check("去标点", _fold("你好，世界！") == "你好世界", _fold("你好，世界！"))
    check("全角转半角", _fold("ＡＢＣ１２３") == "abcn", _fold("ＡＢＣ１２３"))
    check("阿拉伯数字折成 n", _fold("共 2048 个") == "共n个", _fold("共 2048 个"))
    check("中文数字折成 n", _fold("共两千零四十八个") == "共n个", _fold("共两千零四十八个"))
    check("数字不归一化时保留", _fold("共 2048 个", numbers=False) == "共2048个", _fold("共 2048 个", numbers=False))
    check("空白被去掉", _fold("a b\tc") == "abc", _fold("a b\tc"))
    check("引号被去掉", _fold('他说"好"') == "他说好", _fold('他说"好"'))

    # -- 3. 分词 -----------------------------------------------------------
    print("\n[3] 分词")
    check("中文逐字", tokenize("你好世界", cjk=True) == ["你", "好", "世", "界"], tokenize("你好世界", cjk=True))
    check("中文忽略标点", tokenize("你好，世界。", cjk=True) == ["你", "好", "世", "界"], tokenize("你好，世界。", cjk=True))
    check("英文按词切", tokenize("hello big world", cjk=False) == ["hello", "big", "world"], tokenize("hello big world", cjk=False))
    check("英文词内撇号保留", tokenize("don't stop", cjk=False) == ["don't", "stop"], tokenize("don't stop", cjk=False))
    check("英文数字折成 n", tokenize("about 2048 items", cjk=False) == ["about", "n", "items"], tokenize("about 2048 items", cjk=False))
    check("空串切出空表", tokenize("", cjk=True) == [], "")

    # -- 4. 错误计数 -------------------------------------------------------
    print("\n[4] 编辑距离口径")
    same = align(["你", "好"], ["你", "好"])
    check("全对无错误", same.count == 0, same.to_dict())
    check("全对率 0", same.rate == 0.0, same.rate)

    sub = align(["你", "好", "世", "界"], ["你", "坏", "世", "界"])
    check("1 处替换", (sub.sub, sub.delete, sub.insert) == (1, 0, 0), sub.to_dict())
    check("替换率 0.25", abs(sub.rate - 0.25) < 1e-9, sub.rate)

    dele = align(["你", "好", "世", "界"], ["你", "好"])
    check("2 处删除", (dele.sub, dele.delete, dele.insert) == (0, 2, 0), dele.to_dict())
    check("删除率 0.5", abs(dele.rate - 0.5) < 1e-9, dele.rate)

    ins = align(["你", "好"], ["你", "好", "世", "界"])
    check("2 处插入", (ins.sub, ins.delete, ins.insert) == (0, 0, 2), ins.to_dict())
    check("插入率按参考长度算", abs(ins.rate - 1.0) < 1e-9, ins.rate)

    check("空参考不炸", align([], ["你"]).rate == 0.0, "")
    check("两边都空", align([], []).count == 0, "")

    # 数字归一化真的能把「2048」的错扣掉 —— 这是最容易误伤的场景。
    plain = align(
        tokenize("共2048个", cjk=True, numbers=False),
        tokenize("共两千零四十八个", cjk=True, numbers=False),
    )
    folded = align(tokenize(_fold("共2048个"), cjk=True), tokenize(_fold("共两千零四十八个"), cjk=True))
    check("不归一化会算错", plain.count > 0, plain.to_dict())
    check("归一化后算对", folded.count == 0, folded.to_dict())

    # -- 5. 句级对齐 -------------------------------------------------------
    print("\n[5] 句级对齐")

    def total(rows: list[SentenceScore]) -> int:
        return sum(row.errors.count for row in rows)

    ref = "第一句在这里。第二句在这里。第三句在这里。"
    hyp = "第一句在这里。第二句在这里。第三句在这里。"
    whole = score_sentences(ref, hyp)
    check("句级全对", total(whole) == 0, total(whole))
    check("句级句数", len(whole) == 3, len(whole))

    partial = score_sentences(ref, "第一句在这里。第三句在这里。")
    check("漏一句记删除", sum(r.errors.delete for r in partial) > 0, [(r.reference, r.hypothesis) for r in partial])
    check("句级仍有 3 条参考", len(partial) == 3, len(partial))
    check("漏掉的那句被标出", any(not row.hypothesis for row in partial), [row.hypothesis for row in partial])

    check("分句按中文句号", split_sentences("甲。乙！丙？") == ["甲。", "乙！", "丙？"], split_sentences("甲。乙！丙？"))
    # 英文句号不切是有意的：小数点、缩写、版本号都会被误切。
    check("英文句号不切", split_sentences("a. b! c?") == ["a. b!", "c?"], split_sentences("a. b! c?"))
    check("英文感叹号切", split_sentences("one! two!") == ["one!", "two!"], split_sentences("one! two!"))
    check("空串分句为空", split_sentences("") == [], "")

    # -- 6. 参考文本读取 ---------------------------------------------------
    print("\n[6] 参考文本读取")
    work = tmpdir()
    plain_path = work / "ref.txt"
    plain_path.write_text("第一句。第二句。", encoding="utf-8")
    check("UTF-8 文本", load_reference(plain_path) == "第一句。第二句。", load_reference(plain_path))

    bom_path = work / "bom.txt"
    bom_path.write_text("带 BOM 的文本。", encoding="utf-8-sig")
    check("UTF-8 BOM 文本", "带 BOM" in load_reference(bom_path), load_reference(bom_path))

    gb_path = work / "gb.txt"
    gb_path.write_bytes("国标编码。".encode("gb18030"))
    check("GB18030 兜底", load_reference(gb_path) == "国标编码。", load_reference(gb_path))

    srt_path = work / "ref.srt"
    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n第一句字幕\n\n2\n00:00:02,000 --> 00:00:04,000\n第二句字幕\n",
        encoding="utf-8",
    )
    merged = load_reference(srt_path)
    check("SRT 抽出正文", "第一句字幕" in merged and "第二句字幕" in merged, merged)
    check("SRT 不含时间轴", "-->" not in merged, merged)
    check("SRT 两句用句号相连", merged.count("。") >= 1, merged)

    # -- 7. 报告 -----------------------------------------------------------
    print("\n[7] 报告与判定")
    ok_report = Report(unit="字", reference_len=100, errors=Errors(5, 0, 0, 100))
    check("5% 及格", "达标" in ok_report.verdict() and "未达标" not in ok_report.verdict(), ok_report.verdict())
    check("rate 5%", abs(ok_report.rate - 0.05) < 1e-9, ok_report.rate)

    bad_report = Report(unit="字", reference_len=100, errors=Errors(30, 0, 0, 100))
    check("30% 不及格", "未达标" in bad_report.verdict(), bad_report.verdict())

    word_report = Report(unit="词", reference_len=40, errors=Errors(9, 0, 0, 40))
    check("WER 阈值 20%", "未达标" in word_report.verdict(), word_report.verdict())
    check("WER 标签", "WER" in word_report.verdict(), word_report.verdict())
    check("CER 标签", "CER" in bad_report.verdict(), bad_report.verdict())

    check("空报告 rate 0", Report(unit="字", reference_len=0, errors=Errors()).rate == 0.0, "")
    check("to_dict 可序列化", isinstance(ok_report.to_dict(), dict), type(ok_report.to_dict()).__name__)
    check("to_dict 含 verdict", "verdict" in ok_report.to_dict(),
          str(ok_report.to_dict())[:80])

    ranked = Report(
        unit="字",
        reference_len=20,
        errors=Errors(6, 0, 0, 20),
        sentences=[
            SentenceScore("完全正确的句子在这里", "完全正确的句子在这里", Errors(0, 0, 0, 10)),
            SentenceScore("这句差得比较多一些", "这句完全不同", Errors(7, 0, 0, 9)),
            SentenceScore("这句中等", "这句中", Errors(1, 0, 0, 4)),
        ],
    )
    worst = ranked.worst
    check("worst 返回最差句", bool(worst) and worst[0].reference.startswith("这句差得"), worst[0].reference if worst else None)
    check("worst 只含有错的句", len(worst) == 2, [s.reference for s in worst])

    out = render(ranked, top=2)
    check("渲染含 CER", "CER" in out, out[:80])
    check("渲染含错误构成", "替换" in out or "sub" in out.lower(), out[:200])
    check("渲染列出最差句", "这句差得" in out, out[:300])

    print(f"\nPASS {PASSED}  FAIL {len(FAILED)}")
    for item in FAILED:
        print("  -", item)
    return 0 if not FAILED else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        import shutil

        for root in _ROOTS:
            shutil.rmtree(root, ignore_errors=True)
        del traceback
