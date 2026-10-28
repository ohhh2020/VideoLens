"""``backend.pipeline.claims`` 的自检 —— 回答证据校验。

重点覆盖四类真实事故（都来自本项目的实际运行日志）：

1. 模型编造不存在的编号（`[3]`，而检索只返回 2 条）；
2. 编号真实但内容不符（抓阄贴编号）；
3. 完全无引用的陈述句（先验知识混进回答）；
4. 合法回答不能被误伤 —— 这是最容易被忽略的一条，校验器太激进
   会把正确答案改坏。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pipeline.claims import (  # noqa: E402
    ClaimReport,
    dice,
    enforce_claims,
    normalize_text,
    parse_refs,
    split_sentences,
    strip_dangling_refs,
    validate_claims,
)

PASS = 0
FAIL = 0


def ok(condition: object, label: str, detail: object = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok  {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}  {detail}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


REFS = [
    {
        "index": 1,
        "text": "神经网络用反向传播算法训练。它通过链式法则计算梯度。"
        "然后用梯度下降更新每一层的权重。",
    },
    {
        "index": 2,
        "text": "注意力机制让模型动态关注输入的不同位置。Transformer 完全依赖注意力。"
        "它抛弃了循环神经网络的结构。",
    },
    {
        "index": 3,
        "text": "向量数据库存储高维嵌入向量。它通过余弦相似度做近似最近邻搜索。"
        "这是检索增强生成的基础。",
    },
]


class FakeCitation:
    """模拟 ``orchestrator.Citation``（只依赖 index / text 两个属性）。"""

    def __init__(self, index: int, text: str) -> None:
        self.index = index
        self.text = text


section("1. normalize_text / dice")
ok(normalize_text("  ＡＢＣ 123 [1] ") == "abc123", "归一化: 全角转半角+去空白+去引用")
ok(normalize_text("") == "" and normalize_text(None or "") == "", "归一化: 空输入")
ok(dice("梯度下降", "梯度下降") == 1.0, "dice: 完全相同=1")
ok(dice("梯度下降", "注意力机制") == 0.0, "dice: 完全不同=0")
ok(
    abs(dice("反向传播计算梯度", "通过反向传播计算梯度的方法") - 0.6) < 0.25,
    "dice: 部分重合走高",
)
ok(dice("a", "b") == 0.0 and dice("", "") == 0.0, "dice: 单字与空串不炸")
ok(dice("链式法则", "链式法则 [1]") == 1.0, "dice: 引用标记不贡献重合度")

section("2. parse_refs")
ok(parse_refs("见 [1] 和 [2]") == [1, 2], "单个编号")
ok(parse_refs("[1,3]") == [1, 3], "逗号分隔")
ok(parse_refs("[1、3]") == [1, 3], "顿号分隔")
ok(parse_refs("[2-4]") == [2, 4], "范围展开（2-4 → 2,3,4）") if False else None
ok(parse_refs("[2-4]") == [2, 3, 4], "范围展开 2-4")
ok(parse_refs("［1］") == [1], "全角括号")
ok(parse_refs("[3,1,3]") == [3, 1], "去重且保序")
ok(parse_refs("没有引用") == [], "无编号返回空")
ok(parse_refs("[0]") == [], "0 不算编号")

section("3. split_sentences")
ok(
    split_sentences("第一句。第二句！第三句？") == ["第一句。", "第二句！", "第三句？"],
    "中文句末",
)
ok(len(split_sentences("一行\n二行\n三行")) == 3, "按换行切")
ok(split_sentences("") == [], "空串")
ok(
    all("1.5" in s or "2" in s for s in split_sentences("精度 1.5 高于 2")),
    "小数点不被切断",
)
# 模型常把引用写在句号之后（`……所有部分。[1]`）。按标点切分后 `[1]` 会变成
# 独立的一句 —— 那句没有内容可判、又会把真正的正文判成"未标出处"。
ok(split_sentences("正文在这里。[1]") == ["正文在这里。[1]"], "句号后的引用粘回本句")
ok(split_sentences("[1] 引用开头的句子") == ["[1] 引用开头的句子"], "句首引用不拆")
ok(
    split_sentences("第一点。[1]\n第二点。[2]") == ["第一点。[1]", "第二点。[2]"],
    "多行各自带引用",
)

section("4. validate_claims —— 合法回答不误伤")
good = (
    "视频先讲了神经网络的反向传播训练：通过链式法则计算梯度，再用梯度下降更新每层权重 [1]。"
    "接着转向注意力机制，说明 Transformer 完全依赖注意力，抛弃了循环结构 [2]。"
)
r_good = validate_claims(good, REFS)
ok(r_good.ok, "合法回答无 issue", r_good.issues)
ok(
    r_good.supported_ratio == 1.0,
    "合法回答 supported_ratio=1.0",
    r_good.supported_ratio,
)
ok(r_good.dangling == [], "合法回答无悬空编号")
ok(all(c.verdict == "ok" for c in r_good.claims), "两句全判 ok")

section("5. validate_claims —— 悬空编号（真实事故：检索只回 2 条，模型引用 [3]）")
# 事故现场只有 2 条证据，模型却写出了 [3]
TWO = REFS[:2]
bad_ref = "反向传播用链式法则算梯度 [1]。另外提到了 02 知影演示 [3]。"
r_bad = validate_claims(bad_ref, TWO)
ok(r_bad.dangling == [3], "识别悬空编号 [3]", r_bad.dangling)
ok(not r_bad.ok, "有 issue")
ok(any("不存在的证据编号" in i for i in r_bad.issues), "issue 文案正确", r_bad.issues)

section("6. validate_claims —— 编号真实但内容不符")
mismatch = "量子计算可以在多项式时间内分解大整数 [1]。"
r_mis = validate_claims(mismatch, REFS)
ok(r_mis.claims[0].verdict == "unsupported", "判 unsupported", r_mis.claims[0].verdict)
ok(r_mis.claims[0].similarity < 0.12, "重合度极低", r_mis.claims[0].similarity)

section("7. validate_claims —— 无引用的孤立句")
orphan = (
    "反向传播通过链式法则计算梯度，再用梯度下降更新权重 [1]。"
    "这个方法由 Geoffrey Hinton 在 1986 年推广。"
)
r_orph = validate_claims(orphan, REFS)
verdicts = [c.verdict for c in r_orph.claims]
ok(verdicts == ["ok", "orphan"], "第一句 ok、第二句 orphan", verdicts)
ok(any("没有标注来源" in i for i in r_orph.issues), "issue 提示孤立句", r_orph.issues)

section("8. validate_claims —— 边界")
ok(validate_claims("", REFS).ok, "空回答直接通过")
ok(validate_claims("好", REFS).ok, "过短句子跳过判定")
ok(
    validate_claims("这是一句没有证据支撑但足够长的话，用来测试空证据表。", []).ok
    is False,
    "无证据时任何陈述都成 orphan",
)
covered = validate_claims("反向传播要算梯度 [9]。", REFS)
ok(
    covered.dangling == [9] and covered.claims[0].verdict == "dangling",
    "越界且无合法引用 → dangling",
)

section("9. validate_claims —— FakeCitation 对象入参")
r_obj = validate_claims(
    good, [FakeCitation(1, REFS[0]["text"]), FakeCitation(2, REFS[1]["text"])]
)
ok(
    r_obj.ok and r_obj.ref_count == 2,
    "支持带 index/text 的对象",
    (r_obj.ok, r_obj.ref_count),
)

section("10. strip_dangling_refs")
ok(
    strip_dangling_refs("结论甲 [3]。", {1, 2}) == "结论甲。",
    "单个悬空编号整段删除且不留空格",
)
ok(
    strip_dangling_refs("结论甲 [1,3]。", {1, 2}) == "结论甲 [1]。",
    "混合编号只保留合法的",
)
ok(strip_dangling_refs("结论甲 [1]。", {1}) == "结论甲 [1]。", "合法编号原样保留")
ok(strip_dangling_refs("", {1}) == "", "空串")

section("11. enforce_claims —— 清洗")
cleaned, rep = enforce_claims(bad_ref, TWO)
ok("[3]" not in cleaned, "悬空编号已删除", cleaned)
ok("[1]" in cleaned, "合法编号保留", cleaned)
ok(
    "知影演示。" in cleaned or "知影演示 。" not in cleaned,
    "删除编号后不留多余空格",
    cleaned,
)
ok(rep.dangling == [3], "报告仍记录原问题")

section("12. enforce_claims —— 补出处")
attach = "反向传播通过链式法则计算梯度，再用梯度下降更新权重。注意力机制让模型动态关注输入的不同位置。"
cleaned2, rep2 = enforce_claims(attach, REFS, attach_floor=0.25)
ok(cleaned2.count("[") >= 2, "两句都补上了编号", cleaned2)
ok(
    any("补上了最相近的证据编号" in i for i in rep2.fixes),
    "修复动作记进 fixes 而非 issues",
    rep2.fixes,
)
ok(rep2.ok, "补引用不算未修复问题（不能因此标降级）", rep2.issues)

section("13. enforce_claims —— 门槛高时不乱挂")
cleaned3, rep3 = enforce_claims(
    "量子计算可以在多项式时间内分解大整数。", REFS, attach_floor=0.9
)
ok("[" not in cleaned3, "重合度不够时不补编号", cleaned3)
ok(not any("补上了" in i for i in rep3.issues), "报告不谎报")

section("14. enforce_claims —— 不改动合法回答")
cleaned4, rep4 = enforce_claims(good, REFS)
ok(cleaned4 == good.strip(), "合法回答逐字不变")
ok(rep4.ok, "报告无 issue")

section("15. ClaimReport 序列化")
ok(isinstance(ClaimReport().to_dict(), dict), "空报告可序列化")
d = r_good.to_dict()
ok(
    {"ok", "supported_ratio", "ref_count", "checked", "dangling", "claims"} <= set(d),
    "字段齐全",
    sorted(d),
)
ok(d["claims"][0]["verdict"] == "ok", "claim 条目含 verdict")

print("\n" + "=" * 60)
print(f"PASS={PASS}  FAIL={FAIL}")
sys.exit(0 if FAIL == 0 else 1)
