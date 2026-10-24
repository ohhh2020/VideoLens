"""回答证据校验（claim validation）。

## 为什么需要这个模块

本地 7B 模型和云端模型都会做同一件事：**编造一个看起来合理的引用编号**。

实测两次真实事故：

1. 检索只返回 2 条证据，回答正文输出了 `[3]` —— 编号指向不存在的证据。

2. 句子挂着真实编号，但内容与那条证据毫无关系。问"注意力机制怎么解释的"，
   模型答"它在 2017 年的论文里首次提出 [2]"，而证据 2 里根本没有年份和论文名 ——
   这是模型自己的先验知识挂在了一个真实编号上，比事故 1 更难发现。

`orchestrator.extract_citations()` 只能把 `[n]` 映射成真实切片，越界的直接丢弃 ——
但**回答正文里的 `[3]` 还在**，用户看到的是一条指向不存在证据的引用。事故 2
更是完全抓不住：编号合法，内容却是假的。

## 做法

不去要求模型输出 JSON（小模型做不到，会退化成一段坏 JSON 或干脆拒答），
而是**事后校验自由文本**：

1. 按中文标点把回答切成句子；
2. 每句收集它携带的 `[n]`；
3. 逐句与每条证据做字面重合度比较（双字 Dice 系数，对中文有效且无需分词）；
4. 三类问题分别标记：
   - `dangling` —— 引用了不存在的编号（模型编的）；
   - `unsupported` —— 有引用，但句子内容与所引证据几乎不重合（编号是抓阄贴的）；
   - `orphan` —— 完全没标引用的陈述句（可能来自模型自己的先验知识）。

## 清洗策略

`enforce_claims()` 做最小侵入的修补，不删用户看得见的答案：

- 撕掉悬空编号（`[3]` → 删除），因为留着一定是错的；
- 给 orphan 句子补一条最相近的证据编号（只在重合度够高时补，避免乱挂）；
- 其余问题只记进 `issues`，由 API 层作为 `notes` 返回，不修改正文。

这样"删掉的一定是错的，补上的有证据支撑，可疑的如实标注"。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

__all__ = [
    "Claim",
    "ClaimReport",
    "REF_PATTERN",
    "dice",
    "enforce_claims",
    "normalize_text",
    "parse_refs",
    "split_sentences",
    "strip_dangling_refs",
    "validate_claims",
]

#: 引用标记：``[1]`` / ``[1,2]`` / ``[1-3]`` / 全角 ``［1］``。
#: 分组 1 是括号内的原始串，交给 :func:`parse_refs` 展开。
REF_PATTERN: Final[re.Pattern[str]] = re.compile(r"[\[［]([0-9][0-9,，、\-\s]*)[\]］]")

#: 句子切分。中文回答里模型常用「。」「；」「！」「？」收句，
#: 也会用换行和 markdown 列表符；英文句点单独处理（避免切断 1.5、10-30 之类）。
_SENTENCE_SPLIT: Final[re.Pattern[str]] = re.compile(
    r"(?<=[。！？；!?;])|(?<=\.)(?=\s)|(?:\r?\n)+"
)

#: 这些前缀开头的句子是列表项/标题，单独成句但不参与"孤立陈述"统计。
_LIST_PREFIX: Final[tuple[str, ...]] = ("-", "*", "•", "#", ">", "1.", "2.", "3.")

#: 概括句。总结/收尾句本来就是模型对全文的归纳，不可能与某一条证据
#: 字面重合 —— 把它们算成"无出处"会让每次问答都降级，等于校验器自己
#: 制造误报。这类句子跳过判定，但仍保留在 claims 里供人工查看。
_SUMMARY_MARKERS: Final[tuple[str, ...]] = (
    "综上所述",
    "综上",
    "总之",
    "总结",
    "整体来看",
    "总体而言",
    "总的来说",
    "换句话说",
    "也就是说",
)

#: 重合度低于这个值就认为"句子和它引的证据没关系"。
DEFAULT_UNSUPPORTED_FLOOR: Final[float] = 0.12
#: 给 orphan 句子补引用时的门槛，明显高于上面 —— 宁可不补，也不乱挂。
DEFAULT_ATTACH_FLOOR: Final[float] = 0.30


def normalize_text(text: str) -> str:
    """归一化：全角转半角、去空白、去引用标记、小写化。

    比对"句子是否出自证据"时必须先剥掉引用标记，否则 `[2]` 本身会贡献
    两个字符的重合度，在短句上足以把 Dice 从 0.0 抬到 0.3。
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text)
    folded = REF_PATTERN.sub("", folded)
    folded = re.sub(r"[\s\u3000]+", "", folded)
    return folded.lower()


def _bigrams(text: str) -> set[str]:
    """中文按双字切 gram（不含标点）；少于 2 字时退化为单字。"""
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def dice(left: str, right: str) -> float:
    """双字 Dice 系数，取值 0..1（对称，用于挑最相近的证据）。"""
    a = _bigrams(normalize_text(left))
    b = _bigrams(normalize_text(right))
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return (2.0 * inter) / (len(a) + len(b))


def coverage(sentence: str, evidence: str) -> float:
    """句子内容被证据覆盖的比例，0..1（非对称）。

    ``dice`` 是对称的，判断"这句话是不是出自这条证据"时会被**长证据**稀释：
    证据是一整段 200 字转写、句子只有 50 字，分母 ``len(a) + len(b)`` 被撑大，
    真实发言人说过的话也会算成低分。

    实测事故：问"注意力机制是怎么解释的"，回答几乎是证据原句的改写，对称
    Dice 只给 0.10 —— 低于阈值被判"引用与内容不符"。改成以句子为分母的
    覆盖率后同一句得 0.80，判定正确。
    """
    a = _bigrams(normalize_text(sentence))
    b = _bigrams(normalize_text(evidence))
    if not a:
        return 0.0
    return len(a & b) / len(a)


def parse_refs(text: str) -> list[int]:
    """从一段文本里取出全部引用编号（去重、保序、只保留 >=1）。"""
    found: list[int] = []
    seen: set[int] = set()
    for match in REF_PATTERN.finditer(text or ""):
        for number in _expand_ref_body(match.group(1)):
            if number >= 1 and number not in seen:
                seen.add(number)
                found.append(number)
    return found


def _expand_ref_body(body: str) -> list[int]:
    """把 ``1,2`` / ``1-3`` / ``1、2`` 展开成具体编号。"""
    numbers: list[int] = []
    for chunk in re.split(r"[,，、\s]+", body.strip()):
        if not chunk:
            continue
        if "-" in chunk:
            lo_str, _, hi_str = chunk.partition("-")
            try:
                lo, hi = int(lo_str), int(hi_str)
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            # 上限保护：模型偶尔写出 [1-999999] 这种，别真的展开
            if hi - lo > 200:
                continue
            numbers.extend(range(lo, hi + 1))
        else:
            try:
                numbers.append(int(chunk))
            except ValueError:
                continue
    return numbers


#: 只由引用标记构成的片段。模型常把引用写在句号后面（``……所有部分。[1]``），
#: 按标点切分后 ``[1]`` 会变成独立的一句 —— 那句既没有内容可判、又会把
#: 真正的正文判成"未标出处"。切分时把它粘回前一句。
_ONLY_REFS: Final[re.Pattern[str]] = re.compile(
    r"^(?:[\[［][0-9][0-9,，、\-\s]*[\]］])+$"
)


def split_sentences(text: str) -> list[str]:
    """切成句子，保留标点，丢掉空段；孤立的引用标记粘回前一句。"""
    if not text:
        return []
    parts: list[str] = []
    for raw in _SENTENCE_SPLIT.split(text):
        piece = (raw or "").strip()
        if not piece:
            continue
        if parts and _ONLY_REFS.match(piece):
            parts[-1] = parts[-1] + piece
            continue
        parts.append(piece)
    return parts


def _is_list_line(sentence: str) -> bool:
    head = sentence.lstrip()
    return any(head.startswith(prefix) for prefix in _LIST_PREFIX)


def _is_summary(sentence: str) -> bool:
    """概括/收尾句：模型对全文的归纳，不参与字面重合度判定。"""
    head = sentence.lstrip()
    return any(marker in head for marker in _SUMMARY_MARKERS)


@dataclass(slots=True)
class Claim:
    """一句陈述及其出处判定。"""

    index: int
    text: str
    refs: list[int] = field(default_factory=list)
    valid_refs: list[int] = field(default_factory=list)
    dangling_refs: list[int] = field(default_factory=list)
    best_ref: int = 0
    similarity: float = 0.0
    verdict: str = "ok"  # ok / unsupported / orphan / noref / dangling
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict == "ok"

    def to_dict(self) -> dict[str, Any]:
        data = {
            "index": self.index,
            "text": self.text,
            "refs": self.refs,
            "valid_refs": self.valid_refs,
            "dangling_refs": self.dangling_refs,
            "best_ref": self.best_ref,
            "similarity": round(self.similarity, 4),
            "verdict": self.verdict,
        }
        if self.note:
            data["note"] = self.note
        return data


@dataclass(slots=True)
class ClaimReport:
    """整条回答的校验结果。

    ``issues`` 只放**没被修掉**的问题，``fixes`` 放**已经自动修复**的动作。
    分开的理由很实在：第一版把"给 N 句补上了引用"也当 issue，于是
    ``report.ok`` 为假、``AskResult.degraded`` 被置 True —— 明明回答已经
    被修正得更好，接口却告诉调用方"这条答案降级了"，属于撒谎。
    """

    claims: list[Claim] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    ref_count: int = 0
    checked: int = 0

    @property
    def ok(self) -> bool:
        """没有**未修复**的问题才算通过。"""
        return not self.issues

    @property
    def supported_ratio(self) -> float:
        """有证据支撑的句子占比（`ok` 与 `noref` 都算有支撑）。"""
        if not self.claims:
            return 1.0
        good = sum(1 for c in self.claims if c.verdict in ("ok", "noref"))
        return good / len(self.claims)

    @property
    def dangling(self) -> list[int]:
        seen: list[int] = []
        for claim in self.claims:
            for number in claim.dangling_refs:
                if number not in seen:
                    seen.append(number)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "supported_ratio": round(self.supported_ratio, 4),
            "ref_count": self.ref_count,
            "checked": self.checked,
            "dangling": self.dangling,
            "issues": list(self.issues),
            "fixes": list(self.fixes),
            "claims": [c.to_dict() for c in self.claims],
        }


def _ref_bodies(refs: Sequence[Any]) -> dict[int, str]:
    """把 Citation / dict / str 统一成 ``{编号: 文本}``。"""
    bodies: dict[int, str] = {}
    for position, ref in enumerate(refs, start=1):
        number = position
        text = ""
        if isinstance(ref, dict):
            number = int(ref.get("index") or position)
            text = str(ref.get("text") or "")
        else:
            got = getattr(ref, "index", None)
            if isinstance(got, int) and got > 0:
                number = got
            text = str(getattr(ref, "text", "") or "")
        bodies[number] = text
    return bodies


def validate_claims(
    answer: str,
    refs: Sequence[Any],
    *,
    unsupported_floor: float = DEFAULT_UNSUPPORTED_FLOOR,
    min_chars: int = 6,
    enabled: bool = True,
) -> ClaimReport:
    """逐句校验回答与证据的对应关系。

    参数
    ----
    answer:
        模型的自由文本回答。
    refs:
        编号可用的证据（``Citation`` 对象、dict 或任何带 ``text`` 属性的东西）。
    unsupported_floor:
        句子与所引证据的重合度低于此值即判 ``unsupported``。
    min_chars:
        归一化后短于此长度的句子不参与判定（"好""可以"这类无法比对）。
    enabled:
        ``False`` 时只做悬空编号统计，跳过重合度计算（供极低配环境省 CPU）。
    """
    report = ClaimReport(ref_count=len(refs))
    bodies = _ref_bodies(refs)
    report.ref_count = len(bodies)

    sentences = split_sentences(answer)
    if not sentences:
        return report

    for position, sentence in enumerate(sentences, start=1):
        refs_here = parse_refs(sentence)
        claim = Claim(index=position, text=sentence, refs=refs_here)
        claim.valid_refs = [n for n in refs_here if n in bodies]
        claim.dangling_refs = [n for n in refs_here if n not in bodies]

        if _is_list_line(sentence) or _is_summary(sentence):
            claim.verdict = "ok"
            claim.note = "skip:list" if _is_list_line(sentence) else "skip:summary"
            report.claims.append(claim)
            continue

        plain = normalize_text(sentence)
        if len(plain) < min_chars:
            claim.verdict = "ok"
            report.claims.append(claim)
            continue

        report.checked += 1

        if not enabled:
            claim.verdict = "dangling" if claim.dangling_refs else "ok"
            report.claims.append(claim)
            continue

        # 最佳匹配证据：优先在"这句自己引用的编号"里找，都没有再全局找
        candidates = claim.valid_refs or list(bodies)
        best_ref, best_score = 0, 0.0
        for number in candidates:
            score = coverage(plain, bodies.get(number, ""))
            if score > best_score:
                best_ref, best_score = number, score

        claim.best_ref = best_ref
        claim.similarity = best_score

        if claim.dangling_refs and not claim.valid_refs:
            claim.verdict = "dangling"
        elif claim.valid_refs:
            claim.verdict = "ok" if best_score >= unsupported_floor else "unsupported"
        elif best_score >= unsupported_floor:
            # 内容确实出自某条证据，只是模型忘了标编号 —— 记 `noref`，
            # 由 enforce_claims 补上。这类如果并进 orphan，会导致
            # "明明有出处却被报成幻觉"。
            claim.verdict = "noref"
        else:
            claim.verdict = "orphan"

        report.claims.append(claim)

    if report.dangling:
        nums = "、".join(str(n) for n in report.dangling)
        report.issues.append(f"回答引用了不存在的证据编号 [{nums}]，已从正文移除")

    unsupported = [c for c in report.claims if c.verdict == "unsupported"]
    if unsupported:
        nums = "、".join(str(c.index) for c in unsupported[:5])
        report.issues.append(
            f"{len(unsupported)} 句引用与所引证据内容不符（第 {nums} 句）"
        )

    orphans = [c for c in report.claims if c.verdict == "orphan"]
    if orphans:
        report.issues.append(f"{len(orphans)} 句没有标注来源，可能不属于视频内容")

    noref = [c for c in report.claims if c.verdict == "noref"]
    if noref:
        report.issues.append(f"{len(noref)} 句内容与证据吻合但未标出处")

    return report


def strip_dangling_refs(text: str, valid: set[int] | Sequence[int]) -> str:
    """删掉指向不存在证据的编号，保留合法编号。

    ``[3]`` → 整段删除；``[1,3]`` → 变成 ``[1]``。
    """
    allowed = {int(n) for n in valid}
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        numbers = [n for n in _expand_ref_body(match.group(1)) if n in allowed]
        if not numbers:
            return ""
        return "[" + ",".join(str(n) for n in numbers) + "]"

    cleaned = REF_PATTERN.sub(replace, text)
    # 剥掉编号后可能留下 "演示 。" 这种空格，清掉标点前的空白
    cleaned = re.sub(r"[ \t]+([。，；：！？、）」』】])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def enforce_claims(
    answer: str,
    refs: Sequence[Any],
    *,
    report: ClaimReport | None = None,
    attach_floor: float = DEFAULT_ATTACH_FLOOR,
    max_attach: int = 4,
) -> tuple[str, ClaimReport]:
    """清洗回答：撕掉悬空编号、给孤立句补证据、返回校验报告。

    返回 ``(清洗后的回答, 报告)``。正文只做两处改动，都很克制：

    - 悬空编号一定删 —— 留着一定是错的；
    - orphan 句子只在重合度 >= ``attach_floor`` 时补编号，最多补 ``max_attach`` 个。
    """
    bodies = _ref_bodies(refs)
    check = report or validate_claims(answer, refs)

    cleaned = strip_dangling_refs(answer, set(bodies))

    # 悬空编号已经删掉了 —— 它从"问题"变成"已修复"
    if check.dangling:
        kept: list[str] = []
        for issue in check.issues:
            if "不存在的证据编号" in issue:
                check.fixes.append(issue.replace("，已从正文移除", "（已删除）"))
            else:
                kept.append(issue)
        check.issues = kept

    attached = 0
    if attach_floor > 0 and bodies and attach_floor <= 1.0:
        for claim in check.claims:
            if attached >= max_attach:
                break
            if claim.verdict != "noref":
                continue
            best_ref, best_score = claim.best_ref, claim.similarity
            if not best_ref:
                # 报告是 `validate_claims(enabled=False)` 生成时没有 best_ref，补算
                for number, body in bodies.items():
                    score = coverage(claim.text, body)
                    if score > best_score:
                        best_ref, best_score = number, score
            if best_ref and best_score >= attach_floor:
                sentence = claim.text.rstrip()
                trailing = ""
                if sentence and sentence[-1] in "。！？；.!?;":
                    trailing = sentence[-1]
                    sentence = sentence[:-1]
                patched = f"{sentence} [{best_ref}]{trailing}"
                if claim.text in cleaned:
                    cleaned = cleaned.replace(claim.text, patched, 1)
                    attached += 1

    if attached:
        check.fixes.append(f"为 {attached} 句无引用的回答补上了最相近的证据编号")

    # `noref` 句要么被补上了出处（已修复），要么补不上。补不上的通常只是
    # 改写/归纳幅度大（`attach_floor` 0.30 比 `unsupported_floor` 0.12 严），
    # 内容仍然源自证据 —— 这种情况不计为问题，否则模型每次重述都会报降级。
    noref_total = sum(1 for c in check.claims if c.verdict == "noref")
    if noref_total:
        check.issues = [i for i in check.issues if "未标出处" not in i]

    return cleaned, check
