"""知影 ZhiYing —— 转写精度评测（CER / WER）。

需求文档 2.4.1 写的是"中文 CER ≤15%、英文 WER ≤20%"。这个脚本把那条指标
变成可以跑出来的数字：给一段**已知正确文本**的音频，跑一遍本项目的转写链路，
逐字/逐词对齐后算错误率，并列出错得最多的几句。

    python scripts/eval_asr.py --audio tests/media/demo.wav --reference tests/media/demo.txt
    python scripts/eval_asr.py --audio a.wav --text "今天天气很好" --top 5
    python scripts/eval_asr.py --audio a.wav --reference ref.srt --json

参考文本支持 ``.txt`` / ``.srt`` / ``.vtt`` / ``.ass``（后三种用项目自己的
字幕解析器抽纯文本）。没现成语料时最省事的办法是用 TTS 把一段已知文本合成
音频 —— 文本即标准答案，零成本且完全可复现。

指标定义：

* **CER**（字符错误率）= (替换 + 删除 + 插入) / 参考字符数，中文按字算。
* **WER**（词错误率）= 同样算法，英文按空格切词算。

退出码：0 正常、1 转写失败、2 参数或配置错误。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import AppConfig, ConfigError, get_app_config  # noqa: E402

logger = logging.getLogger("zhiying.eval_asr")

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

#: 中日韩字符占比超过这个比例就按"字"算 CER，否则按"词"算 WER。
_CJK_RATIO = 0.30
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])\s*")
_WORD = re.compile(r"[a-z0-9']+|[\u3400-\u4dbf\u4e00-\u9fff]")
_SUBTITLE_SUFFIX = {".srt", ".vtt", ".ass", ".ssa"}
#: 阿拉伯数字与中文数字串。转写输出 "2048"，人读参考写 "两千零四十八" ——
#: 语义一致但字面完全不同，不归一化会凭空多出十几个"替换"。
_NUM_AR = re.compile(r"[0-9]+")
_NUM_CN = re.compile(r"[零〇一二三四五六七八九十百千万亿两]+")


# -----------------------------------------------------------------------------
# 数据结构
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class Errors:
    """一次对齐的编辑统计。"""

    sub: int = 0
    delete: int = 0
    insert: int = 0
    total: int = 0

    @property
    def count(self) -> int:
        return self.sub + self.delete + self.insert

    @property
    def rate(self) -> float:
        return self.count / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub": self.sub,
            "delete": self.delete,
            "insert": self.insert,
            "total": self.total,
            "errors": self.count,
            "rate": round(self.rate, 4),
        }


@dataclass(slots=True)
class SentenceScore:
    """一句参考文本的匹配情况。"""

    reference: str
    hypothesis: str
    errors: Errors = field(default_factory=Errors)

    @property
    def rate(self) -> float:
        return self.errors.rate


@dataclass(slots=True)
class Report:
    """整份音频的评测结果。

    ``errors`` 是数字归一化后的成绩（贴近"听没听对"），``errors_raw`` 是
    字面成绩（含数字记法差异），后者通常更高 —— 两个都报，避免拿归一化
    掩盖真实问题，也避免拿记法差异冤枉模型。
    """

    unit: str
    reference_len: int
    errors: Errors
    errors_raw: Errors = field(default_factory=Errors)
    sentences: list[SentenceScore] = field(default_factory=list)
    transcript: str = ""
    elapsed_ms: int = 0
    model: str = ""
    degraded: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.errors.rate

    @property
    def worst(self) -> list[SentenceScore]:
        return sorted(
            (s for s in self.sentences if s.errors.count > 0),
            key=lambda s: (s.rate, s.errors.count),
            reverse=True,
        )

    def verdict(self) -> str:
        """对照需求文档 2.4.1 的阈值给结论。"""
        ceiling = 0.15 if self.unit == "字" else 0.20
        label = "CER" if self.unit == "字" else "WER"
        if self.rate <= ceiling:
            return f"{label} {self.rate * 100:.1f}% ≤ {ceiling * 100:.0f}%，达标"
        return f"{label} {self.rate * 100:.1f}% > {ceiling * 100:.0f}%，未达标"

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "rate": round(self.rate, 4),
            "rate_raw": round(self.errors_raw.rate, 4),
            "reference_len": self.reference_len,
            "errors": self.errors.to_dict(),
            "errors_raw": self.errors_raw.to_dict(),
            "model": self.model,
            "elapsed_ms": self.elapsed_ms,
            "degraded": self.degraded,
            "verdict": self.verdict(),
            "warnings": self.warnings,
            "worst": [
                {
                    "reference": s.reference,
                    "hypothesis": s.hypothesis,
                    "rate": round(s.rate, 4),
                    "errors": s.errors.to_dict(),
                }
                for s in self.worst
            ],
        }


# -----------------------------------------------------------------------------
# 文本归一化与对齐
# -----------------------------------------------------------------------------


def _fold(text: str, *, numbers: bool = True) -> str:
    """全角转半角、去标点空白、统一小写。

    ``numbers=True`` 时把阿拉伯数字与中文数字串都折叠成单字母 ``n``：
    参考文本写"两千零四十八"、转写输出"2048"，语义一致不该算错。
    """
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    if numbers:
        normalized = _NUM_CN.sub("n", _NUM_AR.sub("n", normalized))
    return "".join(ch for ch in normalized if ch.isalnum())


def _tokens(text: str) -> list[str]:
    """按词切（英文/数字），中文逐字。用于 WER。"""
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    return [match.group(0) for match in _WORD.finditer(normalized)]


def is_cjk_text(text: str) -> bool:
    body = re.sub(r"\s+", "", text or "")
    if not body:
        return True
    return len(_CJK.findall(body)) / len(body) >= _CJK_RATIO


def tokenize(text: str, *, cjk: bool, numbers: bool = True) -> list[str]:
    """切成用于对齐的 token 序列：中文逐字，英文按词。

    ``numbers=True`` 时把数字折成一个 ``n``，两种记法在同一个尺度上比较。
    """
    if cjk:
        return list(_fold(text, numbers=numbers))
    out: list[str] = []
    for token in _tokens(text):
        out.append("n" if numbers and token.isdigit() else token)
    return out


def align(ref: list[str], hyp: list[str]) -> Errors:
    """逐元素对齐，统计替换/删除/插入。

    ``SequenceMatcher`` 的 ``replace`` 段落两边长度可能不同，多出来的部分
    分别算成删除（参考有、转写没有）与插入（转写有、参考没有）。
    """
    stats = Errors(total=len(ref))
    if not hyp:
        stats.delete = len(ref)
        return stats
    matcher = SequenceMatcher(None, ref, hyp, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        left = i2 - i1
        right = j2 - j1
        paired = min(left, right)
        stats.sub += paired
        stats.delete += left - paired
        stats.insert += right - paired
    return stats


def split_sentences(text: str) -> list[str]:
    parts = [piece.strip() for piece in _SENT_SPLIT.split(text or "")]
    return [piece for piece in parts if piece]


def score_sentences(reference: str, hypothesis: str) -> list[SentenceScore]:
    """参考分句后与转写分句做句级对齐，逐句算错误率。"""
    ref_sents = split_sentences(reference)
    hyp_sents = split_sentences(hypothesis)
    if not ref_sents:
        return []
    if not hyp_sents:
        return [
            SentenceScore(
                reference=s,
                hypothesis="",
                errors=Errors(delete=len(_fold(s)), total=len(_fold(s))),
            )
            for s in ref_sents
        ]

    key_ref = [_fold(s) for s in ref_sents]
    key_hyp = [_fold(s) for s in hyp_sents]
    out: list[SentenceScore] = []
    matcher = SequenceMatcher(None, key_ref, key_hyp, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                out.append(
                    SentenceScore(
                        reference=ref_sents[i1 + offset],
                        hypothesis=hyp_sents[j1 + offset],
                        errors=align(
                            list(key_ref[i1 + offset]), list(key_hyp[j1 + offset])
                        ),
                    )
                )
            continue
        # 参考句与转写句数量不等时，按顺序模糊配对，剩下的参考句记为整句缺失。
        span_ref = i2 - i1
        span_hyp = j2 - j1
        for offset in range(span_ref):
            hyp_index = j1 + offset
            if offset < span_hyp and hyp_index < len(key_hyp):
                out.append(
                    SentenceScore(
                        reference=ref_sents[i1 + offset],
                        hypothesis=hyp_sents[hyp_index],
                        errors=align(
                            list(key_ref[i1 + offset]), list(key_hyp[hyp_index])
                        ),
                    )
                )
            else:
                out.append(
                    SentenceScore(
                        reference=ref_sents[i1 + offset],
                        hypothesis="",
                        errors=Errors(
                            delete=len(key_ref[i1 + offset]),
                            total=len(key_ref[i1 + offset]),
                        ),
                    )
                )
    return out


# -----------------------------------------------------------------------------
# 参考文本读取
# -----------------------------------------------------------------------------


def load_reference(path: Path) -> str:
    """读参考文本。字幕文件走项目自己的解析器，其它按 UTF-8 纯文本。"""
    suffix = path.suffix.lower()
    if suffix in _SUBTITLE_SUFFIX:
        from backend.media.subtitles import load_subtitles  # noqa: PLC0415

        # load_subtitles 返回 (cues, 命中文件, 告警)，且收的是候选序列。
        cues, _used, _warnings = load_subtitles([path])
        return "。".join(cue.text.strip(" 。") for cue in cues if cue.text.strip())
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"无法解码参考文本: {path}")


# -----------------------------------------------------------------------------
# 转写
# -----------------------------------------------------------------------------


def run_asr(audio: Path, config: AppConfig) -> Any:
    """跑一次本项目的转写链路（字幕优先，回退 whisper）。"""
    from backend.models.asr import AsrConfig, AsrEngine  # noqa: PLC0415

    engine = AsrEngine(AsrConfig.from_config(config))
    return engine.transcribe(audio, video_id="eval", allow_subtitle=False)


# -----------------------------------------------------------------------------
# 报告渲染
# -----------------------------------------------------------------------------


def _short(text: str, limit: int = 46) -> str:
    body = re.sub(r"\s+", " ", (text or "").strip())
    return body if len(body) <= limit else body[: limit - 1] + "…"


def render(report: Report, *, top: int) -> str:
    unit = report.unit
    lines = ["知影 转写精度评测", ""]
    if report.model:
        lines.append(f"  模型       {report.model}")
    lines.append(f"  参考长度   {report.reference_len} {unit}")
    lines.append(f"  耗时       {report.elapsed_ms / 1000:.1f}s")
    lines.append("")
    lines.append(
        f"  错误构成   替换 {report.errors.sub} / 删除 {report.errors.delete} / 插入 {report.errors.insert}"
    )
    lines.append(f"  {report.verdict()}（需求 2.4.1，数字已归一化）")
    gap = report.errors_raw.rate - report.rate
    if gap > 0.005:
        lines.append(
            f"  · 不归一化时为 {report.errors_raw.rate * 100:.1f}%，"
            f"差值来自数字记法（如“两千零四十八” vs “2048”）"
        )
    if report.degraded:
        lines.append("  · 注意：本次转写处于降级状态，结论仅供参考")
    for warn in report.warnings:
        lines.append(f"  · {warn}")

    worst = report.worst[: max(0, top)]
    if worst:
        lines.append("")
        lines.append(f"  错得最多的 {len(worst)} 句：")
        for index, item in enumerate(worst, 1):
            lines.append(
                f"    {index}. [{item.rate * 100:5.1f}%] 参考：{_short(item.reference)}"
            )
            lines.append(f"       转写：{_short(item.hypothesis) or '（空）'}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_asr", description="知影：算这段音频的 CER / WER"
    )
    parser.add_argument("--audio", required=True, help="待评测音频或视频")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reference", help="参考文本文件（.txt/.srt/.vtt/.ass）")
    group.add_argument("--text", help="直接把参考文本写在命令行里")
    parser.add_argument("--top", type=int, default=8, help="列出最差几句（默认 8）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="日志详细度")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.WARNING if args.verbose == 0 else logging.INFO
    if args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    audio = Path(args.audio)
    if not audio.is_file():
        print(f"音频不存在：{audio}", file=sys.stderr)
        return EXIT_USAGE

    try:
        reference = (
            args.text if args.text is not None else load_reference(Path(args.reference))
        )
    except (OSError, UnicodeDecodeError) as exc:
        print(f"读取参考文本失败：{exc}", file=sys.stderr)
        return EXIT_USAGE
    if not (reference or "").strip():
        print("参考文本为空", file=sys.stderr)
        return EXIT_USAGE

    try:
        config = get_app_config()
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        transcript = run_asr(audio, config)
    except Exception as exc:  # noqa: BLE001 - 转写链路任何环节失败都算评测失败
        print(f"转写失败：{exc}", file=sys.stderr)
        return EXIT_FAIL

    hypothesis = getattr(transcript, "text", "") or ""
    cjk = is_cjk_text(reference)
    ref_tokens = tokenize(reference, cjk=cjk)
    hyp_tokens = tokenize(hypothesis, cjk=cjk)
    errors = align(ref_tokens, hyp_tokens)
    errors_raw = align(
        tokenize(reference, cjk=cjk, numbers=False),
        tokenize(hypothesis, cjk=cjk, numbers=False),
    )

    report = Report(
        unit="字" if cjk else "词",
        reference_len=len(ref_tokens),
        errors=errors,
        errors_raw=errors_raw,
        sentences=score_sentences(reference, hypothesis),
        transcript=hypothesis,
        elapsed_ms=int(getattr(transcript, "elapsed_ms", 0) or 0),
        model=getattr(config.asr, "model", ""),
        degraded=bool(getattr(transcript, "degraded", False)),
        warnings=list(getattr(transcript, "warnings", ()) or ()),
    )

    if args.json:
        payload = report.to_dict()
        payload["transcript"] = hypothesis
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render(report, top=args.top))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
