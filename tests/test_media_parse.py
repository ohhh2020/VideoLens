"""media.parser / media.subtitles / models.asr(字幕路径) 自检。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.media.parser import (  # noqa: E402
    DEFAULT_ALLOWED_EXTENSIONS,
    ParseOptions,
    Platform,
    SourceKind,
    VideoTooLargeError,
    checksum_file,
    derive_video_id,
    guess_platform,
    parse_source,
    sanitize_filename,
)
from backend.media.subtitles import (  # noqa: E402
    SubtitleCue,
    detect_encoding,
    load_subtitles,
    merge_cues,
    parse_ass,
    parse_srt,
    parse_subtitle,
    parse_vtt,
    strip_markup,
)
from backend.models.asr import AsrConfig, AsrEngine, Transcript  # noqa: E402

TMP = (
    __import__("backend.runtime", fromlist=["runtime_path"]).runtime_path("tests/work")
    / "subs"
)
TMP.mkdir(parents=True, exist_ok=True)

PASS = 0
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL.append(f"{name} {detail}")
        print(f"  FAIL {name} {detail}")


def section(title: str) -> None:
    print(f"\n== {title}")


# --------------------------------------------------------------------- SRT
def _err(fn):
    try:
        fn()
    except Exception as exc:
        return type(exc).__name__
    return "NO ERROR"


section("SRT")
SRT = (
    "\ufeff1\r\n00:00:00,000 --> 00:00:02,500\r\n你好，这是第一句。\r\n\r\n"
    "2\r\n00:00:02,500 --> 00:00:06,120\r\n<i>第二句</i>带斜体\r\n\r\n"
    "00:00:06,300 --> 00:00:09,000\r\n第三句没有序号\r\n"
)
cues = parse_srt(SRT)
check("解析出 3 条", len(cues) == 3, str(len(cues)))
check("首条 0..2500", (cues[0].start_ms, cues[0].end_ms) == (0, 2500), str(cues[0]))
check("文本去标签", cues[1].text == "第二句带斜体", cues[1].text)
check("缺编号也能解析", cues[2].start_ms == 6300, str(cues[2].start_ms))
check("index 连续", [c.index for c in cues] == [0, 1, 2], str([c.index for c in cues]))

# --------------------------------------------------------------------- VTT
section("WebVTT")
VTT = """WEBVTT - generated

NOTE 这是注释块，不该被解析

STYLE
::cue { color: white }

00:00:00.000 --> 00:00:02.000 align:start position:10%
<c.yellow>黄色字幕</c>

cue-2
00:00:02.000 --> 00:00:05.500
第二段<00:00:03.000>带时间戳标记

X-TIMESTAMP-MAP=MPEGTS:900000,LOCAL:00:00:00.000
"""
vcues = parse_vtt(VTT)
check("解析出 2 条", len(vcues) == 2, str(len(vcues)))
check("去 c 标签", vcues[0].text == "黄色字幕", vcues[0].text)
check("去内嵌时间戳标记", vcues[1].text == "第二段带时间戳标记", vcues[1].text)
check("cue settings 不影响时间", vcues[0].end_ms == 2000, str(vcues[0].end_ms))

# --------------------------------------------------------------------- ASS
section("ASS")
ASS = """[Script Info]
Title: test
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:04.50,Default,,0,0,0,,{\\an8}第一行\\N第二行
Dialogue: 0,0:00:04.20,0:00:08.00,Default,,0,0,0,,第二句，含逗号,还有,两个
Comment: 0,0:00:09.00,0:00:10.00,Default,,0,0,0,,注释行应忽略
"""
acues = parse_ass(ASS)
check("解析出 2 条（Comment 被忽略）", len(acues) == 2, str(len(acues)))
check("\\N 转空格", acues[0].text == "第一行 第二行", acues[0].text)
check("去 {\\an8}", "an8" not in acues[0].text, acues[0].text)
check(
    "Text 字段内的逗号保留", acues[1].text == "第二句，含逗号,还有,两个", acues[1].text
)
check(
    "时间 1000..4500",
    (acues[0].start_ms, acues[0].end_ms) == (1000, 4500),
    str(acues[0]),
)

section("自动分发 / 兜底")
check("按后缀 srt", len(parse_subtitle(SRT, suffix="srt")) == 3)
check("按内容识别 vtt", len(parse_subtitle(VTT, suffix="")) == 2)
check("按内容识别 ass", len(parse_subtitle(ASS, suffix="")) == 2)
check("后缀错但内容对（vtt 当 srt）", len(parse_subtitle(VTT, suffix="srt")) == 2)

# ------------------------------------------------------------------ markup
section("strip_markup")
check("空标签", strip_markup("<i></i>abc") == "abc", strip_markup("<i></i>abc"))
check("font 标签", strip_markup('<font color="#fff">红</font>') == "红")
check("全角标点保留", strip_markup("你好，世界！") == "你好，世界！")
check("零宽字符清除", strip_markup("a\u200bb") == "ab")

# --------------------------------------------------------------- merge_cues
section("merge_cues")
raw = [
    SubtitleCue(0, 1000, "第一句"),
    SubtitleCue(1050, 2000, "紧接着"),
    SubtitleCue(5000, 5100, "很短"),
    SubtitleCue(9000, 9000, "..."),
    SubtitleCue(9001, 9002, ""),
    SubtitleCue(12000, 12500, "交叠A"),
    SubtitleCue(12400, 13000, "交叠B"),
]
merged = merge_cues(raw, max_gap_ms=120, min_duration_ms=200)
check("间隔 50ms 被合并", merged[0].text == "第一句 紧接着", merged[0].text)
check(
    "合并后区间 0..2000",
    (merged[0].start_ms, merged[0].end_ms) == (0, 2000),
    str(merged[0]),
)
check("纯标点被丢弃", all("..." not in c.text for c in merged))
check("空文本被丢弃", all(c.text.strip() for c in merged))
short = [c for c in merged if c.text == "很短"][0]
check("过短片段补齐到 200ms", short.duration_ms >= 200, str(short.duration_ms))
check(
    "时间轴单调不交叠",
    all(merged[i].end_ms <= merged[i + 1].start_ms for i in range(len(merged) - 1)),
    str([(c.start_ms, c.end_ms) for c in merged]),
)
check(
    "index 重排",
    [c.index for c in merged] == list(range(len(merged))),
    str([c.index for c in merged]),
)

# ------------------------------------------------------------ 编码 & 文件
section("编码探测 / load_subtitles")
gbk_path = TMP / "gbk.srt"
gbk_path.write_bytes(
    "1\n00:00:00,000 --> 00:00:02,000\n编码测试中文\n".encode("gb18030")
)
check(
    "GB18030 被识别",
    detect_encoding(gbk_path) in {"gb18030", "big5"},
    detect_encoding(gbk_path),
)

u16_path = TMP / "utf16.vtt"
u16_path.write_bytes(
    "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nUTF16 内容\n".encode("utf-16")
)
check(
    "UTF-16 BOM 被识别",
    detect_encoding(u16_path) == "utf-16",
    detect_encoding(u16_path),
)

srt_path = TMP / "a.srt"
srt_path.write_bytes(
    SRT.encode("utf-8")
)  # 字节直写，避免 Windows 文本模式把 \n 再扩成 \r\n
vtt_path = TMP / "a.vtt"
vtt_path.write_bytes(VTT.encode("utf-8"))
hit_cues, hit_path, warns = load_subtitles([vtt_path, srt_path])
check("优先级 srt 高于 vtt", hit_path == srt_path, str(hit_path))
check("命中条数 3（join_text=False 不粘文本）", len(hit_cues) == 3, str(len(hit_cues)))
check("首条只含第一句", hit_cues[0].text == "你好，这是第一句。", hit_cues[0].text)
check("无告警", not warns, str(warns))

empty_path = TMP / "zz-empty.srt"  # 命名排在 a.srt 之后，单独验证"空字幕被跳过"
empty_path.write_text(
    "1\n00:00:00,000 --> 00:00:01,000\n\n", encoding="utf-8", newline=""
)
c2, p2, w2 = load_subtitles([srt_path, empty_path])
check("空字幕被跳过，仍命中 a.srt", p2 == srt_path, str(p2))
check("无候选以外不产生空字幕告警", not any("为空" in w for w in w2), str(w2))

c2b, p2b, w2b = load_subtitles([empty_path])
check("只有空字幕时返回空", c2b == [] and p2b is None, f"{c2b} {p2b}")
check("记录空字幕告警", any("为空" in w for w in w2b), str(w2b))

c3, p3, w3 = load_subtitles([])
check("无候选返回空", c3 == [] and p3 is None, f"{c3} {p3}")
check("无候选有告警", bool(w3), str(w3))

# ------------------------------------------------------------------ parser
section("parser: 远程")
s = parse_source("https://youtu.be/dQw4w9WgXcQ")
check("识别 youtube", s.platform is Platform.YOUTUBE, s.platform.value)
check("kind=remote_url", s.kind is SourceKind.REMOTE_URL)
check(
    "video_id 前缀 v", s.video_id.startswith("v") and len(s.video_id) == 17, s.video_id
)
check("requires_download", s.requires_download)

s2 = parse_source('  "https://www.bilibili.com/video/BV1xx411c7mD?p=2#t=30"  ')
check("引号被剥离", s2.raw.strip().startswith("https://"), s2.raw[:30])
check("fragment 被去掉", "#t=30" not in (s2.url or ""), s2.url or "")
check("query 保留", "p=2" in (s2.url or ""), s2.url or "")
check("平台 bilibili", s2.platform is Platform.BILIBILI, s2.platform.value)

check(
    "youtu.be vs youtube 不同 id",
    parse_source("https://youtu.be/x").video_id
    != parse_source("https://youtu.be/y").video_id,
)
check(
    "同 URL 稳定 id",
    parse_source("https://youtu.be/x").video_id
    == parse_source("https://youtu.be/x").video_id,
)
check(
    "裸域名路径：无 scheme 被拒",
    _err(lambda: parse_source("youtu.be/xxxx")) == "UnsupportedMediaError",
)


section("parser: 本地")
media_dir = (
    __import__("backend.runtime", fromlist=["runtime_path"]).runtime_path("tests/work")
    / "media"
)
video = next((p for p in media_dir.glob("*.mp4")), None)
if video is None:
    print("  (跳过：没有合成视频，先跑 t_media.py)")
else:
    ls = parse_source(str(video))
    check("本地 kind", ls.kind is SourceKind.LOCAL_FILE, ls.kind.value)
    check("本地有 probe", ls.info is not None)
    check(
        "duration 与 probe 一致",
        ls.duration_ms == ls.info.duration_ms if ls.info else False,
    )
    check("display_name 是文件名", ls.display_name == video.name, ls.display_name)
    check(
        "相对路径可解析",
        parse_source(video.name, ParseOptions(base_dir=video.parent)).path == video,
    )
    check(
        "file:// 前缀可解析", parse_source(video.as_uri()).path == video, video.as_uri()
    )
    check(
        "size_bytes 与磁盘一致",
        ls.size_bytes == video.stat().st_size,
        str(ls.size_bytes),
    )

    try:
        from backend.media.parser import parse_local

        too_large = False
        try:
            parse_local(video, max_bytes=10)
        except VideoTooLargeError:
            too_large = True
        check("max_bytes 触发 VideoTooLargeError", too_large)
        too_long = False
        try:
            parse_local(video, max_duration_ms=100)
        except Exception as exc:  # noqa: BLE001
            too_long = type(exc).__name__ == "VideoTooLongError"
            if not too_long:
                print(f"  实际异常: {type(exc).__name__}")
        check("max_duration_ms 触发 VideoTooLongError", too_long)
    except ImportError as exc:  # pragma: no cover
        print("  skip", exc)

print("  -- 本地错误分支 --")
check(
    "目录被拒",
    _err(lambda: parse_source(str(ROOT / "core"))) == "UnsupportedMediaError",
)
check(
    "不存在被拒",
    _err(lambda: parse_source(str(ROOT / "nope.mp4"))) == "UnsupportedMediaError",
)
check(
    "扩展名被拒",
    _err(lambda: parse_source(str(ROOT / "configs" / "models.yaml")))
    == "UnsupportedMediaError",
)
check("空输入被拒", _err(lambda: parse_source("   ")) == "UnsupportedMediaError")
check(
    "URL 超长被拒",
    _err(lambda: parse_source("https://x.com/" + "a" * 3000))
    == "UnsupportedMediaError",
)

section("parser: 工具函数")
check(
    "sanitize 去非法字符",
    sanitize_filename('a<b>:c/d\\e|f?g*h"i') == "a_b__c_d_e_f_g_h_i",
    sanitize_filename('a<b>:c/d\\e|f?g*h"i'),
)
check(
    "sanitize 空串回退", sanitize_filename("   ") == "video", sanitize_filename("   ")
)
check("sanitize 截断", len(sanitize_filename("x" * 500)) <= 120)
check("sanitize 保留中文", sanitize_filename("中文 文件名") == "中文 文件名")
check(
    "guess_platform 子域",
    guess_platform("https://m.youtube.com/watch?v=1") is Platform.YOUTUBE,
)
check(
    "guess_platform 未知",
    guess_platform("https://foo.example.com/a.mp4") is Platform.GENERIC,
)
check("derive_video_id 稳定", derive_video_id("abc") == derive_video_id("abc"))
check("derive_video_id 长度 17", len(derive_video_id("abc")) == 17)
ck1 = checksum_file(srt_path)
check("checksum 稳定", ck1 == checksum_file(srt_path))
check("checksum 区分内容", ck1 != checksum_file(vtt_path))
check("扩展名白名单含 mp4", "mp4" in DEFAULT_ALLOWED_EXTENSIONS)
check("扩展名白名单不含 exe", "exe" not in DEFAULT_ALLOWED_EXTENSIONS)

# --------------------------------------------------------------------- ASR
section("AsrConfig / 字幕路径转写")
cfg_default = AsrConfig()
check(
    "默认模型名 small",
    cfg_default.model_name == "faster-whisper-small",
    cfg_default.model_name,
)
check("默认设备空", cfg_default.device is None, str(cfg_default.device))

try:
    from backend.config import get_app_config

    app_cfg = get_app_config()
    asr_cfg = AsrConfig.from_config(app_cfg)
    # 断言"配置里写什么就读到什么"，而不是钉死某个型号名 ——
    # 换模型（large-v3 → turbo 省内存）不该让这条测试变红。
    check(
        "from_config 读到模型",
        asr_cfg.model_name == app_cfg.asr.model,
        asr_cfg.model_name,
    )
    check("from_config 读到 beam_size", asr_cfg.beam_size >= 1, str(asr_cfg.beam_size))
    check(
        "from_config 字幕优先级是 tuple",
        isinstance(asr_cfg.subtitle_priority, tuple),
        type(asr_cfg.subtitle_priority).__name__,
    )
    from backend.media.downloader import DownloadConfig

    dl_cfg = DownloadConfig.from_config(app_cfg)
    check(
        "DownloadConfig.videos_dir",
        str(dl_cfg.videos_dir).endswith("videos"),
        str(dl_cfg.videos_dir),
    )
    check(
        "DownloadConfig.max_retries=3", dl_cfg.max_retries == 3, str(dl_cfg.max_retries)
    )
    check(
        "DownloadConfig.backoff_base",
        dl_cfg.backoff_base > 1.0,
        str(dl_cfg.backoff_base),
    )
    from backend.media.parser import ParseOptions as PO

    po = PO.from_config(app_cfg)
    check(
        "ParseOptions.max_url_length=2048",
        po.max_url_length == 2048,
        str(po.max_url_length),
    )
    check(
        "ParseOptions.max_duration_ms=2h",
        po.max_duration_ms == 7_200_000,
        str(po.max_duration_ms),
    )
    check(
        "ParseOptions.base_dir 绝对",
        Path(po.base_dir).is_absolute() if po.base_dir else False,
        str(po.base_dir),
    )
except Exception as exc:  # pragma: no cover
    print("  skip config:", type(exc).__name__, exc)

engine = AsrEngine(AsrConfig(device="cpu"))
tr = engine.transcribe(
    "unused.wav", video_id="vTEST01", subtitles=[srt_path], duration_ms=9000
)
check("source=subtitle", tr.source == "subtitle", tr.source)
check("段数 3", len(tr.segments) == 3, str(len(tr.segments)))
check("末段 end=9000", tr.segments[-1].end_ms == 9000, str(tr.segments[-1].end_ms))
check(
    "时间单调",
    all(
        tr.segments[i].end_ms <= tr.segments[i + 1].start_ms
        for i in range(len(tr.segments) - 1)
    ),
)
check("text 拼接非空", bool(tr.text), tr.text[:40])
check("language=zh", tr.language == "zh", tr.language)
check("elapsed_ms 有值", tr.elapsed_ms >= 0, str(tr.elapsed_ms))
check("subtitle_path 记录", tr.subtitle_path == srt_path, str(tr.subtitle_path))
check("to_dict 可序列化", isinstance(tr.to_dict()["segments"], list))

from backend.models.asr import AsrError

try:
    engine.transcribe("unused.wav", video_id="vT2", subtitles=[], duration_ms=1000)
except AsrError as exc:
    check("无字幕且音频缺失必须失败", "音频" in str(exc))
else:
    check("无字幕且音频缺失必须失败", False)

only_sub = engine.transcribe_subtitles_only("vT3", [vtt_path], duration_ms=5500)
check("subtitles_only 命中", only_sub.source == "subtitle", only_sub.source)

check("Transcript.is_empty", Transcript(video_id="x").is_empty is True)
check("Transcript.char_count", Transcript(video_id="x").char_count == 0)

# 幻觉清洗
hall_srt = TMP / "hall.srt"
hall_srt.write_text(
    "1\n00:00:00,000 --> 00:00:02,000\n请不吝点赞，订阅，转发，打赏支持明镜与点点栏目\n\n"
    "2\n00:00:02,000 --> 00:00:04,000\n真正的正文内容\n\n"
    "3\n00:00:04,000 --> 00:00:06,000\n真正的正文内容\n",
    encoding="utf-8",
)
tr_h = AsrEngine(AsrConfig()).transcribe("x.wav", video_id="vH", subtitles=[hall_srt])
check(
    "幻觉行被丢弃",
    all("点赞" not in s.text for s in tr_h.segments),
    str([s.text for s in tr_h.segments]),
)
check("不同时间的相同原文保留溯源", len(tr_h.segments) == 2, str([s.text for s in tr_h.segments]))
check("保留正常文本", tr_h.segments[0].text == "真正的正文内容", tr_h.segments[0].text)

print("\n  -- 下载器路径绝对化 --")
# 回归：曾因直接取 paths.videos 的相对字符串，yt-dlp 把它解析到
# paths.temp 基准目录下，产出 data\temp\data\videos\... 双重路径并触发重试。
from backend.config import load_app_config as _lac  # noqa: PLC0415
from backend.media.downloader import DownloadConfig as _DC  # noqa: PLC0415

_cfg = _lac(root=str(ROOT))
_dl_cfg = _DC.from_config(_cfg)
check("videos_dir 为绝对路径", _dl_cfg.videos_dir.is_absolute(), _dl_cfg.videos_dir)
check(
    "temp_dir 为绝对路径", (_dl_cfg.temp_dir or Path()).is_absolute(), _dl_cfg.temp_dir
)
check(
    "videos_dir 与 cfg.path 一致",
    _dl_cfg.videos_dir == _cfg.path("videos"),
    (_dl_cfg.videos_dir, _cfg.path("videos")),
)

print(f"\n{'=' * 46}\nPASS {PASS}  FAIL {len(FAIL)}\n{'=' * 46}")
for item in FAIL:
    print(" FAIL:", item, file=sys.stderr)
sys.exit(1 if FAIL else 0)
