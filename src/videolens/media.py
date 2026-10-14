"""媒体处理：所有时间都用整数毫秒，所有外部命令都用参数数组执行。

本模块只负责取得可验证的媒体产物，不负责数据库或任务状态。调用方传入的
``check`` 会在耗时循环中执行；它可以抛出暂停/取消异常，我们负责关闭子进程。
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

Check = Callable[[], None]


class MediaError(RuntimeError):
    """输入媒体不可用，或底层媒体工具失败。"""


def _run(args: list[str], check: Check = lambda: None, timeout: float = 1800) -> tuple[str, str]:
    """等待工具时定期检查取消；communicate 同时排空两根管道，防止死锁。"""
    check()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   encoding="utf-8", errors="replace", creationflags=flags)
    except OSError as exc:
        raise MediaError(f"无法启动媒体工具 {Path(args[0]).name}：请运行一键配置环境。") from exc
    started = time.monotonic()
    try:
        while True:
            check()
            if time.monotonic() - started > timeout:
                raise MediaError("媒体工具执行超时。")
            try:
                stdout, stderr = process.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        # yt-dlp 可能启动 FFmpeg 合并音视频；只终止这个 Popen 所属的子进程树。
        # 不按可执行文件名查杀，避免误伤用户正在使用的播放器或其他任务。
        try:
            import psutil
            for child in reversed(psutil.Process(process.pid).children(recursive=True)):
                try:
                    child.kill()
                except psutil.Error:
                    pass
        except Exception:
            pass
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            # 缺少 psutil 时至少关闭管道；原取消异常始终保留。
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
        raise
    if process.returncode:
        # stderr 不直接向外暴露，避免第三方工具错误信息夹带 URL 令牌或本机路径。
        raise MediaError(f"{Path(args[0]).name} 执行失败（退出码 {process.returncode}）。请检查文件是否损坏、格式是否支持及工具版本。")
    check()
    return stdout, stderr


def probe(path: Path, settings: Any, check: Check = lambda: None) -> dict:
    """读取真实媒体流，而不是根据扩展名猜测时长、音轨或字幕。"""
    path = Path(path)
    if not path.is_file():
        raise MediaError("视频文件不存在。")
    if path.stat().st_size > settings.max_video_bytes:
        raise MediaError("视频超过配置的最大文件大小。")
    out, _ = _run([settings.ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], check, 60)
    data = json.loads(out)
    streams = data.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
    if not videos:
        raise MediaError("文件中没有可用的视频流。")
    video = videos[0]
    try:
        duration = float(data.get("format", {}).get("duration", video.get("duration", 0)))
    except (TypeError, ValueError) as exc:
        raise MediaError("无法读取视频时长。") from exc
    if not math.isfinite(duration) or duration <= 0 or duration > settings.max_video_seconds:
        raise MediaError("视频时长无效或超过配置的最大时长。")
    return {"duration_ms": round(duration * 1000), "width": int(video.get("width", 0)),
            "height": int(video.get("height", 0)), "has_audio": any(s.get("codec_type") == "audio" for s in streams),
            "subtitle_streams": [{"index": s["index"], "language": s.get("tags", {}).get("language", "und"),
                                  "codec": s.get("codec_name", "")} for s in streams if s.get("codec_type") == "subtitle"]}


def download(url: str, target_dir: Path, settings: Any, check: Check = lambda: None) -> Path:
    """使用 yt-dlp 下载；认证平台是否可访问由该平台和用户网络决定。"""
    parsed = urlparse(url)
    if len(url) > 2048 or parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise MediaError("请输入不超过 2048 字符的 HTTP/HTTPS 视频链接。")
    if not settings.allow_video_download:
        raise MediaError("在线视频下载已在设置中关闭。")
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-progress", "--newline",
            "--retries", "3", "--fragment-retries", "3", "--retry-sleep", "exp=1:8", "--continue",
            "--socket-timeout", "30", "--max-filesize", str(settings.max_video_bytes),
            # 直链的初始元数据可能没有时长；问号允许先下载，再由 probe 严格验证。
            "--match-filter", f"duration <=? {settings.max_video_seconds}",
            "--write-subs", "--write-auto-subs", "--sub-langs", "zh.*,en.*", "--sub-format", "srt/vtt/ass/best",
            "--ffmpeg-location", settings.ffmpeg, "--merge-output-format", "mkv",
            "-o", str(target_dir / "source.%(ext)s"), "--print", "after_move:filepath", "--", url]
    stdout, _ = _run(args, check, timeout=14400)
    for line in reversed(stdout.splitlines()):
        candidate = Path(line.strip()).resolve()
        if candidate.is_relative_to(target_dir) and candidate.is_file() and candidate.suffix.lower() not in {".srt", ".vtt", ".ass"}:
            probe(candidate, settings, check)
            return candidate
    raise MediaError("下载器未产生可用视频，可能被时长或大小限制过滤。")


def extract_audio(path: Path, out: Path, settings: Any, check: Check = lambda: None) -> Path:
    """转写统一使用 16kHz、单声道 PCM，避免模型重复进行格式转换。"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run([settings.ffmpeg, "-nostdin", "-y", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)], check)
    return out


def _milliseconds(value: str) -> int:
    """支持 SRT/VTT 的 hh:mm:ss.mmm 和 ASS 的 h:mm:ss.cc。"""
    parts = value.strip().replace(",", ".").split(":")
    if len(parts) == 2:
        parts.insert(0, "0")
    if len(parts) != 3:
        raise ValueError("字幕时间格式错误。")
    return round((int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])) * 1000)


def _subtitle_text(value: str) -> str:
    value = re.sub(r"\{[^}]*\}", "", value).replace(r"\N", "\n").replace(r"\n", "\n").replace(r"\h", " ")
    value = re.sub(r"<[^>]+>", "", value)
    return " ".join(html.unescape(value).split())


def read_subtitles(path: Path) -> list[dict]:
    """解析字幕文本，输出有序、非空、正时长的规范化记录。

    自己生成的文件始终为 UTF-8；外部输入兼容带 BOM 的 UTF-16 与 GB18030。
    """
    data = Path(path).read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("gb18030")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    rows: list[tuple[int, int, str]] = []
    if Path(path).suffix.lower() in {".ass", ".ssa"}:
        fields = ["layer", "start", "end", "style", "name", "marginl", "marginr", "marginv", "effect", "text"]
        in_events = False
        for line in text.splitlines():
            if line.strip().startswith("["):
                in_events = line.strip().lower() == "[events]"
            elif in_events and line.lower().startswith("format:"):
                fields = [v.strip().lower() for v in line.split(":", 1)[1].split(",")]
            elif in_events and line.lower().startswith("dialogue:"):
                values = line.split(":", 1)[1].lstrip().split(",", len(fields) - 1)
                entry = dict(zip(fields, values))
                try:
                    rows.append((_milliseconds(entry["start"]), _milliseconds(entry["end"]), _subtitle_text(entry.get("text", ""))))
                except (KeyError, ValueError):
                    continue
    else:
        timing = re.compile(r"(?P<start>\d{1,3}:\d{2}(?::\d{2})?[.,]\d{1,3})\s*-->\s*(?P<end>\d{1,3}:\d{2}(?::\d{2})?[.,]\d{1,3})[^\n]*\n(?P<body>.*?)(?=\n\s*\n|\Z)", re.S)
        for match in timing.finditer(text.strip() + "\n"):
            rows.append((_milliseconds(match["start"]), _milliseconds(match["end"]), _subtitle_text(match["body"])))
    rows = sorted({(max(0, a), b, t) for a, b, t in rows if b > max(0, a) and t})
    return [{"id": f"s{i:06d}", "start_ms": a, "end_ms": b, "text": text} for i, (a, b, text) in enumerate(rows, 1)]


def find_subtitles(video: Path, work: Path, settings: Any, check: Check = lambda: None) -> list[dict]:
    """外部同名字幕优先，其次从内嵌文本字幕轨提取；位图字幕留给视觉通道。"""
    video, work = Path(video), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    preferred = settings.subtitle_language.lower()
    candidates = [p for p in video.parent.iterdir() if p.is_file() and p.suffix.lower() in {".srt", ".vtt", ".ass", ".ssa"}
                  and (p.stem == video.stem or p.name.startswith(video.stem + "."))]
    candidates.sort(key=lambda p: (not any(part == preferred or part.startswith(preferred + "-") for part in p.stem.lower().split(".")), p.stem != video.stem, p.name))
    for candidate in candidates:
        check()
        try:
            rows = read_subtitles(candidate)
        except (UnicodeError, ValueError):
            continue
        if rows:
            return rows
    streams = probe(video, settings, check)["subtitle_streams"]
    languages = {"zh": {"zh", "zho", "chi", "zh-cn", "zh-hans"}, "en": {"en", "eng"}}.get(preferred, {preferred})
    streams.sort(key=lambda s: s["language"].lower() not in languages)
    for stream in streams:
        check()
        if stream["codec"] not in {"subrip", "ass", "ssa", "webvtt", "mov_text", "text"}:
            continue
        output = work / f"subtitle-{stream['index']}.srt"
        try:
            _run([settings.ffmpeg, "-nostdin", "-y", "-v", "error", "-i", str(video), "-map", f"0:{stream['index']}", "-c:s", "srt", str(output)], check)
            rows = read_subtitles(output)
        except (MediaError, UnicodeError, ValueError):
            continue
        if rows:
            return rows
    return []


def make_clips(segments: list[dict], duration_ms: int, settings: Any) -> list[dict]:
    """按句末/间隔靠近目标长度切片，同时保证完整覆盖视频时间线。

    跨越切片边界的字幕会在相邻切片中引用同一 segment_id，不凭空切断原文。
    """
    if duration_ms <= 0:
        return []
    minimum, maximum = int(settings.clip_min_seconds * 1000), int(settings.clip_max_seconds * 1000)
    target = max(minimum, min(maximum, int(settings.clip_target_seconds * 1000)))
    if minimum <= 0 or maximum < minimum:
        raise ValueError("切片最小值和最大值配置无效。")
    ordered = sorted(segments, key=lambda s: (s["start_ms"], s["end_ms"]))
    boundaries: list[tuple[int, bool]] = []
    for index, segment in enumerate(ordered):
        end = min(duration_ms, int(segment["end_ms"]))
        next_start = ordered[index + 1]["start_ms"] if index + 1 < len(ordered) else duration_ms
        natural = bool(re.search(r"[。！？.!?；;]$", segment["text"].strip())) or next_start - end >= 500
        boundaries.append((end, natural))
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < duration_ms:
        if duration_ms - start <= maximum:
            ranges.append((start, duration_ms))
            break
        candidates = [(end, natural) for end, natural in boundaries if start + minimum <= end <= start + maximum]
        if candidates:
            end, _ = min(candidates, key=lambda item: (abs(item[0] - start - target) + (0 if item[1] else target // 3), item[0]))
        else:
            end = start + target
        # 避免最后剩下极短片段；不得因此超过上限。
        if 0 < duration_ms - end < minimum and duration_ms - start <= maximum:
            end = duration_ms
        ranges.append((start, end))
        start = end
    clips = []
    for index, (start, end) in enumerate(ranges, 1):
        covered = [s for s in ordered if s["start_ms"] < end and s["end_ms"] > start]
        clips.append({"id": f"c{index:06d}", "start_ms": start, "end_ms": end,
                      "text": " ".join(s["text"] for s in covered), "segment_ids": [s["id"] for s in covered]})
    return clips


def _scene_times(video: Path, settings: Any, check: Check) -> list[int]:
    """showinfo 给出被选中帧的实际 PTS；无需把整个视频解码成图片。"""
    _, stderr = _run([settings.ffmpeg, "-nostdin", "-hide_banner", "-i", str(video), "-an",
                      "-vf", f"select=gt(scene\\,{float(settings.scene_threshold)}),showinfo", "-vsync", "vfr", "-f", "null", "-"], check)
    return sorted({round(float(value) * 1000) for value in re.findall(r"pts_time:([\d.]+)", stderr)})


def extract_frames(video: Path, clips: list[dict], out_dir: Path, settings: Any, check: Check = lambda: None) -> list[dict]:
    """场景帧优先，均匀采样补足；近似相同的静态画面在每个切片内去重。"""
    from PIL import Image
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = _scene_times(Path(video), settings, check)
    results: list[dict] = []
    frame_count = max(1, min(5, int(settings.frames_per_clip)))
    for clip in clips:
        check()
        start, end = clip["start_ms"], clip["end_ms"]
        middle = (start + end) / 2
        scene_candidates = sorted([t for t in scenes if start <= t < end], key=lambda t: abs(t - middle))
        uniform = [round(start + (end - start) * (i + 0.5) / frame_count) for i in range(frame_count)]
        candidates = list(dict.fromkeys(scene_candidates[:frame_count] + uniform))
        accepted_hashes: list[tuple[int, tuple[int, int, int]]] = []
        for timestamp in candidates:
            check()
            temp = out_dir / "candidate.png"
            _, stderr = _run([settings.ffmpeg, "-nostdin", "-y", "-hide_banner", "-ss", f"{timestamp / 1000:.3f}",
                              "-copyts", "-i", str(video), "-an", "-vf", "showinfo", "-frames:v", "1", str(temp)], check)
            if not temp.exists():
                continue
            with Image.open(temp) as image:
                image = image.convert("RGB")
                thumbnail = image.resize((8, 8)).convert("L")
                small = [thumbnail.getpixel((x, y)) for y in range(8) for x in range(8)]
                mean = sum(small) / len(small)
                fingerprint = sum((1 << i) for i, p in enumerate(small) if p >= mean)
                # aHash 对纯色都相同，加入平均颜色距离，防止漏掉红/蓝幻灯片切换。
                colour = tuple(image.resize((1, 1)).getpixel((0, 0)))
                signature = (fingerprint, colour)
                if any((fingerprint ^ old[0]).bit_count() <= 2 and sum(abs(a - b) for a, b in zip(colour, old[1])) < 30 for old in accepted_hashes):
                    temp.unlink(missing_ok=True)
                    continue
                accepted_hashes.append(signature)
                image.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
                frame_id = f"f{len(results) + 1:06d}"
                output = out_dir / f"{frame_id}.jpg"
                for quality in (90, 80, 70, 60, 50, 40):
                    image.save(output, format="JPEG", quality=quality, optimize=True)
                    if output.stat().st_size <= settings.frame_max_bytes:
                        break
                actual = re.findall(r"pts_time:([\d.]+)", stderr)
                actual_ms = round(float(actual[0]) * 1000) if actual else timestamp
                results.append({"id": frame_id, "clip_id": clip["id"], "timestamp_ms": actual_ms,
                                "path": str(output), "width": image.width, "height": image.height,
                                "size_bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                                "timestamp_verified": bool(actual)})
            temp.unlink(missing_ok=True)
            if len(accepted_hashes) >= frame_count:
                break
    return results
