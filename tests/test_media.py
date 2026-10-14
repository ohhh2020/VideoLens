"""媒体测试使用真实字幕与 FFmpeg；未安装工具时明确跳过真实媒体测试。"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from videolens.media import MediaError, _run, download, extract_audio, extract_frames, find_subtitles, make_clips, probe, read_subtitles


def settings(**changes):
    values = dict(ffmpeg="ffmpeg", ffprobe="ffprobe", max_video_bytes=4_000_000_000,
                  max_video_seconds=7200, subtitle_language="zh", clip_target_seconds=20,
                  clip_min_seconds=10, clip_max_seconds=30, frames_per_clip=3,
                  frame_max_bytes=200000, scene_threshold=0.3, allow_video_download=True)
    values.update(changes)
    return SimpleNamespace(**values)


def test_srt_chinese_html_and_duplicate(tmp_path):
    path = tmp_path / "中文 & 字幕.srt"
    path.write_text("1\n00:00:01,250 --> 00:00:03,500\n<b>你好</b> &amp; 世界\n\n2\n00:00:01,250 --> 00:00:03,500\n<b>你好</b> &amp; 世界\n", encoding="utf-8")
    assert read_subtitles(path) == [{"id": "s000001", "start_ms": 1250, "end_ms": 3500, "text": "你好 & 世界"}]


def test_vtt_settings_and_short_timestamp(tmp_path):
    path = tmp_path / "字幕.vtt"
    path.write_text("WEBVTT\n\nlabel\n00:01.000 --> 00:02.500 align:start\n<v 说话人>第一行</v>\n第二行\n\n", encoding="utf-8")
    assert read_subtitles(path)[0] == {"id": "s000001", "start_ms": 1000, "end_ms": 2500, "text": "第一行 第二行"}


def test_ass_format_and_comma_inside_text(tmp_path):
    path = tmp_path / "字幕.ass"
    path.write_text("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.25,0:00:04.20,Default,,0,0,0,,{\\b1}第一行\\N第二行,逗号\n", encoding="utf-8")
    assert read_subtitles(path)[0]["text"] == "第一行 第二行,逗号"
    assert read_subtitles(path)[0]["end_ms"] == 4200


def test_empty_or_invalid_cues_are_skipped(tmp_path):
    path = tmp_path / "空.srt"
    path.write_text("1\n00:00:02,000 --> 00:00:01,000\n倒序\n\n2\n00:00:04,000 --> 00:00:05,000\n   \n", encoding="utf-8")
    assert read_subtitles(path) == []


def test_clips_cover_silent_video_without_gaps():
    clips = make_clips([], 95000, settings())
    assert clips[0]["start_ms"] == 0
    assert clips[-1]["end_ms"] == 95000
    assert all(a["end_ms"] == b["start_ms"] for a, b in zip(clips, clips[1:]))
    assert all(0 < c["end_ms"] - c["start_ms"] <= 30000 for c in clips)
    assert len({c["id"] for c in clips}) == len(clips)


def test_clips_prefer_sentence_boundary_and_preserve_evidence():
    segments = [{"id": "s000001", "start_ms": 1000, "end_ms": 18500, "text": "一个完整句子。"},
                {"id": "s000002", "start_ms": 19000, "end_ms": 35000, "text": "下一句话。"}]
    clips = make_clips(segments, 60000, settings())
    assert clips[0]["end_ms"] == 18500
    assert clips[0]["segment_ids"] == ["s000001"]
    assert set(sum((c["segment_ids"] for c in clips), [])) == {"s000001", "s000002"}


def test_cancel_terminates_subprocess_quickly():
    class Cancelled(Exception):
        pass
    started = time.monotonic()
    def check():
        if time.monotonic() - started > 0.3:
            raise Cancelled()
    with pytest.raises(Cancelled):
        _run([sys.executable, "-c", "import time; time.sleep(20)"], check)
    assert time.monotonic() - started < 4


def test_subprocess_errors_hide_private_paths():
    with pytest.raises(MediaError) as caught:
        _run([sys.executable, "-c", "import sys; print('secret=PRIVATE', file=sys.stderr); sys.exit(3)"])
    assert "PRIVATE" not in str(caught.value)
    assert "3" in str(caught.value)


@pytest.fixture
def ffmpeg_settings():
    import os
    ffmpeg = os.environ.get("VIDEOLENS_TEST_FFMPEG") or shutil.which("ffmpeg")
    ffprobe = os.environ.get("VIDEOLENS_TEST_FFPROBE") or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("真实 FFmpeg/FFprobe 未安装，不能把模拟测试当作媒体实测。")
    return settings(ffmpeg=ffmpeg, ffprobe=ffprobe)


def test_real_media_chinese_path_scene_frames_and_audio(tmp_path, ffmpeg_settings):
    cfg = ffmpeg_settings
    video = tmp_path / "中文 & 测试视频.mp4"
    subprocess.run([cfg.ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i", "color=red:s=640x360:r=10:d=2",
                    "-f", "lavfi", "-i", "color=blue:s=640x360:r=10:d=2", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=4",
                    "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a", "-c:v", "mpeg4", "-c:a", "aac", str(video)], check=True, capture_output=True)
    info = probe(video, cfg)
    assert info["duration_ms"] == 4000 and info["has_audio"]
    assert info["width"] == 640 and info["height"] == 360
    audio = extract_audio(video, tmp_path / "音轨.wav", cfg)
    import wave
    with wave.open(str(audio)) as wav:
        assert wav.getnchannels() == 1 and wav.getframerate() == 16000
    clips = make_clips([], info["duration_ms"], cfg)
    frames = extract_frames(video, clips, tmp_path / "关键帧", cfg)
    assert len(frames) == 2  # 红蓝两种画面，静态重复帧被去重。
    assert all(frame["timestamp_verified"] and frame["width"] == 640 and frame["size_bytes"] < 200000 for frame in frames)
    assert all(0 <= frame["timestamp_ms"] < 4000 for frame in frames)
    assert any(abs(frame["timestamp_ms"] - 2000) < 110 for frame in frames)


def test_external_chinese_subtitle_preferred(tmp_path):
    video = tmp_path / "source.mp4"
    video.touch()
    (tmp_path / "source.en.srt").write_text("1\n00:00:00,000 --> 00:00:02,000\nEnglish\n", encoding="utf-8")
    (tmp_path / "source.zh-Hans.srt").write_text("1\n00:00:00,000 --> 00:00:02,000\n中文\n", encoding="utf-8")
    assert find_subtitles(video, tmp_path / "工作", settings())[0]["text"] == "中文"


def test_real_embedded_chinese_and_english_subtitles_without_audio(tmp_path, ffmpeg_settings):
    cfg = ffmpeg_settings
    english, chinese = tmp_path / "英语.srt", tmp_path / "中文.srt"
    english.write_text("1\n00:00:00,000 --> 00:00:01,500\nEnglish subtitle\n", encoding="utf-8")
    chinese.write_text("1\n00:00:00,000 --> 00:00:01,500\n中文内嵌字幕\n", encoding="utf-8")
    video = tmp_path / "多语言.mkv"
    subprocess.run([cfg.ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i", "color=green:s=320x180:r=10:d=2",
                    "-i", str(english), "-i", str(chinese), "-map", "0:v", "-map", "1:0", "-map", "2:0",
                    "-c:v", "mpeg4", "-c:s", "srt", "-metadata:s:s:0", "language=eng", "-metadata:s:s:1", "language=zho",
                    "-t", "2", str(video)], check=True, capture_output=True)
    metadata = probe(video, cfg)
    assert not metadata["has_audio"]
    assert {row["language"] for row in metadata["subtitle_streams"]} == {"eng", "zho"}
    assert find_subtitles(video, tmp_path / "工作", cfg)[0]["text"] == "中文内嵌字幕"
    cfg.subtitle_language = "en"
    assert find_subtitles(video, tmp_path / "工作", cfg)[0]["text"] == "English subtitle"


def test_real_corrupt_video_is_rejected(tmp_path, ffmpeg_settings):
    path = tmp_path / "损坏的视频.mp4"
    path.write_bytes(b"This is not a valid media container.")
    with pytest.raises(MediaError):
        probe(path, ffmpeg_settings)


@pytest.fixture
def local_http_video(tmp_path, ffmpeg_settings):
    """真实 HTTP + 真实 yt-dlp；slow 路径故意限速，便于验证下载中途取消。"""
    pytest.importorskip("yt_dlp", reason="需要实际下载器才能验证下载行为")
    cfg = ffmpeg_settings
    source = tmp_path / "http-source.mp4"
    subprocess.run([cfg.ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=3",
                    "-an", "-c:v", "mpeg4", "-q:v", "2", str(source)], check=True, capture_output=True)
    payload = source.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()

        def do_GET(self):
            self.do_HEAD()
            try:
                for start in range(0, len(payload), 1024):
                    self.wfile.write(payload[start:start + 1024])
                    self.wfile.flush()
                    if self.path.startswith("/slow"):
                        time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", source, cfg
    server.shutdown()
    server.server_close()


def test_real_ytdlp_http_download_is_byte_exact_and_probeable(tmp_path, local_http_video):
    base_url, source, cfg = local_http_video
    result = download(base_url + "/fixture.mp4", tmp_path / "下载结果", cfg)
    assert hashlib.sha256(result.read_bytes()).digest() == hashlib.sha256(source.read_bytes()).digest()
    metadata = probe(result, cfg)
    assert metadata["duration_ms"] == 3000 and metadata["width"] == 640
    assert not metadata["has_audio"] and not metadata["subtitle_streams"]
    assert find_subtitles(result, tmp_path / "工作", cfg) == []


def test_real_ytdlp_cancel_stops_active_partial_download(tmp_path, local_http_video):
    import psutil
    base_url, _, cfg = local_http_video
    target = tmp_path / "取消下载"
    started = time.monotonic()
    cancelled_at, children = [], []

    class Cancelled(Exception):
        pass

    def check():
        # 只有已经产生真实 .part 内容才取消，避免只测到下载器尚未启动的情况。
        if any(path.stat().st_size > 0 for path in target.glob("*.part")):
            cancelled_at.append(time.monotonic())
            children.extend(psutil.Process().children(recursive=True))
            raise Cancelled()
        if time.monotonic() - started > 20:
            raise TimeoutError("下载未按预期生成可取消的部分文件")

    with pytest.raises(Cancelled):
        download(base_url + "/slow.mp4", target, cfg, check)
    assert cancelled_at and time.monotonic() - cancelled_at[0] < 4
    assert any(path.stat().st_size > 0 for path in target.glob("*.part"))
    assert not any(child.is_running() for child in children)
