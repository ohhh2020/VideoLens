"""密钥保存与云端认证失败回归：不访问真实云端。"""

from pathlib import Path
import io
import json
import sys
import tempfile
import urllib.error
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.user_settings import (
    SettingsError,
    current_settings,
    save_settings,
    settings_file,
)
from backend.models.llm import (
    OpenAICompatibleBackend,
    LlmMessage,
    LlmAuthenticationError,
)
from backend.models.vl import (
    OpenAiVisionRunner,
    QwenVideoRunner,
    VlService,
    VlAuthenticationError,
)
from PIL import Image

passed = 0


def check(label, value):
    global passed
    assert value, label
    passed += 1
    print("PASS " + label)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    secret = "sk-test-complete-credential-123456"
    save_settings(
        {"llm.mode": "cloud", "llm.cloud.api_key": secret, "vl.cloud_api_key": secret},
        root=root,
    )
    shown = current_settings(root)
    before = settings_file(root).read_bytes()
    for name in ["llm.cloud.api_key", "vl.cloud_api_key"]:
        try:
            save_settings({"llm.mode": "cloud", name: shown[name]}, root=root)
        except SettingsError as exc:
            check("拒绝显示掩码 " + name, "掩码" in str(exc) and secret not in str(exc))
        else:
            raise AssertionError("不应保存显示掩码")
        check(
            "坏密钥不产生部分写入 " + name, settings_file(root).read_bytes() == before
        )
    save_settings({"llm.cloud.model": "qwen-plus"}, root=root)
    raw = json.loads(settings_file(root).read_text(encoding="utf-8"))
    check(
        "修改其它设置保留两个完整密钥",
        raw["llm.cloud.api_key"] == raw["vl.cloud_api_key"] == secret,
    )
    save_settings({"vl.cloud_api_key": ""}, root=root)
    check(
        "明确清除密钥仍然有效",
        "vl.cloud_api_key"
        not in json.loads(settings_file(root).read_text(encoding="utf-8")),
    )

    frame = root / "frame.jpg"
    Image.new("RGB", (16, 16), "red").save(frame)

    def unauthorized(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://invalid.test/v1",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"error":{"message":"Invalid API-key"}}'),
        )

    backend = OpenAICompatibleBackend(
        "https://invalid.test/v1", "model", secret, max_retries=5
    )
    with patch("urllib.request.urlopen", side_effect=unauthorized) as transport:
        for _ in range(3):
            try:
                backend.complete([LlmMessage("user", "测试")])
            except LlmAuthenticationError:
                pass
            else:
                raise AssertionError("应报告认证错误")
        check("文本 401 只请求一次，不反复重试", transport.call_count == 1)

    runner = OpenAiVisionRunner(
        "https://invalid.test/v1", "model", secret, max_retries=5
    )
    service = VlService(runner, max_retries=5)
    with patch("urllib.request.urlopen", side_effect=unauthorized) as transport:
        first = service.describe_sequence([(frame, 0)])
        second = service.describe_sequence([(frame, 1000)])
        third = service.describe(frame)
        check("视觉认证失败保留原因", first.degraded and "401" in first.error)
        check(
            "视觉 401 后停止其它帧请求",
            transport.call_count == 1 and second.degraded and third.degraded,
        )
        check("永久认证错误不走外层重试", first.attempts == 1)

    with patch("urllib.request.urlopen") as transport:
        bad = OpenAICompatibleBackend(
            "https://invalid.test/v1", "model", "sk-a******1234"
        )
        try:
            bad.complete([LlmMessage("user", "测试")])
        except LlmAuthenticationError:
            pass
        else:
            raise AssertionError("应拒绝掩码")
        visual = VlService(
            OpenAiVisionRunner("https://invalid.test/v1", "model", "sk-a******1234")
        )
        check("掩码视觉密钥本地拦截", visual.describe(frame).degraded)
        segment = QwenVideoRunner("https://invalid.test/v1", "model", "sk-a******1234")
        try:
            segment.describe_segment(root / "video.mp4", 0, 1000, "描述")
        except VlAuthenticationError:
            pass
        else:
            raise AssertionError("视频段不应编码或发送掩码密钥")
        check("掩码密钥不发送网络请求", transport.call_count == 0)

    # 视频段认证错误也应停止后续视频段及图片回退请求。
    video = root / "video.mp4"
    video.write_bytes(b"fixture")
    segment = QwenVideoRunner("https://invalid.test/v1", "model", secret, max_retries=5)

    def fake_encode(source, start, end, dest):
        dest.write_bytes(b"fixture")
        return dest

    segment.encode_segment = fake_encode
    sequence = VlService(
        OpenAiVisionRunner("https://invalid.test/v1", "model", secret),
        segment_runner=segment,
        max_retries=5,
    )
    with patch("urllib.request.urlopen", side_effect=unauthorized) as transport:
        a = sequence.describe_video_segment(video, 0, 1000)
        b = sequence.describe_video_segment(video, 1000, 2000)
        c = sequence.describe(frame)
        check(
            "视频段认证错误停止重试与图片回退",
            transport.call_count == 1 and a.degraded and b.degraded and c.degraded,
        )
        check("视频段认证原因可见", "401" in a.error and a.attempts == 1)

    def missing_model(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://invalid.test/v1",
            404,
            "Not Found",
            {},
            io.BytesIO(b'{"error":{"message":"Model not exist"}}'),
        )

    missing = VlService(
        OpenAiVisionRunner("https://invalid.test/v1", "missing-model", secret),
        max_retries=5,
    )
    with patch("urllib.request.urlopen", side_effect=missing_model) as transport:
        a = missing.describe(frame)
        b = missing.describe(frame)
        check(
            "模型不存在时不重复重试其它帧",
            transport.call_count == 1 and a.degraded and b.degraded,
        )
        check(
            "模型不存在提示可执行修法",
            "模型不存在" in a.error and "模型清单" in a.error,
        )
print(f"ALL PASSED {passed} checks")
