"""测试夹具生成器 —— 用 ffmpeg 合成"可复现、体积小、场景可数"的视频。

为什么不用真实视频：真实样片无法入库（体积 + 版权），而合成片有确定性的性质，
断言可以写得很硬（例如"恰好 4 个场景、切点误差 0ms"）。

合成规则：
- ``demo.mp4``：40s / 1280x720 / 25fps，4 段视觉风格完全不同的 10s 画面
  （testsrc2 → smptebars → testsrc → rgbtestsrc）拼成，配 440Hz 正弦音轨。
  → 场景检测应恰好得到 4 段；抽帧短边 720 满足需求文档要求；有音轨可供 ASR 走通。
- ``low.mp4``：12s / 640x360，短边 360 < 720，用于验证"不放大、只告警"的降级路径。

这些文件被 ``.gitignore`` 排除（``tests/media/``），首次运行由 ``ensure()`` 现场生成，
因此克隆仓库后可直接 ``python tests/run_all.py``（需要 ffmpeg 在 PATH 上）。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent / "media"
DEMO = BASE / "demo.mp4"
LOW = BASE / "low.mp4"


def _run(args: list[str], expect: Path, *, verbose: bool = False) -> None:
    """执行一次 ffmpeg 合成；已存在且非空则跳过（幂等）。"""
    if expect.exists() and expect.stat().st_size > 1024:
        if verbose:
            print(f"skip {expect.name} ({expect.stat().st_size} bytes)")
        return
    BASE.mkdir(parents=True, exist_ok=True)
    if verbose:
        print("ffmpeg " + " ".join(args[:8]) + " …")
    proc = subprocess.run(  # noqa: S603 - 固定参数
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"生成 {expect.name} 失败（ffmpeg 退出码 {proc.returncode}）：\n{proc.stderr[-2000:]}"
        )
    if verbose:
        print(f"ok   {expect.name} ({expect.stat().st_size} bytes)")


def ensure(*, verbose: bool = False) -> tuple[Path, Path]:
    """确保两个夹具存在，返回 ``(demo, low)`` 路径。"""
    _run(
        [
            "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=25:d=10",
            "-f", "lavfi", "-i", "smptebars=s=1280x720:r=25:d=10",
            "-f", "lavfi", "-i", "testsrc=s=1280x720:r=25:d=10",
            "-f", "lavfi", "-i", "rgbtestsrc=s=1280x720:r=25:d=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=40",
            "-filter_complex",
            "[0:v]format=yuv420p[v0];[1:v]format=yuv420p[v1];"
            "[2:v]format=yuv420p[v2];[3:v]format=yuv420p[v3];"
            "[v0][v1][v2][v3]concat=n=4:v=1:a=0[v]",
            "-map", "[v]", "-map", "4:a",
            "-r", "25", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-shortest",
            str(DEMO),
        ],
        DEMO,
        verbose=verbose,
    )
    _run(
        [
            "-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=12",
            "-f", "lavfi", "-i", "sine=frequency=330:duration=12",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", "-shortest",
            str(LOW),
        ],
        LOW,
        verbose=verbose,
    )
    return DEMO, LOW


if __name__ == "__main__":
    try:
        demo, low = ensure(verbose=True)
    except RuntimeError as exc:
        print(exc)
        raise SystemExit(1) from exc
    print(f"fixtures ready:\n  {demo}\n  {low}")
    sys.exit(0)
