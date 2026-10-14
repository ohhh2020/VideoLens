"""安装依赖并区分版本兼容失败与普通网络失败。

退出码 42 专门表示检测到 Python 版本不兼容；PowerShell 只有收到它才降级 Python。
避免把断网或代理故障误判为 Python 3.13 不兼容，从而无故改动现有环境。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="安装知影依赖，识别 Python 兼容问题")
    parser.add_argument("--minimal", action="store_true")
    args = parser.parse_args()
    extras = ".[dev]" if args.minimal else ".[heavy,dev]"
    command = [sys.executable, "-m", "pip", "install", "--retries", "3", "-e", extras]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace")
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    code = process.wait()
    if code == 0:
        return 0
    output = "".join(lines)
    # 无法找到版本也可能来自断网；仅明确出现 Python 版本条件时才建议换版本。
    compatibility = re.search(r"requires a different Python|not in ['\"]?[><=!].*|Requires-Python.*(?:<3\.13|<=3\.12)",
                              output, flags=re.IGNORECASE)
    return 42 if compatibility else code


if __name__ == "__main__":
    raise SystemExit(main())
