#调用 analyze_url 分析视频
"""
用 argparse 接收一个视频链接，
通过 analyze_url 分析，
并用一个实时刷新的字符进度条在终端展示进度，
最后把标题、输出目录、总结和生成的文件列出来。

"""
from __future__ import annotations

import argparse                     #解析命令行参数

from src.pipeline import analyze_url


def main() -> None:
    parser = argparse.ArgumentParser(description="VideoLens 视频链接智能总结系统")
    parser.add_argument("url", help="视频链接或本地 MP4 路径")
    args = parser.parse_args()

    def progress(percent: float, message: str) -> None:
        bar = "█" * int(percent * 30)
        print(f"\r[{bar:<30}] {percent:5.1%} {message}", end="", flush=True)

    result = analyze_url(args.url, progress_callback=progress)
    print("\n")
    print("=" * 60)
    print("视频标题：", result["metadata"].get("title"))
    print("输出目录：", result["output_dir"])
    print("=" * 60)
    print(result["summary"].get("markdown", ""))
    print("\n结果文件：")
    print("  result.json")
    print("  summary.md")
    print("  transcript.json")
    print("  captions.json")
    print("  keyframes/")


if __name__ == "__main__":
    main()

