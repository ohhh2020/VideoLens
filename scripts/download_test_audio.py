"""取得短小的真实中英文语音并保存来源、许可证、参考文本及固定 SHA-256。

素材只写入被 Git 忽略的 data/fixtures/public-audio，不加入产品分发包。
网络失败不会生成“验证成功”报告；实际 ASR 由 verify_real_models.py 另行执行。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from videolens.bootstrap import download_file
from videolens.config import atomic_json

FUNASR_COMMIT = "f793eae5778eae93854646f9b4b9c42a23584651"
WHISPER_COMMIT = "0c7be696371069e0baa2515ed7aa08d5a5473aa6"
SAMPLES = [
    {"id": "funasr_zh", "language": "zh", "filename": "funasr_zh.wav",
     "url": f"https://raw.githubusercontent.com/modelscope/FunASR/{FUNASR_COMMIT}/runtime/funasr_api/asr_example.wav",
     "sha256": "a1bd32dc78493c123f9625a66deee562aed2895f53fbc39f2cca3be7e6f4f20f",
     "reference": "欢迎大家来体验达摩院推出的语音识别模型",
     "reference_source": f"https://github.com/modelscope/FunASR/blob/{FUNASR_COMMIT}/README_zh.md",
     "license": "FunASR 仓库 MIT；仓库内演示音频未附独立许可声明，保留项目版权声明和来源。",
     "license_url": f"https://github.com/modelscope/FunASR/blob/{FUNASR_COMMIT}/LICENSE",
     "source_note": "模型仓库公开演示样本；单段结果只用于功能冒烟，不能作为中文测试集准确率。"},
    {"id": "jfk_en", "language": "en", "filename": "jfk_en.wav",
     "url": f"https://raw.githubusercontent.com/ggml-org/whisper.cpp/{WHISPER_COMMIT}/samples/jfk.wav",
     "sha256": "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e",
     "reference": "And so my fellow Americans ask not what your country can do for you ask what you can do for your country",
     "reference_source": "https://www.jfklibrary.org/learn/about-jfk/historic-speeches/inaugural-address",
     "license": "美国总统履行公务发表的就职演说，美国联邦政府作品；音频样本来自 MIT 许可 whisper.cpp 仓库。",
     "license_url": f"https://github.com/ggml-org/whisper.cpp/blob/{WHISPER_COMMIT}/LICENSE",
     "source_note": "1961-01-20 John F. Kennedy 就职演说节选；仅代表此 11 秒左右英文样本。"},
]
MIT_FUNASR = """MIT License
Copyright (c) 2025 FunASR

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="下载有固定来源及哈希的中英文真实人声示例")
    parser.add_argument("--output", type=Path, default=Path("data/fixtures/public-audio"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for sample in SAMPLES:
        target = args.output / sample["filename"]
        download_file(sample["url"], target, sample["sha256"])
        target.with_suffix(".txt").write_text(sample["reference"] + "\n", encoding="utf-8")
        print(f"素材已校验：{target.resolve()}")
    (args.output / "LICENSE-FunASR.txt").write_text(MIT_FUNASR, encoding="utf-8")
    atomic_json(args.output / "sources.json", {"samples": SAMPLES,
                "note": "参考文本应在首次评测时人工核对；此下载过程没有执行语音识别。"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
