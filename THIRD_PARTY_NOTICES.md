# 第三方来源与许可证说明

本文件记录设计参考、运行依赖、模型和验证素材。自编写实现独立组织为 Python 单机应用，第三方许可证保留其原始语言；中文说明用于解释来源和使用方式，不替代许可证原文。

## 1. 两个参考项目

| 项目 | 固定参考提交 | 参考方式 | 许可原文 |
|---|---|---|---|
| [dirge2024/VideoHub](https://github.com/dirge2024/VideoHub) | `e1336f2936c6a4a4a20b4325170da436d24f8d81` | 参考异步任务、时序上下文、检查点、分层检索、证据校验与执行预算，独立编写 Python 实现 | [MIT 原文](docs/licenses/VideoHub-MIT.txt) |
| [Zhang1933/WebSideChat](https://github.com/Zhang1933/WebSideChat) | `7a3ff656c8913deef14c7ce321795152b1497af8` | 参考模型设置、提示词、流式输出与会话隔离的交互思路；不复制代码、图标、页面素材或文案 | [CC BY-NC 4.0 原文](docs/licenses/WebSideChat-CC-BY-NC-4.0.txt) |

VideoHub 的 MIT 许可允许在保留版权与许可声明条件下使用和修改。当前实现没有直接移植其源文件；若后续实际改编代码，需在对应文件或来源记录中注明原路径、提交及改动，并继续保留许可。

WebSideChat 的 CC BY-NC 4.0 包含署名与非商业条件。本项目使用独立实现的界面和后端，参考功能行为；这不表示取得了复制该项目代码或素材用于其他用途的额外许可。

## 2. 运行依赖

主要依赖包括 FastAPI、Uvicorn、Gradio、HTTPX、Pydantic、PyTorch、Transformers、Sentence Transformers、Chroma、faster-whisper、CTranslate2、Hugging Face Hub、Pillow、jieba、pysubs2、yt-dlp、keyring、psutil、filelock，以及开发工具 pytest 和 Ruff。

这些依赖通过 Miniconda/pip 安装，不作为复制的源码合并到本项目。各包的 LICENSE、NOTICE 与元信息随发行包保留；实际安装版本见 `data/environment-backups/requirements-installed.txt`。发布二进制或重新分发依赖时，应以对应实际版本的许可为准。

FFmpeg 使用 Conda 包或 [FFmpeg 官网列出的 Gyan Windows 构建](https://ffmpeg.org/download.html)。静态回退固定为 Gyan 8.1.2 essentials，ZIP 的 SHA-256 为 `db580001caa24ac104c8cb856cd113a87b0a443f7bdf47d8c12b1d740584a2ec`。该构建包含 GPL 组件；安装时保留完整许可文件与构建说明。程序通过外部命令调用 FFmpeg，不宣称自己的许可覆盖 FFmpeg。

llama.cpp 来自 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)，采用 MIT 许可。默认使用固定 `b11612` Windows CPU 发行包，保留包内声明并记录下载来源及 SHA-256。CUDA 发行包及运行库可选，需遵守其各自随附许可。

## 3. 模型来源

| 角色 | 仓库与固定 revision | 用途 |
|---|---|---|
| ASR | `Systran/faster-whisper-small` / `536b0662742c02347bc0e980a01041f333bce120` | 本地语音转写 |
| 文本向量 | `BAAI/bge-m3` / `5617a9f61b028005a4858fdac845db406aefb181` | 文本语义检索 |
| 图文向量 | `google/siglip2-base-patch16-224` / `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | 图像与文字对齐检索 |
| 本地生成 | `Qwen/Qwen3-4B-GGUF` / `bc640142c66e1fdd12af0bd68f40445458f3869b` | Q4_K_M 量化本地问答 |

模型由用户本机下载到 `models/`，不提交 Git。下载器保留可用的模型卡和 LICENSE，并将固定版本、文件大小、官方大文件哈希及本地校验值记入 `model-manifest.json`。模型的使用条件以对应版本模型卡与许可为准，模型许可与推理框架许可是不同层次。

Qwen GGUF 文件名 `Qwen3-4B-Q4_K_M.gguf`，固定 SHA-256 为 `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`。

## 4. 测试素材

项目自行生成的色块、场景切换、字幕和时间线仅用于媒体处理验收。`scripts/download_test_audio.py` 另提供固定来源的真实人声短片段，不把静音或合成语音当成公开真实语料。

- 中文：FunASR 仓库 `runtime/funasr_api/asr_example.wav`，固定提交 `f793eae5778eae93854646f9b4b9c42a23584651`；仓库许可 MIT，音频没有独立许可文件，来源记录明确保留这一边界。对应中文演示参考文本来自上游 README，首次评测应人工核对。
- 英文：whisper.cpp 仓库 `samples/jfk.wav`，固定提交 `0c7be696371069e0baa2515ed7aa08d5a5473aa6`；来源为 JFK 1961 年就职演说节选，美国联邦政府公务作品，仓库为 MIT。参考演讲文字来自 JFK Library。

下载脚本生成 `data/fixtures/public-audio/sources.json` 和原始参考文本，并校验固定 SHA-256。素材只用于本机验证，不放入源码分发。两段短样本的 CER/WER 不能代表普遍中英文准确率；若需要发表或商业评测，应另选具有明确音频级授权与标准划分的语料。
