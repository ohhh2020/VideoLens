# 知影 VideoLens

知影是一款在个人电脑运行的视频智能分析助手。导入视频后，可以查看转录、关键帧、时间线和摘要，按文字或图片检索视频，再围绕同一视频连续提问。回答关联原始证据和时间戳，便于返回视频核对。

项目以 Python、FastAPI、Gradio、SQLite 和 Chroma 构建，默认只监听本机 `127.0.0.1`。语音转写和向量检索在本地执行；云端文本与视觉模型分别配置。缺少云端服务时保留可运行的本地流程，报告会注明哪些阶段采用了降级。

**本机已完成部署，可以直接双击 `一键启动.cmd`。** 当前复用 `VideoLens` 的 Python 3.13 环境，四类模型均已下载并完成真实推理；本机 llama.cpp 已启用经过验证的 RTX 5060 CUDA 路径，向量模型使用 CPU。最终自动化 **84 项通过**，短视频完整流程和一小时合成字幕视频专项均通过。真实云端尚未配置，启动与端到端首 Token 等性能目标仍有差距，详见 [测试报告](docs/测试报告.md)。

## 从零开始使用

1. 安装 **Miniconda**，确保可以使用 `conda`。已经有 `VideoLens` 环境时会复用，不会安装到 `base`。
2. 将整个项目放在固定目录，双击 **`一键配置环境.cmd`**。首次运行会备份现有环境，安装依赖和 FFmpeg，下载固定版本模型与 llama.cpp CPU 运行时，再执行真实 CPU 推理检查。模型下载需要网络和数 GB 磁盘空间，可以重跑以续传或复用缓存。
3. 双击 **`一键启动.cmd`**，等待模型与后端健康检查完成。浏览器打开 [知影网页](http://127.0.0.1:8000/app)。
4. 登录用户名为 `videolens`。登录密码是项目目录 `data/.access-token` 文件中的随机字符串，启动终端也会提示取用方式。该文件属于本地访问凭据，请勿发给他人。
5. 在“导入”页选择视频和可选字幕，点击“上传并分析”。在任务页查看阶段和降级原因，完成后到报告页查阅、定位、追问和导出。
6. 关闭时回到启动窗口按 `Ctrl+C`；启动脚本会清理这次启动所管理的本地模型服务。

Windows 的 Conda 有时会接着询问“终止批处理操作吗 (Y/N)”。本次验证使用 `N` 让外层脚本继续完成退出清理；若再次询问仍选择 `N`，直到看到模型服务 `stopped`。主动停止可能返回非零退出码，这不等于分析任务失败。不要直接强制关闭窗口代替正常清理。

**先试用轻量流程：**如果暂时不下载模型，可在 PowerShell 7 中执行 `pwsh -File scripts/setup.ps1 -Minimal -SkipModels -SkipRuntime`，随后执行 `pwsh -File scripts/start.ps1 -SkipLocal`。有字幕的视频仍能完成切片、关键帧、关键词检索和基础摘录；真实 ASR、向量和模型生成需要完整版依赖及权重。

所有命令都在项目根目录运行。下面的命令会明确选择 Miniconda 环境，无需手动激活：

```powershell
# 启动网页与 API，终端保持运行。
conda run --no-capture-output -n VideoLens python -m videolens serve --open-browser

# 另开终端，导入并等待分析完成；按自己的实际路径修改文件名。
conda run --no-capture-output -n VideoLens python -m videolens analyze "D:\视频\课程.mp4" --subtitle "D:\视频\课程.srt" --wait

# 使用上一条命令返回的视频 ID 替换 VIDEO_ID。
conda run --no-capture-output -n VideoLens python -m videolens search VIDEO_ID "主要结论"
conda run --no-capture-output -n VideoLens python -m videolens ask VIDEO_ID "这个视频的核心结论是什么？" --stream
conda run --no-capture-output -n VideoLens python -m videolens export VIDEO_ID --format markdown --output "我的视频笔记.md"
```

## 功能与默认边界

| 能力 | 实现方式与默认行为 |
|---|---|
| 视频输入 | MP4、MKV、AVI、MOV；yt-dlp 支持的 HTTP/HTTPS 链接，默认最大 2 小时/4GB |
| 语音与字幕 | SRT/VTT/ASS 字幕优先；无字幕时使用 faster-whisper-small，GPU 失败可退回 CPU |
| 画面分析 | 按场景变化与均匀采样提取关键帧；云端视觉接口提供描述与 OCR |
| 检索 | BGE-M3、SigLIP2 与关键词融合；向量失效时仍能关键词检索 |
| 报告与问答 | 云端文本模型优先，llama.cpp 本地 Qwen3-4B 降级，再降级为原文摘录 |
| 证据 | 引用 ID、原文、时间戳；后端校验后展示，实际内容仍建议对照原视频 |
| 任务 | 后台分析、阶段进度、暂停、恢复、取消、检查点及重启恢复 |
| 数据 | SQLite 主数据、可重建 Chroma 索引；受管媒体默认 7 天后可清理，原始文件不参与删除 |
| 输出 | UTF-8 Markdown、JSON；网页、CLI 与 `/api/v1` API |

首版面向 Windows 本机使用，外部平台视频下载能力受网站变化和访问条件影响。Obsidian、MCP、QQ-Bot、浏览器扩展和公网部署属于后续路线图。真实云端测试需要你配置有效服务；测试进度与实际证据见 [测试报告](docs/测试报告.md)，未验证的能力不会标成已通过。

## 配置与模型

网页“设置”中可配置文本/视觉供应商、模型名、提示词、切片、帧数、保留天数等。高级选项使用 `config.local.json`；可参考 [配置样例](config.example.json)。路径相对于配置文件所在目录解析，支持中文路径。

- `data/`：视频受管副本、数据库、临时产物、导出与测试记录。
- `models/`：Hugging Face 缓存及 `model-manifest.json`；记录模型提交、实际文件大小与 SHA-256。
- `runtime/`：llama.cpp 可执行文件、已校验的下载包、进程身份清单与运行日志。
- `config.local.json`：个人设置，不含云端 API Key。API Key 写入 Windows 凭据管理器；也支持后端环境变量。

```powershell
# 单独下载、校验模型，不要求网页已经启动。
conda run --no-capture-output -n VideoLens python -m videolens.bootstrap models
conda run --no-capture-output -n VideoLens python -m videolens.bootstrap verify-models

# 默认部署 CPU 运行时；GPU 可用性必须另做真实推理检查。
conda run --no-capture-output -n VideoLens python -m videolens.bootstrap runtime
conda run --no-capture-output -n VideoLens python -m videolens.bootstrap start-local
```

## 验证与开发

双击 **`一键测试.cmd`** 会运行环境冒烟及离线自动化测试。真实模型验证需已下载模型；真实云端验证需有效配置，均与离线测试分开记录。

完整配置脚本默认重建 CPU 运行时基线；如果在已启用 CUDA 的机器上再次运行完整配置，运行时会回到 CPU 选择。恢复 GPU 的步骤与显存不足时退回 CPU 的方法见 [使用手册](docs/使用手册.md)。

```powershell
conda run --no-capture-output -n VideoLens python -m pytest -q -m "not real_model and not cloud"
conda run --no-capture-output -n VideoLens python -m ruff check src tests scripts
conda run --no-capture-output -n VideoLens python scripts/download_test_audio.py
conda run --no-capture-output -n VideoLens python scripts/verify_real_models.py --audio data/fixtures/public-audio/funasr_zh.wav
```

源码、注释和自编写文档均采用 UTF-8，文档使用简体中文。[requirements-lock.txt](requirements-lock.txt) 保存本次 Windows / Python 3.13 / PyTorch CPU 验证环境的确切依赖版本，不含个人安装绝对路径。项目使用本地 Git 分阶段记录，不自动推送远程仓库。模型、私人视频、数据库、凭据和环境不会提交到 Git。

| 文档 | 用途 |
|---|---|
| [已批准计划](docs/Plan.md) | 项目范围、默认决策与验收目标 |
| [使用手册](docs/使用手册.md) | 从安装、模型配置到分析、对话、导出的完整操作 |
| [开发手册](docs/开发手册.md) | 环境、目录、扩展方式、Git 与测试 |
| [架构与代码导读](docs/架构与代码导读.md) | 一次分析和追问的代码路径、数据表与关键算法 |
| [API 文档](docs/API文档.md) | 鉴权、路由、输入、SSE 与错误格式 |
| [故障排查](docs/故障排查.md) | 安装、下载、FFmpeg、GPU、模型与网络问题 |
| [测试报告](docs/测试报告.md) | 本机实际测试、复现命令、失败和未执行项目 |
| [需求追踪](docs/需求追踪.md) | 计划要求与实现/验证证据对应 |
| [第三方说明](THIRD_PARTY_NOTICES.md) | VideoHub、WebSideChat、依赖、模型与测试素材来源 |
