# 知影 VideoLens

在自己的电脑上导入视频、生成有证据的 Markdown 报告，并围绕视频持续问答。网页已精简为**设置、输入、输出**，打开后直接使用，无需账号密码。

## 开始使用

1. 首次安装双击 **一键配置环境.cmd**，将 Miniconda、`VideoLens` Python 环境、依赖、模型和缓存部署到项目内部。
2. 日常双击 **一键启动.cmd**，等待检查完成，打开 `http://127.0.0.1:8000/app`。
3. 拖入本地视频，或输入视频 URL，点击“分析视频”。网页不再要求单独上传字幕。
4. 分析完成后下载 **`.md` 报告**；输入问题并发送，回答会直接显示在输出区。
5. 展开“设置”配置云端模型、管理本地 llama.cpp 模型，或修改系统提示词。

**完整步骤、云端 API Key 获取方法、模型推荐和本地模型下载/导入/删除操作，见 [详细使用教程](docs/使用手册.md)。**

## 模型怎么选

- 中文入门：百炼文本模型 `qwen-plus` 配合视觉模型 `qwen3-vl-plus`；先检查所在地域和业务空间的权限。
- 云端替代：OpenAI `gpt-6-luna`、DeepSeek `deepseek-flash`。这是依据官方接口文档的配置建议，尚未使用你的云端账号完成实测。
- 本机离线文本生成：默认 `Qwen3-4B-Q4_K_M.gguf`，通过 llama.cpp 运行。RTX 5060 8GB 可从 GPU 层数 99、上下文 8192 开始；显存不足时减小上下文或用 CPU。

模型接口、地址与官方来源见教程。云端总开关默认关闭。开启后会发送转录、问题和必要关键帧；本地 Qwen 文本模型不具备本地画面理解能力。

## 目录与数据

所有应用管理的文件保存在项目内：

| 目录 | 内容 |
|---|---|
| `runtime/miniconda3/` | 独立 Miniconda |
| `envs/VideoLens/` | Python 与软件包 |
| `cache/` | Conda、pip、临时文件、模型工具与独立浏览器缓存 |
| `models/` | 模型快照、独立 GGUF 仓库与校验清单 |
| `runtime/` | FFmpeg、llama.cpp 和受管进程记录 |
| `data/` | SQLite、Chroma、视频副本、关键帧、会话、导出和 DPAPI 密文凭据 |
| `logs/` | 应用与 llama.cpp 日志 |

一键启动使用独立 Edge/Chrome 资料目录，默认把报告下载到 `data/exports`。手动使用其他浏览器时，请自行将下载目录设到项目内。操作系统、显卡驱动和既有浏览器程序属于系统组件，不由项目搬迁。原始输入文件不会被自动删除或移动。

日志统一为：

```text
[2026-10-15 10:20:30][INFO][任务ID或线程名]: 日志内容
```

API Key 只作为项目内 Windows DPAPI 密文保存，不写入 Git、明文配置或页面回读。网页无需登录，内部 API/CLI 仍使用本机令牌。服务仅监听回环地址。

## 开发与验证

核心采用 Python、FastAPI、Gradio、SQLite 和 Chroma。媒体处理用 FFmpeg；转写用 faster-whisper；文本/图文向量用 BGE-M3 与 SigLIP2。保留任务检查点、证据校验、失败降级与 CLI/API 高级功能。

```powershell
# 项目根目录执行，明确使用项目内环境。
.\envs\VideoLens\python.exe -m videolens --help
.\envs\VideoLens\python.exe -m videolens doctor
.\envs\VideoLens\python.exe -m ruff check src tests scripts
```

双击 **一键测试.cmd** 执行自动化与环境检查。真实模型测试与模拟协议测试分别记录；最新结果和限制见 [测试报告](docs/测试报告.md)。代码与自编文档采用 UTF-8、简体中文注释，使用本地 Git 管理，不推送远程仓库。

| 文档 | 内容 |
|---|---|
| [使用教程](docs/使用手册.md) | 零基础操作、云端配置、模型推荐、本地模型管理 |
| [开发手册](docs/开发手册.md) | 环境、调试、关键数据流和扩展方法 |
| [架构与代码导读](docs/架构与代码导读.md) | 各层职责、分析与问答调用路径 |
| [API 文档](docs/API文档.md) | 鉴权、SSE、模型管理接口 |
| [测试报告](docs/测试报告.md) | 已验证、未验证、复现命令 |
| [故障排查](docs/故障排查.md) | 常见依赖、模型、网络与启动问题 |
| [需求追踪](docs/需求追踪.md) | 需求到代码及验证映射 |
| [原始批准计划](docs/Plan.md) | 首版历史范围与验收目标 |
| [第三方来源说明](THIRD_PARTY_NOTICES.md) | 上游参考与许可 |

参考 VideoHub 的任务恢复、证据和检索设计；参考 WebSideChat 的模型配置与流式交互，独立实现。许可原文保留在 `docs/licenses`。
