# VideoLens 视频链接智能总结系统

## 1. 项目简介

VideoLens 是一个面向 Python 程序设计期末大作业的 AI 视频理解应用。用户只需要粘贴一个视频链接，系统就会自动完成：

1. 下载或读取视频；
2. 提取音频并转写为逐句文本；
3. 按场景和固定时间间隔抽取关键帧；
4. 调用多模态大模型理解关键帧内容；
5. 融合“语音转写 + 画面描述”，生成结构化视频总结、章节时间线和关键词；
6. 在 Web 界面中展示结果，并支持导出 JSON/Markdown。

项目突出“链接进、总结出”的完整闭环，适合现场演示。

## 2. 实际应用价值

- **学生/学习者**：快速浏览长课程、讲座、培训视频，节省时间。
- **内容运营/自媒体**：对 B 站、YouTube 等平台视频做内容速览、选题复盘。
- **知识管理**：把视频转成可搜索、可归档的文字摘要和时间线。
- **办公场景**：会议录屏、产品讲解视频的会后纪要。
- **无障碍辅助**：为听力障碍用户提供视频文字稿和画面描述。

## 3. 核心功能

- 支持视频链接：Bilibili、YouTube、抖音等由 `yt-dlp` 支持的站点，以及直链 MP4 和本地视频文件。
- 自动下载：自动选择 MP4 格式，支持 cookie 文件。
- 语音转写：使用 `faster-whisper` 本地转写，不依赖云端语音接口。
- 关键帧抽取：均匀采样 + 场景变化检测，控制视觉 token 预算。
- 多模态理解：通过 OpenAI-compatible API 调用 VLM，生成画面描述。
- 智能总结：融合音频文本与视觉描述，输出一句话总结、核心要点、时间线、关键词和标签。
- Web 演示：Gradio 界面输入链接，一键生成结果。
- 可导出：自动保存原始转写、关键帧、画面描述和总结报告。

## 4. 技术栈

| 模块 | 技术 |
|------|------|
| 视频下载 | yt-dlp |
| 音频转写 | faster-whisper |
| 关键帧抽取 | OpenCV |
| 多模态理解 | OpenAI-compatible API / 任意 VLM |
| 文本总结 | OpenAI-compatible API / 规则回退 |
| Web 界面 | Gradio |
| 数据处理 | NumPy、jieba |

## 5. 快速开始

### 5.1 环境准备

- Python 3.10+（建议 3.11）
- FFmpeg，需加入系统 PATH

### 5.2 安装依赖

```bash
cd video_summary_app
pip install -r requirements.txt
```

### 5.3 配置

复制 `.env.example` 为 `.env`，填写 API 配置：

```bash
cp .env.example .env
```

本地语音转写默认开启，无需 API Key。若想使用视觉理解与 LLM 总结，需要填写：

```text
OPENAI_API_KEY=你的KEY
OPENAI_BASE_URL=https://api.openai.com/v1
VLM_MODEL=gpt-4o-mini
LLM_MODEL=gpt-4o-mini
```

兼容任意 OpenAI 接口协议的服务，例如 Qwen-VL、GLM-4V、DeepSeek-VL 等。

### 5.4 启动 Web 界面

```bash
python app.py
```

浏览器打开 Gradio 地址，输入视频链接，点击“开始分析”。

### 5.5 命令行运行

```bash
python cli.py "https://www.bilibili.com/video/BVxxxx"
```

结果默认输出到 `data/output/<job_id>/`。

## 6. 现场演示建议

1. 提前准备一个 3–8 分钟的短视频链接，避免下载和转写过久。
2. 演示前先跑通一次，确认目标站点可下载。
3. 现场先展示“输入链接”，再展示生成的总结、时间线、关键帧。
4. 如果现场网络不稳，可准备一个本地 MP4 作为备份演示。

## 7. 项目结构

```text
video_summary_app/
├── app.py                     # Gradio 界面
├── cli.py                     # 命令行入口
├── requirements.txt
├── .env.example
├── README.md
├── docs/
│   ├── 01_需求分析文档.md
│   ├── 02_项目分析与设计文档.md
│   └── 03_现场演示脚本.md
└── src/
    ├── config.py
    ├── models.py
    ├── downloader.py
    ├── media.py
    ├── transcriber.py
    ├── keyframes.py
    ├── vlm.py
    ├── summarizer.py
    ├── storage.py
    └── pipeline.py
```

## 8. 合规说明

本系统仅用于学习和技术演示。下载视频前请确认内容版权和平台服务条款，不得用于商业分发或侵犯权利人权益。

