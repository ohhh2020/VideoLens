# 知影 · Obsidian 插件

把本地运行的知影服务接进 Obsidian：分析视频、把**带 `clip_id` 与毫秒时间戳**的问答
写进笔记。视频、转录、向量都不离开本机 —— 插件只与 `127.0.0.1` 通信。

## 先决条件

1. 知影服务在跑（二选一）：

   ```bash
   python -m clients.cli serve          # 默认 127.0.0.1:8000
   uvicorn api.main:app --port 8000
   ```

2. Obsidian **桌面版**（移动端不支持自定义插件；本插件 `isDesktopOnly: true`）。

## 安装

```text
<你的仓库>/.obsidian/plugins/zhiying-video/
├── manifest.json
├── main.js
└── styles.css
```

把本目录三个文件拷进上面的路径，重启 Obsidian → `设置 → 第三方插件` → 关闭安全模式 →
启用「知影 · 视频智能分析」。

> 本插件没有构建步骤：`main.js` 是手写的 CommonJS，不需要 `npm install` / `esbuild`。
> 想改逻辑直接编辑这三个文件，重启 Obsidian 生效。

## 设置

| 项 | 默认 | 说明 |
| --- | --- | --- |
| 服务地址 | `http://127.0.0.1:8000` | 知影服务根地址 |
| API Token | 空 | 服务端配了 `api.token` 时必填 |
| 轮询间隔 | `1200` ms | 分析任务进度刷新频率 |
| 分析超时 | `45` 分钟 | 1 小时视频在 CPU 上可能要 30 分钟 |
| 分析完成后自动插入笔记 | 开 | 写入摘要 + 时间线 + `zhiying_video_id` |
| 插入标题 | `## 知影问答` | 问答插入时的二级标题 |

## 命令

| 命令 | 作用 |
| --- | --- |
| `知影：分析视频（本地路径或 URL）` | 弹窗填路径/链接 → 提交任务 → 轮询进度 → 完成插入笔记 |
| `知影：对当前笔记关联的视频提问` | 读当前笔记 frontmatter 的 `zhiying_video_id` 提问 |
| `知影：浏览视频库` | 列出库内视频，点选后提问 |
| `知影：把视频报告导入为新笔记` | 填 `video_id` → 生成完整时间线笔记并打开 |
| `知影：检查服务连通性` | 打一次 `/api/health` 与 `/api/stats` |

## 笔记约定

插入的视频笔记带 frontmatter，后续提问直接读它：

```yaml
---
zhiying_video_id: v1a2b3c4d5e6f
zhiying_status: ready
zhiying_duration_ms: 2403000
---
```

问答插入成 callout，引用是表格（时间区间为 `HH:MM:SS.mmm`，可直接对回视频）：

```markdown
> [!quote] 知影问答
> **问**：切片长度是怎么定的？
>
> 根据 [1]，切片默认在 10–30 秒之间……

**引用证据**

| # | 时间区间 | clip_id | 原文 |
| --- | --- | --- | --- |
| 1 | 00:00:20.100 – 00:00:42.300 | `v1a2b3c4d5e6f-c0001` | 二十秒左右…… |
```

## 故障排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 状态栏「知影：未连接」 | 服务没起 / 端口不对 | `python -m clients.cli serve`，再「测试连接」 |
| `HTTP 401` | 服务端开了鉴权 | 设置里填 API Token（同 `ZHIYING_API_TOKEN`） |
| `HTTP 422` | 路径不存在或问题超 500 字 | 检查输入；看服务端日志 |
| 分析一直 0% | 模型权重没下载，卡在下模型 | `python scripts/download_models.py --all` |
| 笔记里没有 `zhiying_video_id` | 手动建的笔记 | 用「浏览视频库」选一个，或「导入视频报告」 |

## 数据边界

- 插件只发 `POST /api/videos/analyze`、`POST /api/videos/{id}/ask`、`GET /api/*` 三类请求。
- 不上传笔记正文，不读取仓库里与视频无关的文件。
- 服务端默认只监听回环地址；`api.host` 改成 `0.0.0.0` 前请先配 `api.token`。
