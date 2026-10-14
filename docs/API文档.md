# API 文档

服务默认地址 `http://127.0.0.1:8000`，业务接口前缀 `/api/v1`。除 `/health` 外，API 需要请求头 `Authorization: Bearer <本地访问令牌>`。令牌来自 `data/.access-token` 或 `VIDEOLENS_API_TOKEN` 环境变量。网页通过自己的登录会话访问受保护媒体。

## 1. 获取最新接口定义

项目以 FastAPI 输入模型生成 OpenAPI 与 JSON Schema。下面的代码从已启动服务读取当前定义；实际字段与约束以运行中的接口定义为准。

```powershell
$token = (Get-Content -Raw -Encoding UTF8 data/.access-token).Trim()
$headers = @{ Authorization = "Bearer $token" }
Invoke-RestMethod 'http://127.0.0.1:8000/api/v1/openapi.json' -Headers $headers
```

不要把 `$token` 打印到公开日志或提交到 Git。以下所有示例都使用已设置的 `$headers`。JSON 文本使用 UTF-8；内部时间是整数毫秒，公开证据同时提供格式化时间。接口响应会去除内部文件路径。

## 2. 接口索引

| 方法与路径（省略 `/api/v1`） | 输入 | 用途 |
|---|---|---|
| `GET /videos` | 无 | 获取视频列表 |
| `POST /videos/import` | JSON `path`，可选 `subtitle` | 导入本机文件；该路由仅接受回环来源 |
| `POST /videos/upload` | multipart `file`，可选 `subtitle` | 流式上传视频和字幕 |
| `POST /videos/url` | JSON `url` | 创建在线视频来源 |
| `GET /videos/{video_id}` | 视频 ID | 详情、报告与证据 |
| `DELETE /videos/{video_id}` | 视频 ID | 删除视频受管数据 |
| `GET /videos/{video_id}/media` | 视频 ID | 播放受管视频 |
| `GET /videos/{video_id}/frames/{frame_id}` | 视频 ID、帧 ID | 读取属于此视频的关键帧 |
| `GET /videos/{video_id}/export?format=markdown` | `markdown` 或 `json` | 下载报告 |
| `POST /videos/{video_id}/tasks` | `{"kind":"analyze"}` 或 `{"kind":"reindex"}` | 返回 202 和后台任务 |
| `GET /tasks` | 无 | 任务列表 |
| `GET /tasks/{task_id}` | 任务 ID | 当前状态、阶段、进度及原因 |
| `POST /tasks/{task_id}/{action}` | `pause`、`resume`、`cancel` | 请求任务控制 |
| `GET /tasks/{task_id}/events` | 可选 `after` 或 `Last-Event-ID` | SSE 任务事件 |
| `POST /videos/{video_id}/search` | `{"query":"问题"}` | 文本混合检索 |
| `POST /videos/{video_id}/search-image` | multipart `image`，可选 `query` | 图片相似检索；图片上限 20MiB |
| `POST /videos/{video_id}/sessions` | 无 | 创建属于该视频的会话 |
| `GET /videos/{video_id}/sessions` | 视频 ID | 会话列表 |
| `GET /sessions/{session_id}/messages` | 会话 ID | 历史消息 |
| `POST /sessions/{session_id}/ask` | `question`、`stream` | 普通 JSON 或 SSE 回答 |
| `GET /settings` | 无 | 公开设置，无 API Key 和访问令牌 |
| `PATCH /settings` | 要修改的设置字段 | 校验并保存设置 |
| `PUT /keys/{reference}` | `{"key":"实际密钥"}` | 写系统凭据库；没有读密钥接口 |
| `POST /providers/{kind}/check` | `text` 或 `vision` | 检查对应供应商连接 |
| `GET /doctor` | 无 | 运行环境、模型与存储诊断 |
| `POST /cleanup` | 无 | 按已配置 TTL 清理受管数据 |

输入模型拒绝未知字段。URL 最长 2048 字符；检索文本最大 10000 字符，问答最大 20000 字符；具体范围随公开 Schema 一起校验。

## 3. 一次完整 API 调用

先导入，再提交任务。导入完成并不代表分析已完成。

```powershell
$body = @{ path = 'D:\视频\课程.mp4'; subtitle = 'D:\视频\课程.srt' } | ConvertTo-Json
$video = Invoke-RestMethod -Method Post 'http://127.0.0.1:8000/api/v1/videos/import' `
  -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
$job = Invoke-RestMethod -Method Post "http://127.0.0.1:8000/api/v1/videos/$($video.id)/tasks" `
  -Headers $headers -ContentType 'application/json' -Body '{"kind":"analyze"}'
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/tasks/$($job.id)" -Headers $headers
```

任务完成后创建会话、提问并导出：

```powershell
$session = Invoke-RestMethod -Method Post "http://127.0.0.1:8000/api/v1/videos/$($video.id)/sessions" -Headers $headers
$question = @{ question = '视频的核心结论是什么？'; stream = $false } | ConvertTo-Json
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/api/v1/sessions/$($session.id)/ask" `
  -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($question))
Invoke-WebRequest "http://127.0.0.1:8000/api/v1/videos/$($video.id)/export?format=markdown" `
  -Headers $headers -OutFile '视频报告.md'
```

## 4. SSE 与断线处理

任务 SSE 的每条持久化事件都有整数编号。客户端记录最近收到的 `id`，断线后通过 `Last-Event-ID` 请求头或 `after` 查询参数继续获取。

```text
id: 12
event: progress
data: {"stage":"frames","progress":0.5}

```

上面只是格式示意，实际 `data` 字段由任务事件决定。空行是事件结束标记；收到一块网络数据不一定收到完整事件。客户端应逐行累计 `data:`，遇到空行再解析 JSON。

任务进入终态后发送 `done`。暂停也会结束当前监听，因此用户恢复任务后可以重新订阅。连接保持事件以冒号开头，无需作为业务数据解析。

问答使用 `POST /sessions/{id}/ask` 且 `stream=true`。先确认会话存在，再开始 SSE；流启动后的异常通过 `error` 事件传递，因为 HTTP 状态码已经发出。只有最终完成事件才能确认回答完整，错误事件和连接突然断开都不能当作成功。

## 5. 错误、权限与状态

错误使用统一 JSON 包装，包含可识别错误码与中文说明。常见 HTTP 状态包括：401 未提供有效令牌、404 不存在的视频/会话、409 状态冲突、413 上传过大、422 输入校验失败、429 访问频率超限、503 模型或服务不可用。具体业务错误码以响应为准。

不存在“读取云端密钥”接口。公开设置只保留 Provider 的 `key_ref`；将它交给后端凭据库取得实际密钥。视频路径导入仅用于本机接口；HTTP 响应中的 ID 不能被当作任意磁盘路径使用。

API 的分析任务状态和用户请求状态不同：收到 `pause` 请求后可能先进入暂停请求状态，直到安全检查点才变为已暂停。取消、失败、部分成功需要按实际 `status` 判断，不能仅因 HTTP 200 就认为视频分析成功。
