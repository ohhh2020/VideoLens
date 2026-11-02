# AstrBot 与 QQ 视频分析

从零安装请阅读[AstrBot 详细接入教程](AstrBot接入教程.md)，包含环境安装、NapCat、MCP、插件配置、日常使用与排错。本页保留功能概览和本机部署示例。

这是可选的本机接入功能：NapCat提供个人QQ的OneBot v11连接，AstrBot负责收消息和插件，MCP桥接调用已运行的VideoLens。视频分析与问答继续使用VideoLens原有的三个本地模型和两个云端配置，不为机器人重复加载模型，不需要给AstrBot另配一个文本模型。

## 使用方法

1. 群友从Bilibili分享一个视频，支持视频链接、b23.tv短链接和包含真实链接的QQ分享卡片。
2. **回复这条原视频消息**，发送 `视频分析`。可附带问题，例如 `视频分析 为什么作者推荐这款CPU？`，完成报告后自动回答。
3. 继续**回复同一条原视频消息**，直接写问题，例如 `两款CPU的游戏性能有什么差别？`，不必重复关键词。

普通群消息、没有引用的“视频分析”、回复报告的普通问题都不会调用VideoLens。原消息包含多个视频时会要求提供单一视频，不擅自选取。回复原视频时，已完成的分析结果可复用；追问按群、原视频消息和用户隔离上下文。不同群之间不会共用问答会话。

分析报告和追问回答默认采用**图片输出**：使用AstrBot自带的本地Markdown渲染器，展示标题、加粗、列表、表格和代码块；超长结果按图片分页，不丢弃尾部。图片消息引用原视频，追问仍需回复原视频。接收提示、等待提示和错误提示使用简短文字，不代表最终报告以文字输出。

渲染在本机完成，不调用外部转图服务。图片以真实图片内容进入OneBot的`image`消息段，不把Markdown、图片文件路径或JSON字符串当作普通文字发送。图片渲染失败时明确报错，不静默切换为文字。网页仍可查看和复制原始Markdown；如需文字输出，可显式将插件的`result_format`改为`text`。

### 更新插件使图片输出生效

VideoLens仓库中的插件源码与AstrBot实际加载的`data/plugins/`是两个目录。只修改仓库，不会自动更新已加载的插件。请在Git Bash中手动执行：

```bash
cd /d/Project/vl/VideoLens
conda activate D:/Environment/AstrBot
# 只核对源码版本和输出模式，不修改安装文件、不连接服务
python integrations/install_astrbot_plugin.py --astrbot-root D:/Project/AstrBot --check
# 备份旧文件并同步源码；开启image模式，清空临时自测前缀
python integrations/install_astrbot_plugin.py --astrbot-root D:/Project/AstrBot
```

随后在AstrBot面板中手动重载“知影视频分析”插件。若AstrBot尚未运行，按下面的日常启动步骤启动即可。安装脚本不会控制窗口、发送消息、启动或停止程序，也不会调用重载接口；保留允许群、MCP名称等其他设置。旧插件文件与旧插件配置备份在AstrBot的`data/plugin_data/astrbot_plugin_videolens/source_backups/`下。

### Bilibili分享卡片与结果图片的区别

视频分享卡片用于打开Bilibili视频，分析结果图片用于阅读报告。粘贴裸链接不会由本插件自动转换为分享卡片；群友可使用Bilibili的“分享到QQ”发送原生卡片。两种来源都能作为引用分析对象。

需要由机器人主动分享视频时，可手动执行以下工具。它先读取视频标题和封面，再调用NapCat的`get_mini_app_ark`生成完整小程序数据，最后发送`json`消息段；不使用手工拼接的简化卡片，也不在失败时改发裸链接。

```bash
python integrations/send_bilibili_card.py \
  --credentials ../VideoLens-runtime/config/qqbot-local.json \
  --group 你明确允许发送的群号 \
  --url https://www.bilibili.com/video/BV1rkxi66Ej2
```

该命令会真正发送群消息，仅在你希望主动分享时运行；图片输出不需要执行它。

## 本机部署位置

本次部署采用独立Conda环境：

| 内容 | 本机路径或入口 |
| --- | --- |
| AstrBot v4.29.4 源码和面板 | `D:/Project/AstrBot`；`http://127.0.0.1:6185` |
| AstrBot Python 3.12 环境 | `D:/Environment/AstrBot` |
| NapCat v4.18.42 Windows Node包 | `D:/Project/NapCatQQ`；面板 `http://127.0.0.1:6099` |
| VideoLens Python 3.13环境 | `D:/Environment/VideoLens` |
| VideoLens网页和模型服务 | `http://127.0.0.1:8000/ui` |
| OneBot反向WebSocket | `ws://127.0.0.1:6199/ws` |

三个主程序在可见Git Bash窗口运行，按Ctrl+C关闭对应程序。MCP是AstrBot管理的stdio子进程，关闭AstrBot时一起退出。

## 日常启动

在三个Git Bash窗口分别执行：

```bash
# VideoLens
conda activate D:/Environment/VideoLens
cd /d/Project/vl/VideoLens
bash scripts/start_web.sh --port 8000
```

```bash
# AstrBot：本机已生成启动脚本，会激活独立Conda环境
cd /d/Project/AstrBot
bash start-local.sh
```

```bash
# NapCat：使用发行包内自带Node，不影响当前桌面QQ
cd /d/Project/NapCatQQ
bash start-local.sh
```

QQ已设置优先快速登录；会话失效或要求验证码/设备验证时，通过NapCat本机面板完成验证。QQ密码不保存在项目源码或部署配置里。

## 配置与数据

- AstrBot的MCP配置：`D:/Project/AstrBot/data/mcp_server.json`。模板见 [mcp-server.example.json](../../integrations/mcp-server.example.json)，其中路径按实际安装位置修改。
- 插件源码：`integrations/astrbot_plugin_videolens/`；实际安装在AstrBot的 `data/plugins/astrbot_plugin_videolens/`。
- 插件设置：AstrBot面板中“知影视频分析”插件，配置允许的群号、关键词、等待上限、结果格式和文字模式分条长度。本次仅启用指定测试群，群号保存在本机配置里。
- 本机私密配置：`VideoLens-runtime/config/qqbot-local.json`，保存面板管理密码及连接凭据，**不得上传或分享**；不包含QQ密码。
- MCP会话与任务映射：`VideoLens-runtime/data/mcp/qqbot.sqlite3`；群消息幂等和视频消息关联：AstrBot的 `data/plugin_data/astrbot_plugin_videolens/messages.sqlite3`。
- 报告、转写、模型和检索索引仍在原VideoLens运行目录，见[工作流程与数据存放](../技术分析/工作流程与数据存放.md)。

`self_test_prefix` 仅供使用机器人账号验收时临时启用；日常留空，NapCat的 `reportSelfMessage` 关闭，防止机器人处理自己的输出。

实现边界见[本地MCP接入](../技术分析/本地MCP接入.md)，真实测试记录见[2026-11-02接入验收](../更新日志/2026-11-02AstrBot接入验收.md)。
