"""知影客户端层：CLI、Gradio Web、Obsidian 插件。

三个客户端都只跟"一个人的两个入口"打交道：

- **本地直连**：``Orchestrator``（不需要起服务，CLI 默认走这条）
- **HTTP**：``/api/*``（Obsidian 插件、多端共享、以及想跑成服务时）

分层约定（见 ``core/__init__.py``）：``clients`` 属于最外层，只依赖公开接口，
不碰 ``core`` 内部私有属性 —— 这样换掉 Web 框架或加一个新端，改动不外溢。

====================================  ==========================================
``clients.cli``                      命令行：analyze / ask / list / show / doctor …
``clients.gradio_ui``                浏览器 UI（上传、时间线、关键帧画廊、问答）
``clients.obsidian_plugin``          Obsidian 插件（HTTP 调本地服务）
====================================  ==========================================
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
