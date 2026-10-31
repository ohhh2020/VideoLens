"""统一控制台/文件日志，以及在线程池之间传递的任务上下文。"""

from __future__ import annotations

import contextvars
import copy
import logging
import os
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from backend.runtime import runtime_path
from backend.terminal import CONSOLE_COLOR_ENV, supports_ansi, use_color

_TASK = contextvars.ContextVar("videolens_log_task", default="")
MANAGED_LOG_ENV = "VIDEOLENS_MANAGED_LOG"
ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|[0-9@-Z\\-_])")


def open_session_log(directory, *, date=None):
    """每次启动独占创建新文件；并发启动也不会覆盖或复用旧日志。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    day = date or datetime.now().strftime("%Y-%m-%d")
    pattern = re.compile(rf"^{re.escape(day)}-(\d+)\.log$")
    count = max((int(match[1]) for path in directory.iterdir()
                 if (match := pattern.fullmatch(path.name))), default=0) + 1
    while True:
        try:
            return (directory / f"{day}-{count}.log").open("x", encoding="utf-8")
        except FileExistsError:
            count += 1


@contextmanager
def log_task(task):
    token = _TASK.set(str(task))
    try:
        yield
    finally:
        _TASK.reset(token)


class TaskFormatter(logging.Formatter):
    """控制台以白色为底，仅高亮级别与关键状态；文件保持纯文本。"""

    WHITE = "\033[37m"
    GREEN = "\033[32m"
    ORANGE = "\033[38;5;208m"
    RED = "\033[31m"
    COLORS = {
        logging.INFO: GREEN,
        logging.WARNING: ORANGE,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }
    # 长词优先，避免把“未完成/未通过”等否定状态的一部分误标为成功。
    KEYWORDS = re.compile(
        r"(?P<error>未成功|未通过|失败|错误|异常|超时|不可用|缺失|缺少|不完整)"
        r"|(?P<warning>未完成|未就绪|未设置|未启用|未加载|未安装|警告|重试|降级|跳过|取消|排队|等待)"
        r"|(?P<success>完成|成功|就绪|通过|已保存|已加载|已启用|已连接|已启动)"
    )

    def __init__(self, fmt, datefmt=None, *, color=False):
        super().__init__(fmt, datefmt)
        self.color = color

    def format(self, record):
        # 一个记录会依次写入终端和文件，颜色和异常缓存不能污染原记录。
        record = copy.copy(record)
        record.task = record.threadName or _TASK.get() or "-"
        if self.color:
            colors = {"error": self.RED, "warning": self.ORANGE, "success": self.GREEN}
            record.msg = self.KEYWORDS.sub(
                lambda match: f"{colors[match.lastgroup]}{match[0]}{self.WHITE}",
                record.getMessage(),
            )
            record.args = ()
            level_color = self.COLORS.get(record.levelno)
            if level_color:
                record.levelname = f"{level_color}{record.levelname}{self.WHITE}"
            return f"{self.WHITE}{super().format(record)}\033[0m"
        return super().format(record)


class ProgressAccessFilter(logging.Filter):
    """省略成功的高频页面状态读取，错误请求和其他访问日志照常保留。"""

    PATH = re.compile(r"^/internal/(?:tasks/[A-Za-z0-9_-]+(?:/events)?|memory)$")

    def filter(self, record):
        args = record.args
        if record.name == "uvicorn.access" and isinstance(args, tuple) and len(args) == 5:
            _, method, path, _, status = args
            if method == "GET" and status == 200 and (
                self.PATH.fullmatch(str(path).split("?", 1)[0])
                or str(path) == "/internal/health?service_control=1"
            ):
                return False
        return True


def configure_logging(cfg):
    root = logging.getLogger()
    if getattr(root, "_zhiying_configured", False):
        return
    root.setLevel(cfg.logging.level)
    import sys
    stream = sys.stderr
    managed = os.environ.get(MANAGED_LOG_ENV) == '1'
    ansi = (os.environ[CONSOLE_COLOR_ENV] == '1') if managed and CONSOLE_COLOR_ENV in os.environ else supports_ansi(stream)
    color = use_color(cfg.logging.color, ansi)
    console = logging.StreamHandler(stream)
    console.setFormatter(
        TaskFormatter(cfg.logging.format, cfg.logging.date_format, color=color)
    )
    root.addHandler(console)
    if os.environ.get(MANAGED_LOG_ENV) != "1":
        # 常规网页启动由前台控制台统一落盘，独立调用本函数时仍有文件日志。
        file = logging.StreamHandler(open_session_log(runtime_path(cfg.logging.directory, cfg.root_dir)))
        file.setFormatter(TaskFormatter(cfg.logging.format, cfg.logging.date_format))
        root.addHandler(file)
    root._zhiying_configured = True
    logging.getLogger("uvicorn.access").addFilter(ProgressAccessFilter())
    for name in ("httpx", "chromadb", "urllib3", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
