"""统一日志格式；任务 ID 优先，没有任务时记录线程名。"""
from __future__ import annotations

import logging
import re
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

task_context: ContextVar[str] = ContextVar("task_context", default="")


class ProjectFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        label = getattr(record, "task_id", "") or task_context.get() or record.threadName
        message = record.getMessage()
        message = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", message)
        message = re.sub(r"(?i)(Bearer\s+|api[_-]?key[=: ]+|token[=: ]+)[^\s,;]+", r"\1[已隐藏]", message)
        message = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[已隐藏]", message)
        # 多行异常也必须每行符合格式；只保留异常类型，避免第三方异常回显凭据。
        if record.exc_info:
            message += "；异常类型：" + record.exc_info[0].__name__
        prefix = f"[{stamp}][{record.levelname}][{label}]: "
        return "\n".join(prefix + line for line in message.splitlines())


def configure_logging(root: Path) -> None:
    logger = logging.getLogger()
    if any(getattr(h, "_videolens", False) for h in logger.handlers):
        return
    (root / "logs").mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), RotatingFileHandler(root / "logs/videolens.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")]
    for handler in handlers:
        handler._videolens = True
        handler.setFormatter(ProjectFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logging.captureWarnings(True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        target = logging.getLogger(name)
        target.handlers.clear()
        target.propagate = True
    for name in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
