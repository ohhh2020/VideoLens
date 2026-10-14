"""独立日志转发进程：部署命令退出后，llama 日志仍保持统一格式。"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from .config import atomic_json
from .logging_setup import ProjectFormatter


def main():
    descriptor = Path(sys.argv[1])
    info = json.loads(descriptor.read_text(encoding="utf-8"))
    log = logging.getLogger("llama.cpp")
    log.setLevel(logging.INFO)
    handler = logging.FileHandler(info["log"], encoding="utf-8")
    handler.setFormatter(ProjectFormatter())
    log.addHandler(handler)
    log.propagate = False
    try:
        process = subprocess.Popen(info["command"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        atomic_json(descriptor.with_suffix(".ready"), {"pid": process.pid})
        for line in process.stdout:
            text = line.rstrip()
            if text:
                level = (logging.ERROR if re.search(r"\bE\s|\berror\b", text, re.I) else
                         logging.WARNING if re.search(r"\bW\s|\bwarning\b", text, re.I) else logging.INFO)
                log.log(level, text, extra={"task_id": "llama.cpp"})
        process.wait()
    finally:
        descriptor.unlink(missing_ok=True)
        descriptor.with_suffix(".ready").unlink(missing_ok=True)
        handler.close()


if __name__ == "__main__":
    main()
