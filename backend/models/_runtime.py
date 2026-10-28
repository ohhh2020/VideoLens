"""推理库仅从当前 Miniconda 环境导入，CUDA DLL 也仅在此环境查找。"""

from __future__ import annotations
import importlib.util
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

_DLL_HANDLES = []
_DLL_READY = False
_GPU_LOCK = threading.RLock()


@contextmanager
def gpu_slot(device):
    """同一显卡上的本地推理有界，CPU 模型可以继续独立执行。"""
    if str(device).startswith("cuda"):
        with _GPU_LOCK:
            yield
    else:
        yield


def ensure_torch_path() -> None:
    spec = importlib.util.find_spec("torch")
    if (
        spec is not None
        and spec.origin
        and not Path(spec.origin).resolve().is_relative_to(Path(sys.prefix).resolve())
    ):
        raise RuntimeError("torch 必须安装在当前 Miniconda 环境，禁止导入外部运行库")


def ensure_cuda_dlls() -> None:
    global _DLL_READY
    if _DLL_READY or sys.platform != "win32":
        return
    _DLL_READY = True
    prefix = Path(sys.prefix)
    candidates = [prefix / "Library/bin"]
    candidates.append(prefix / "Lib/site-packages/torch/lib")
    candidates += list((prefix / "Lib/site-packages/nvidia").glob("*/bin"))
    for directory in candidates:
        if directory.is_dir():
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
