"""只根据真实输出句柄启用终端控制，不把继承的 Bash 环境变量当作能力证明。"""
from __future__ import annotations

import ctypes
import os
import re
import sys

CONSOLE_COLOR_ENV = 'VIDEOLENS_CONSOLE_COLOR'
_MSYS_PTY = re.compile(r'\\(?:msys|cygwin)-[a-z0-9]+-pty\d+-(?:to|from)-master$', re.I)


def _windows_console_vt(stream):
    """在当前输出句柄启用并核验 VT；不修改其他终端或系统设置。"""
    import msvcrt
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetConsoleMode.restype = wintypes.BOOL
    kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.SetConsoleMode.restype = wintypes.BOOL
    handle = msvcrt.get_osfhandle(stream.fileno())
    mode = wintypes.DWORD()
    if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
        return False
    # VT 输出同时需要 ENABLE_PROCESSED_OUTPUT，保留所有其他模式位。
    if mode.value & 5 != 5 and not kernel.SetConsoleMode(handle, mode.value | 5):
        return False
    return bool(kernel.GetConsoleMode(handle, ctypes.byref(mode)) and mode.value & 5 == 5)


def _windows_pty_name(stream):
    """读取 mintty/MSYS 真正的命名管道名；普通重定向管道不会被误判。"""
    import msvcrt
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
    buffer = ctypes.create_string_buffer(65536)
    if not kernel.GetFileInformationByHandleEx(msvcrt.get_osfhandle(stream.fileno()), 2, buffer, len(buffer)):
        return ''
    length = ctypes.c_uint32.from_buffer(buffer).value
    if length > len(buffer)-4 or length % 2:
        return ''
    return buffer.raw[4:4+length].decode('utf-16-le')


def is_msys_pty(stream):
    if sys.platform != 'win32':
        return False
    try:
        return bool(_MSYS_PTY.search(_windows_pty_name(stream)))
    except (OSError, ValueError, AttributeError):
        return False


def supports_ansi(stream):
    if os.environ.get('TERM') == 'dumb':
        return False
    if sys.platform == 'win32':
        try:
            if _windows_console_vt(stream):
                return True
        except (OSError, ValueError, AttributeError):
            pass
        return is_msys_pty(stream)
    try:
        return bool(stream.isatty())
    except (OSError, ValueError, AttributeError):
        return False


def interactive_input(stream):
    try:
        if stream.isatty():
            return True
    except (OSError, ValueError, AttributeError):
        pass
    return is_msys_pty(stream)


def use_color(configured, ansi):
    return bool(ansi and configured != 'never' and (configured == 'always' or 'NO_COLOR' not in os.environ))
