"""密钥只保存为项目内 DPAPI 密文，不向 Windows 凭据库写入项目数据。"""
from __future__ import annotations

import base64
import ctypes
import hashlib
import os
from pathlib import Path

from .errors import VideoLensError


def protect(value: bytes, decrypt: bool = False) -> bytes:
    """Windows 当前用户加密；密文搬到别的账户后需要重新填写 API Key。"""
    if os.name != "nt":
        raise RuntimeError("此凭据存储需要 Windows DPAPI；其他系统请使用环境变量")

    class Blob(ctypes.Structure):
        _fields_ = [("size", ctypes.c_ulong), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    buffer = ctypes.create_string_buffer(value)
    incoming = Blob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    outgoing = Blob()
    function = ctypes.windll.crypt32.CryptUnprotectData if decrypt else ctypes.windll.crypt32.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(Blob)]
    function.restype = ctypes.c_int
    if not function(ctypes.byref(incoming), None, None, None, None, 1, ctypes.byref(outgoing)):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(outgoing.data, outgoing.size)
    finally:
        ctypes.windll.kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        ctypes.windll.kernel32.LocalFree(outgoing.data)


class SecretStore:
    def __init__(self, directory: Path | None = None):
        from .paths import configure_paths
        self.directory = directory or configure_paths() / "data/secrets"

    def _path(self, ref: str) -> Path:
        return self.directory / (hashlib.sha256(ref.encode("utf-8")).hexdigest() + ".dpapi")

    def get(self, ref: str) -> str | None:
        if not ref:
            return None
        name = "VIDEOLENS_KEY_" + "".join(c if c.isalnum() else "_" for c in ref).upper()
        if os.environ.get(name):
            return os.environ[name]
        path = self._path(ref)
        if not path.exists():
            return None
        try:
            return protect(base64.b64decode(path.read_bytes()), decrypt=True).decode("utf-8")
        except Exception as exc:
            raise VideoLensError("KEYSTORE_UNAVAILABLE", "无法解密项目凭据，请重新保存 API Key", 503) from exc

    def set(self, ref: str, value: str) -> None:
        if not ref or len(ref) > 100 or not value:
            raise VideoLensError("INVALID_KEY", "凭据名称与密钥不能为空")
        try:
            from filelock import FileLock
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self._path(ref)
            with FileLock(str(path) + ".lock"):
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(base64.b64encode(protect(value.encode("utf-8"))))
                temporary.replace(path)
        except Exception as exc:
            raise VideoLensError("KEYSTORE_UNAVAILABLE", "项目凭据保存失败，请检查目录权限", 503) from exc
