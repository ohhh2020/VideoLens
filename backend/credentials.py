"""密钥输入检查；不回显或记录密钥内容。"""

from __future__ import annotations


def is_masked_secret(value: str | None) -> bool:
    """识别显示掩码，避免把打码值当作真实凭据保存和发送。"""
    text = str(value or "")
    return any(mark in text for mark in ("*", "•", "●", "…", "..."))


def secret_input_error(value: str | None) -> str:
    if is_masked_secret(value):
        return "保存的 API Key 是显示掩码，请在设置页重新填写完整密钥"
    if any(char in str(value or "") for char in ("\r", "\n")):
        return "API Key 含换行，请重新粘贴完整密钥"
    return ""
