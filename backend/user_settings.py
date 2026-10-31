"""知影 ZhiYing —— 界面可改的运行时设置（``data/user_settings.json``）。

为什么单独一层，而不是直接改 ``configs/app_config.yaml``
--------------------------------------------------------
1. **密钥不该进版本控制**：``data/`` 已在 ``.gitignore`` 里，而 yaml 是要提交的。
2. **改 yaml 会丢注释**：任何"读回来再写回去"的实现都会把用户手写的解释性注释
   冲掉，而那个文件的价值一半在注释里。
3. **优先级要可预期**：内置默认 < ``app_config.yaml`` < 本文件 < 环境变量 <
   显式 ``overrides``。运维能用环境变量临时顶掉界面设置，界面设置又能压过仓库
   里的默认值 —— 三层各管各的，谁都不需要知道对方。

白名单
------
只有 :data:`SETTABLE_KEYS` 里的键允许通过网络请求写入。理由：这个模块会被
网页设置接口直接调用，回调参数来自浏览器；不做白名单就等于把整个 ``AppConfig``
的任意字段（包括 ``paths.sqlite``）暴露成远程可写。

值的类型在写入前按白名单里声明的类型强制转换，转不动就整批拒绝 —— 半批写入是
最难排查的状态：配置看起来改了，实际只改了一半。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from backend.config import AppConfig, get_app_config, user_settings_path
from backend.credentials import secret_input_error

logger = logging.getLogger("zhiying.user_settings")

__all__ = [
    "BackendStatus",
    "SETTABLE_KEYS",
    "SettingsError",
    "current_settings",
    "describe_backends",
    "format_backends",
    "save_settings",
    "preview_settings",
    "settings_file",
    "validate_settings",
]

#: JSON 文件里的键 → 类型。值为 ``None`` 表示"字符串，允许空"。
#: 键名用 ``a.b.c`` 点号路径，与 :func:`backend.config.load_app_config` 的
#: ``overrides`` 参数完全一致，所以保存后重建配置不需要任何翻译层。
SETTABLE_KEYS: Final[dict[str, type]] = {
    # ---- 大模型（LLM）----
    "llm.mode": str,  # auto / cloud / local
    "llm.cloud.base_url": str,
    "llm.cloud.model": str,
    "llm.cloud.api_key": str,
    "llm.cloud.api_key_env": str,
    "llm.cloud.timeout_s": int,
    "llm.cloud.temperature": float,
    "llm.cloud.max_tokens": int,
    # ---- 视觉（VL）----
    "vl.cloud_enabled": bool,
    "vl.cloud_base_url": str,
    "vl.cloud_model": str,
    "vl.cloud_api_key": str,
    "vl.cloud_api_key_env": str,
    "vl.cloud_timeout_s": int,
    "vl.cloud_sequence_frames": int,
    "vl.sequence_frames": int,
    "vl.max_image_side": int,
    # ---- 本地模型 ----
    "models.device": str,  # auto / cpu / cuda
    "asr.device": str,
    "asr.compute_type": str,
    "embedding.text_device": str,
    "embedding.image_device": str,
    "ui.report_mode": str,
    "ui.theme_mode": str,
    # 注意：这里不放 models.preload —— AppConfig 的 ModelsSection 没有该字段，
    # 严格模式下写进去会让下一次 load_app_config 直接 UnknownConfigKeyError。
    # ---- 检索 ----
    "retrieval.top_k": int,
}

from dataclasses import fields as _policy_fields
from backend.analysis.policy import IntegritySection as _IntegritySection
SETTABLE_KEYS.update({"integrity." + field.name: field.type for field in _policy_fields(_IntegritySection) if field.name not in {'enabled', 'upload_chunk_bytes', 'upload_ttl_hours'}})
SETTABLE_KEYS.update({"media.download.format_max_height": int, "limits.max_file_bytes": int,
                     "limits.max_duration_ms": int, "limits.max_question_chars": int})

#: 允许的三个枚举值（写错会让配置校验抛错，不如这里先挡）。
_ENUMS: Final[dict[str, tuple[str, ...]]] = {
    "llm.mode": ("cloud",),
    "models.device": ("auto", "cpu", "cuda"),
    "asr.device": ("cpu", "cuda"),
    "asr.compute_type": ("int8", "float16", "int8_float16", "float32"),
    "embedding.text_device": ("cpu", "cuda"),
    "embedding.image_device": ("cpu", "cuda"),
    "ui.report_mode": ("user", "detailed"),
    "ui.theme_mode": ("system", "light", "dark"),
}

#: 这些键是密钥：读出来时打码，写日志时永不落明文。
_SECRET_KEYS: Final[frozenset[str]] = frozenset(
    {"llm.cloud.api_key", "vl.cloud_api_key"}
)


class SettingsError(ValueError):
    """设置不合法（未知键、类型不对、枚举越界）。"""


# --------------------------------------------------------------------------- #
# 路径与读写
# --------------------------------------------------------------------------- #
def settings_file(root: str | Path | None = None) -> Path:
    """设置文件路径（不保证存在）。"""
    return user_settings_path(root)


def _read_raw(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取设置文件 %s 失败，按空设置处理: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def current_settings(root: str | Path | None = None) -> dict[str, Any]:
    """读出已保存的设置（**密钥已打码**，可直接给界面展示）。"""
    raw = _read_raw(settings_file(root))
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _SECRET_KEYS and isinstance(value, str) and value:
            out[key] = _mask(value)
        else:
            out[key] = value
    return out


def _mask(secret: str) -> str:
    """固定数量的实心圆点，不暴露密钥内容或长度。"""
    return "●" * 8 if secret else ""


# --------------------------------------------------------------------------- #
# 校验与保存
# --------------------------------------------------------------------------- #
def _coerce(key: str, value: Any, expected: type) -> Any:
    """按白名单声明的类型转换；失败抛 :class:`SettingsError`。"""
    if expected is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off", ""}:
                return False
        raise SettingsError(f"{key} 需要布尔值，收到 {value!r}")

    if expected is int:
        if isinstance(value, bool):
            raise SettingsError(f"{key} 需要整数，收到布尔值")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise SettingsError(f"{key} 需要整数，收到 {value!r}") from exc

    if expected is float:
        if isinstance(value, bool):
            raise SettingsError(f"{key} 需要数字，收到布尔值")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise SettingsError(f"{key} 需要数字，收到 {value!r}") from exc

    return "" if value is None else str(value).strip()


def validate_settings(values: dict[str, Any]) -> dict[str, Any]:
    """校验并归一化一批设置。**全部通过才返回**，任一不合法就整体抛错。"""
    if not isinstance(values, dict):
        raise SettingsError("设置必须是字典")

    clean: dict[str, Any] = {}
    unknown = sorted(key for key in values if key not in SETTABLE_KEYS)
    if unknown:
        raise SettingsError(f"这些键不支持在界面上修改: {', '.join(unknown)}")

    for key, value in values.items():
        if key in _SECRET_KEYS:
            error = secret_input_error(value)
            if error:
                raise SettingsError(f"{key}: {error}；已有设置未被修改")
        coerced = _coerce(key, value, SETTABLE_KEYS[key])
        allowed = _ENUMS.get(key)
        if allowed is not None and coerced not in allowed:
            raise SettingsError(f"{key} 只能是 {' / '.join(allowed)}，收到 {coerced!r}")
        clean[key] = coerced
    return clean


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """先写临时文件再 ``os.replace``：中途断电不会留下半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), prefix=".settings-", delete=False
    )
    try:
        with handle as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def save_settings(
    values: dict[str, Any],
    *,
    root: str | Path | None = None,
    merge: bool = True,
) -> Path:
    """校验后落盘，返回设置文件路径。

    :param values: 要写入的键值（``a.b.c`` 点号路径）。
    :param merge: ``True`` 与已有设置合并；``False`` 整份替换。
    """
    clean = validate_settings(values)
    if clean.get("vl.cloud_enabled") is False:
        raise SettingsError("视觉理解必须使用云端，不能关闭")
    path = settings_file(root)
    payload: dict[str, Any] = {} if not merge else _read_raw(path)

    for key, value in clean.items():
        # 空字符串表示"清除这一项"：密钥尤其需要这个语义，否则用户清空输入框
        # 会被写成 "" 覆盖掉已有值，看起来像没生效。
        if value == "" and key in _SECRET_KEYS:
            payload.pop(key, None)
            continue
        payload[key] = value

    _atomic_write(path, payload)
    logger.info(
        "已保存 %d 项设置到 %s（%s）",
        len(clean),
        path,
        ", ".join(sorted(clean)),
    )
    return path


# --------------------------------------------------------------------------- #
# 展示
# --------------------------------------------------------------------------- #
def preview_settings(values, *, root=None):
    """校验完整配置而不写盘，设备切换成功后才保存；保留环境变量优先级。"""
    from backend.config import load_app_config

    clean = validate_settings(values)
    if clean.get("vl.cloud_enabled") is False:
        raise SettingsError("视觉理解必须使用云端，不能关闭")
    merged = _read_raw(settings_file(root))
    for key, value in clean.items():
        if key in _SECRET_KEYS and value == "":
            merged.pop(key, None)
        else:
            merged[key] = value
    return clean, load_app_config(root=root, user_settings_values=merged)


@dataclass(frozen=True, slots=True)
class BackendStatus:
    """一个后端通道的当前状态，供界面直接渲染。"""

    name: str
    enabled: bool
    detail: str
    warning: str = ""

    def line(self) -> str:
        mark = "已启用" if self.enabled else "未启用"
        text = f"**{self.name}**：{mark} — {self.detail}"
        if self.warning:
            text += f"\n  - ⚠️ {self.warning}"
        return text


def describe_backends(config: AppConfig | None = None) -> list[BackendStatus]:
    """把四个通道（LLM 云端/本地、视觉云端/本地）的实际状态摊平。

    界面需要"我现在到底在用哪个模型"这个问题的答案，而不是让用户去猜配置里
    三个开关的相互作用。所以这里照 :meth:`LlmGateway.from_config` 与
    :meth:`VlService.from_config` 的真实判定顺序模拟一遍。
    """
    cfg = config or get_app_config()
    llm_cfg = cfg.llm
    vl_cfg = cfg.vl
    cloud = llm_cfg.cloud
    out: list[BackendStatus] = []

    # -- LLM 云端 --
    want_cloud = llm_cfg.mode == "cloud"
    key = cloud.resolved_key()
    key_error = secret_input_error(key)
    cloud_warning = ""
    if key_error:
        cloud_warning = key_error
    elif not key:
        cloud_warning = (
            f"没有密钥：在界面里填，或设环境变量 {cloud.api_key_env}"
            if cloud.api_key_env
            else "没有密钥：在界面里填"
        )
    elif not cloud.base_url:
        cloud_warning = "base_url 为空"
    elif not cloud.model:
        cloud_warning = "模型名为空"
    out.append(
        BackendStatus(
            name="云端大模型",
            enabled=bool(want_cloud and key and cloud.base_url and cloud.model and not key_error),
            detail=f"{cloud.model} @ {cloud.base_url}"
            if cloud.base_url
            else "未配置地址",
            warning=""
            if (want_cloud and key and cloud.base_url and cloud.model and not key_error)
            else (cloud_warning or "云端通道未配置"),
        )
    )

    # -- 视觉云端 --
    vl_key = (vl_cfg.cloud_api_key or "").strip() or (
        (os.environ.get(vl_cfg.cloud_api_key_env) or "").strip()
        if vl_cfg.cloud_api_key_env
        else ""
    )
    vl_cloud_ok = bool(
        vl_cfg.cloud_enabled and vl_cfg.cloud_base_url and vl_cfg.cloud_model
    )
    vl_warning = ""
    vl_key_error = secret_input_error(vl_key)
    if vl_cfg.cloud_enabled and vl_key_error:
        vl_warning = vl_key_error
        vl_cloud_ok = False
    elif vl_cfg.cloud_enabled and not vl_cfg.cloud_base_url:
        vl_warning = "已开启但 base_url 为空"
    elif vl_cfg.cloud_enabled and not vl_cfg.cloud_model:
        vl_warning = "已开启但 model 为空"
    elif vl_cloud_ok and not vl_key:
        vl_warning = "缺少云端视觉密钥"
    out.append(
        BackendStatus(
            name="云端视觉",
            enabled=vl_cloud_ok,
            detail=(
                f"{vl_cfg.cloud_model} @ {vl_cfg.cloud_base_url}"
                if vl_cfg.cloud_base_url
                else "未配置地址"
            ),
            warning=vl_warning,
        )
    )

    return out


def format_backends(config: AppConfig | None = None) -> str:
    """把 :func:`describe_backends` 渲染成 Markdown（Gradio 直接显示）。"""
    statuses = describe_backends(config)
    lines = [status.line() for status in statuses]
    active = sum(1 for status in statuses if status.enabled)
    header = f"当前有 **{active}/{len(statuses)}** 个通道可用。"
    return "\n\n".join([header, *lines])
