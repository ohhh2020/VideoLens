"""知影 (ZhiYing) —— 本地模型权重管理器。

设计边界
--------
本模块**只负责模型权重文件的供应链**：定位、下载、校验、版本记录、清理。
它不负责加载推理会话（那是各 ``core/adapters/*`` 的职责），但它是唯一
允许知道"权重在哪儿"的组件。任何业务代码都必须通过 :meth:`ModelManager.ensure_model`
拿到 ``Path``，禁止在别处拼接模型目录或写死权重路径。

核心契约
--------
1. ``ensure_model(name) -> Path``：本地完整则直接返回；否则按 ``modelscope -> huggingface``
   顺序拉取，任一源成功即返回。
2. 下载与落盘是**原子**的：先写入 ``<cache>/.partial/<name>-<ts>``，校验通过后
   ``os.replace`` 换入最终目录，旧目录进 ``.old-*`` 后删除。中途断电不会留下半成品目录。
3. 每个模型目录内写 ``manifest.json``（来源、revision、文件清单、总大小、下载时间），
   供"模型变更时全量重建"与审计使用。
4. 单一阶段失败不抛出裸异常，全部包装为 :class:`ModelManagerError` 子类，携带
   ``model``/``source``/``attempts`` 上下文，便于上层按错误码降级。

国内网络
--------
``modelscope`` 优先；失败回退 ``huggingface_hub``，并可用 ``network.hf_endpoint``
（默认 ``https://hf-mirror.com``）走镜像。注意 ``HF_ENDPOINT`` 必须在首次
``import huggingface_hub`` 之前写入环境变量，因此本模块在顶层**不导入** hf/modlescope，
一律延迟到真正下载时导入。

作者: 知影团队
"""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import importlib
import inspect
import json
import logging
import os
import random
import shutil
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

try:  # PyYAML 是硬依赖，但缺失时给出可执行的修复提示而不是 ImportError 栈
    import yaml
except ImportError:  # pragma: no cover - 依赖缺失路径
    yaml = None  # type: ignore[assignment]

try:  # tqdm 缺失时降级为静默进度，保证 CI 无依赖环境仍可跑通
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - 依赖缺失路径
    _tqdm = None  # type: ignore[assignment]


__version__ = "1.0.0"
__all__ = [
    "CacheConfig",
    "ConfigError",
    "DownloadReport",
    "MissingDependencyError",
    "ModelDownloadError",
    "ModelIntegrityError",
    "ModelManager",
    "ModelManagerConfig",
    "ModelManagerError",
    "ModelSpec",
    "ModelState",
    "ModelStatus",
    "NetworkConfig",
    "SourceSpec",
    "UnknownModelError",
    "PROJECT_ROOT",
    "configure_logging",
    "get_model_manager",
    "reset_model_manager",
]

LOGGER_NAME = "zhiying.models"
_LOG = logging.getLogger(LOGGER_NAME)

ENV_CONFIG = "ZHIYING_CONFIG"
ENV_MODEL_DIR = "ZHIYING_MODEL_DIR"
DEFAULT_CONFIG_RELPATH = Path("configs") / "models.yaml"
MANIFEST_SCHEMA = 1

#: 下载目录内不参与 manifest 统计的目录（hf/modlescope 的元数据缓存）
_IGNORED_DIR_NAMES: frozenset[str] = frozenset({".cache", ".git", "__pycache__"})

#: HTTP 状态码中"重试有意义"的集合（限流、网关、服务端错误）
_RETRYABLE_STATUS: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 507})

#: 明确不值得重试的异常类名（仓库不存在、鉴权失败、离线模式未命中）
_NON_RETRYABLE_EXC_NAMES: frozenset[str] = frozenset(
    {
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "EntryNotFoundError",
        "GatedRepoError",
        "LocalEntryNotFoundError",
        "DisabledRepoError",
        "HTTPError",  # requests.HTTPError：状态码已在上面判定，其余视为不可重试
        "AuthError",
        "NotLoginException",
        "HTTPErrorNotFound",
        "ValueError",
        "TypeError",
    }
)


# --------------------------------------------------------------------------- #
# 异常体系
# --------------------------------------------------------------------------- #
class ModelManagerError(RuntimeError):
    """模型管理器所有异常的基类。上层按此类型统一捕获并触发降级。"""


class ConfigError(ModelManagerError):
    """配置文件缺失、字段类型非法或模型名非法。"""


class UnknownModelError(ModelManagerError, KeyError):
    """请求的模型名不在 ``configs/models.yaml`` 中。"""

    def __init__(self, name: str, known: Sequence[str]) -> None:
        self.name = name
        self.known = tuple(known)
        super().__init__(f"未配置的模型: {name!r}；可选: {', '.join(self.known) or '<空>'}")

    def __str__(self) -> str:  # KeyError 会给消息加引号，这里去掉
        return str(self.args[0])


class ModelDownloadError(ModelManagerError):
    """所有下载源均失败。``errors`` 保留每个源的失败原因，便于日志与界面展示。"""

    def __init__(self, model: str, errors: Sequence[str], attempts: int) -> None:
        self.model = model
        self.errors = tuple(errors)
        self.attempts = attempts
        detail = " | ".join(self.errors) if self.errors else "无可用下载源"
        super().__init__(f"模型 {model!r} 拉取失败（{attempts} 次尝试）: {detail}")


class ModelIntegrityError(ModelManagerError):
    """下载完成但关键文件缺失或哈希不符。"""

    def __init__(self, model: str, detail: str) -> None:
        self.model = model
        self.detail = detail
        super().__init__(f"模型 {model!r} 完整性校验未通过: {detail}")


class MissingDependencyError(ModelManagerError):
    """缺少某个下载后端（modelscope / huggingface_hub）。"""


# --------------------------------------------------------------------------- #
# 进度条抽象：让 tqdm 与"No-op 实现"共享同一协议
# --------------------------------------------------------------------------- #
class ProgressHandle(Protocol):
    """最小进度句柄协议，``tqdm`` 与 :class:`_NullBar` 都满足它。"""

    @property
    def n(self) -> float: ...

    def update(self, n: float = 1) -> Any: ...

    def refresh(self) -> None: ...

    def close(self) -> None: ...

    def set_description(self, desc: str | None, refresh: bool = True) -> None: ...


class _NullBar:
    """``tqdm`` 不可用时的空实现，保持调用侧代码零分支。"""

    __slots__ = ("_n",)

    def __init__(self) -> None:
        self._n = 0.0

    @property
    def n(self) -> float:
        return self._n

    def update(self, n: float = 1) -> None:
        self._n += n

    def refresh(self) -> None:
        return None

    def close(self) -> None:
        return None

    def set_description(self, desc: str | None, refresh: bool = True) -> None:
        return None


# --------------------------------------------------------------------------- #
# 配置模型
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SourceSpec:
    """单个下载源的仓库坐标。"""

    repo_id: str
    revision: str = "main"
    allow_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """一个可管理模型的静态描述（完全来自 ``models.yaml``）。"""

    name: str
    kind: str
    required: bool
    description: str
    sources: Mapping[str, SourceSpec]
    required_files: tuple[str, ...] = ()
    allow_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = ()
    size_hint_bytes: int = 0
    license: str = "unknown"
    targets: Mapping[str, Any] = field(default_factory=dict)
    sha256: Mapping[str, str] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    @property
    def directory_name(self) -> str:
        """落盘目录名。模型名已通过 :func:`_validate_model_name` 白名单校验。"""
        return self.name


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    """网络与重试策略。"""

    sources_order: tuple[str, ...] = ("modelscope", "huggingface")
    timeout: float = 60.0
    etag_timeout: float = 30.0
    retries: int = 4
    backoff_base: float = 2.0
    backoff_max: float = 45.0
    backoff_jitter: float = 0.35
    progress_interval: float = 0.4
    max_workers: int = 4
    hf_endpoint: str | None = None
    modelscope_endpoint: str | None = None
    hf_offline: bool = False
    show_progress: bool = True
    verify_sha256: bool = False


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """本地缓存布局。"""

    root: Path
    partial_dirname: str = ".partial"
    manifest_name: str = "manifest.json"
    partial_retention_hours: int = 24

    @property
    def partial_root(self) -> Path:
        return self.root / self.partial_dirname


@dataclass(frozen=True, slots=True)
class ModelManagerConfig:
    """解析后的完整配置快照。"""

    config_path: Path
    cache: CacheConfig
    network: NetworkConfig
    models: Mapping[str, ModelSpec]
    version: int = 1
    raw: Mapping[str, Any] = field(default_factory=dict)


class ModelState(str, Enum):
    """本地模型状态。"""

    MISSING = "missing"        # 目录不存在
    INCOMPLETE = "incomplete"  # 目录存在但关键文件缺失（半成品/被手工删改）
    CORRUPT = "corrupt"        # 哈希校验失败
    PRESENT = "present"        # 可用


@dataclass(frozen=True, slots=True)
class ModelStatus:
    """``ensure_model`` / ``status`` 的结构化检查结果。"""

    name: str
    kind: str
    required: bool
    state: ModelState
    path: Path
    size_bytes: int = 0
    file_count: int = 0
    source: str | None = None
    revision: str | None = None
    missing_files: tuple[str, ...] = ()
    message: str = ""

    @property
    def available(self) -> bool:
        return self.state is ModelState.PRESENT


@dataclass(frozen=True, slots=True)
class DownloadReport:
    """批量拉取结果汇总。"""

    ok: Mapping[str, Path] = field(default_factory=dict)
    failed: Mapping[str, str] = field(default_factory=dict)
    skipped: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return not self.failed


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _as_tuple(value: Any) -> tuple[str, ...]:
    """把 yaml 里的 str / list 统一成 ``tuple[str, ...]``。"""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value if str(item).strip())
    raise ConfigError(f"期望字符串或字符串列表，得到 {type(value).__name__}")


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _detect_project_root() -> Path:
    """定位项目根：环境变量 > 向上搜索 ``configs/models.yaml`` > 本文件上溯两层。"""
    env = os.environ.get("ZHIYING_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / DEFAULT_CONFIG_RELPATH).is_file():
            return parent
        if (parent / "pyproject.toml").is_file() and (parent / "core").is_dir():
            return parent
    # core/models/manager.py -> parents[0]=models, [1]=core, [2]=项目根
    return here.parents[2]


#: 项目根目录（决定默认的 ``configs/models.yaml`` 与 ``models/`` 位置）
PROJECT_ROOT: Path = _detect_project_root()


def _validate_model_name(name: str) -> str:
    """模型名同时是目录名，必须限制为安全字符集，防止路径穿越。"""
    if not name or not isinstance(name, str):
        raise ConfigError(f"非法模型名: {name!r}")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if set(name) - allowed or name in {".", ".."} or name.startswith("."):
        raise ConfigError(f"模型名只允许 [A-Za-z0-9._-] 且不得以 '.' 开头: {name!r}")
    return name


def _dir_size(root: Path, *, include_metadata: bool = False) -> int:
    """递归统计目录字节数。

    ``include_metadata=True`` 时把库自身的缓存目录也算进来——下载中的分块文件
    （如 hf 的 ``.cache/huggingface/download/*.incomplete``）就躺在那儿，进度条
    必须统计它们，否则进度会长时间停在 0 然后突然跳到 100。
    """
    total = 0
    try:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if not include_metadata and any(
                part in _IGNORED_DIR_NAMES for part in path.relative_to(root).parts[:-1]
            ):
                continue
            try:
                total += path.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def _rmtree(path: Path) -> None:
    """尽力删除目录，失败只记日志（Windows 上句柄未释放是常态）。"""
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except OSError as exc:  # pragma: no cover - 平台相关
        _LOG.warning("清理目录失败 %s: %s", path, exc)


def _sha256_of(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _status_of_exception(exc: BaseException) -> int | None:
    """尽最大努力从各类 HTTP 异常里挖出状态码。"""
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _is_retryable(exc: BaseException) -> bool:
    """判断异常是否值得重试：网络抖动重试，配置/鉴权错误立即失败。"""
    if isinstance(exc, (KeyboardInterrupt, SystemExit, MemoryError)):
        return False
    if isinstance(exc, (FileNotFoundError, PermissionError, IsADirectoryError, ConfigError)):
        return False
    if type(exc).__name__ in _NON_RETRYABLE_EXC_NAMES:
        return False
    status = _status_of_exception(exc)
    if status is not None:
        if status in (401, 403, 404, 400, 405, 422):
            return False
        return status in _RETRYABLE_STATUS or status >= 500
    # 网络类异常：连接被重置、DNS 抖动、读超时
    if isinstance(exc, (ConnectionError, TimeoutError, socket.gaierror, BrokenPipeError)):
        return True
    if isinstance(exc, OSError):
        return True
    if isinstance(exc, (ValueError, TypeError, NotImplementedError)):
        return False
    return True


@contextlib.contextmanager
def _socket_timeout(seconds: float | None) -> Iterator[None]:
    """临时把默认 socket 超时设为 ``seconds``。

    ``modelscope`` 的 ``snapshot_download`` 不接受 timeout 参数，只能通过全局默认值
    生效；退出时无条件还原，避免污染进程内其它网络调用。
    """
    previous = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(seconds)
        yield
    finally:
        socket.setdefaulttimeout(previous)


def configure_logging(
    *,
    level: int = logging.INFO,
    log_file: Path | None = None,
    force: bool = False,
) -> logging.Logger:
    """为模型管理器单独挂一个 handler。

    默认只往 stderr 输出，不触碰 root logger；传入 ``log_file`` 时额外写
    轮转文件（10MB × 3），"日志不得记录密钥"由调用方保证不把 key 传进来。
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
    if not logger.handlers:
        stream = logging.StreamHandler(stream=sys.stderr)
        stream.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(stream)
    if log_file is not None and not any(
        isinstance(h, RotatingFileHandler) for h in logger.handlers
    ):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# 配置解析
# --------------------------------------------------------------------------- #
def _resolve_config_path(config_path: str | Path | None) -> Path:
    candidates: list[Path] = []
    if config_path is not None:
        candidates.append(Path(config_path).expanduser())
    env_value = os.environ.get(ENV_CONFIG)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(PROJECT_ROOT / DEFAULT_CONFIG_RELPATH)

    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
        resolved = resolved.resolve()
        if resolved.is_file():
            return resolved
    searched = ", ".join(str(c.resolve()) for c in candidates)
    raise ConfigError(
        f"未找到模型配置文件 models.yaml。已尝试: {searched}；"
        f"可通过构造函数参数或环境变量 {ENV_CONFIG} 指定路径。"
    )


def _parse_source(raw: Any, default_revision: str, where: str) -> SourceSpec:
    if isinstance(raw, str):
        return SourceSpec(repo_id=raw.strip(), revision=default_revision)
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where}: 下载源应为字符串或映射，得到 {type(raw).__name__}")
    repo_id = raw.get("repo_id") or raw.get("repo") or raw.get("model_id")
    if not repo_id:
        raise ConfigError(f"{where}: 缺少 repo_id 字段")
    return SourceSpec(
        repo_id=str(repo_id).strip(),
        revision=str(raw.get("revision") or default_revision),
        allow_patterns=_as_tuple(raw.get("allow_patterns")),
        ignore_patterns=_as_tuple(raw.get("ignore_patterns")),
    )


def _parse_model(name: str, raw: Mapping[str, Any], default_revision: str) -> ModelSpec:
    where = f"models.{name}"
    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, Mapping) or not sources_raw:
        raise ConfigError(f"{where}: sources 必须是一个非空映射（至少包含 modelscope 或 huggingface）")
    sources = {
        str(key).strip().lower(): _parse_source(value, default_revision, f"{where}.sources.{key}")
        for key, value in sources_raw.items()
    }
    size_hint_gb = _as_float(raw.get("size_hint_gb"), 0.0)
    return ModelSpec(
        name=_validate_model_name(name),
        kind=str(raw.get("kind") or "unknown"),
        required=_as_bool(raw.get("required"), True),
        description=str(raw.get("description") or ""),
        sources=sources,
        required_files=_as_tuple(raw.get("required_files")),
        allow_patterns=_as_tuple(raw.get("allow_patterns")),
        ignore_patterns=_as_tuple(raw.get("ignore_patterns")),
        size_hint_bytes=int(size_hint_gb * (1024**3)) if size_hint_gb > 0 else _as_int(raw.get("size_hint_bytes"), 0),
        license=str(raw.get("license") or "unknown"),
        targets=dict(raw.get("targets") or {}),
        sha256={str(k): str(v) for k, v in dict(raw.get("sha256") or {}).items()},
        tags=_as_tuple(raw.get("tags")),
    )


def _effective_allow_patterns(spec: ModelSpec, source_spec: SourceSpec) -> tuple[str, ...]:
    """解析某个下载源最终生效的 allow 白名单。

    规则：**源级覆盖模型级**。``sources.<name>.allow_patterns`` 显式声明时以它为准
    （某些源仓库的布局与主仓库不同，需要单独收窄），否则回退到模型级
    ``models.<name>.allow_patterns``。这条回退是关键的：模型级白名单写在 yaml 顶部，
    如果只读源级，GGUF 仓库会把全部量化版本（数十 GB）都拉下来。
    """
    if source_spec.allow_patterns:
        return tuple(source_spec.allow_patterns)
    return tuple(spec.allow_patterns)


def _effective_ignore_patterns(spec: ModelSpec, source_spec: SourceSpec) -> tuple[str, ...]:
    """解析某个下载源最终生效的 ignore 黑名单：源级与模型级取并集（都排除）。"""
    merged: list[str] = []
    for pattern in tuple(source_spec.ignore_patterns) + tuple(spec.ignore_patterns):
        if pattern not in merged:
            merged.append(pattern)
    return tuple(merged)


def _parse_config(raw: Mapping[str, Any], config_path: Path, cache_root: Path | None) -> ModelManagerConfig:
    if not isinstance(raw, Mapping):
        raise ConfigError("models.yaml 顶层必须是映射")
    version = _as_int(raw.get("version"), 1)

    cache_raw = raw.get("cache") or {}
    if not isinstance(cache_raw, Mapping):
        raise ConfigError("cache 段必须是映射")
    root_value = cache_root
    if root_value is None:
        env_root = os.environ.get(ENV_MODEL_DIR)
        if env_root:
            root_value = Path(env_root).expanduser()
        elif cache_raw.get("root"):
            root_value = Path(str(cache_raw["root"])).expanduser()
        else:
            root_value = PROJECT_ROOT / "models"
    root_resolved = Path(root_value)
    if not root_resolved.is_absolute():
        root_resolved = (PROJECT_ROOT / root_resolved).resolve()
    cache = CacheConfig(
        root=root_resolved,
        partial_dirname=str(cache_raw.get("partial_dirname") or ".partial"),
        manifest_name=str(cache_raw.get("manifest_name") or "manifest.json"),
        partial_retention_hours=_as_int(cache_raw.get("partial_retention_hours"), 24),
    )

    net_raw = raw.get("network") or {}
    if not isinstance(net_raw, Mapping):
        raise ConfigError("network 段必须是映射")
    sources_order = _as_tuple(net_raw.get("sources_order")) or ("modelscope", "huggingface")
    sources_order = tuple(s.lower() for s in sources_order)
    hf_endpoint = net_raw.get("hf_endpoint")
    network = NetworkConfig(
        sources_order=sources_order,
        timeout=_as_float(net_raw.get("timeout"), 60.0),
        etag_timeout=_as_float(net_raw.get("etag_timeout"), 30.0),
        retries=max(1, _as_int(net_raw.get("retries"), 4)),
        backoff_base=max(0.1, _as_float(net_raw.get("backoff_base"), 2.0)),
        backoff_max=max(1.0, _as_float(net_raw.get("backoff_max"), 45.0)),
        backoff_jitter=min(1.0, max(0.0, _as_float(net_raw.get("backoff_jitter"), 0.35))),
        progress_interval=max(0.1, _as_float(net_raw.get("progress_interval"), 0.4)),
        max_workers=max(1, _as_int(net_raw.get("max_workers"), 4)),
        hf_endpoint=str(hf_endpoint).rstrip("/") if hf_endpoint else None,
        modelscope_endpoint=(
            str(net_raw["modelscope_endpoint"]).rstrip("/")
            if net_raw.get("modelscope_endpoint")
            else None
        ),
        hf_offline=_as_bool(net_raw.get("hf_offline"), False),
        show_progress=_as_bool(net_raw.get("show_progress"), True),
        verify_sha256=_as_bool(net_raw.get("verify_sha256"), False),
    )

    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, Mapping):
        raise ConfigError("defaults 段必须是映射")
    default_revision = str(defaults.get("revision") or "main")

    models_raw = raw.get("models")
    if not isinstance(models_raw, Mapping) or not models_raw:
        raise ConfigError("models 段必须是非空映射")
    models: dict[str, ModelSpec] = {}
    for name, item in models_raw.items():
        if not isinstance(item, Mapping):
            raise ConfigError(f"models.{name} 必须是映射")
        spec = _parse_model(str(name), item, default_revision)
        if spec.name in models:
            raise ConfigError(f"模型名重复: {spec.name}")
        models[spec.name] = spec
    for model_name, spec in models.items():
        unknown = [s for s in spec.sources if s not in {"modelscope", "huggingface"}]
        if unknown:
            raise ConfigError(f"models.{model_name}.sources 含未知下载源: {unknown}")

    return ModelManagerConfig(
        config_path=config_path,
        cache=cache,
        network=network,
        models=models,
        version=version,
        raw=dict(raw),
    )


# --------------------------------------------------------------------------- #
# 主类
# --------------------------------------------------------------------------- #
class ModelManager:
    """模型权重供应链管理器（线程安全、可多实例、幂等）。

    典型用法::

        manager = ModelManager()
        weights = manager.ensure_model("bge-m3")     # -> Path
        report = manager.ensure_all()                # 全量预热

    实例内部对每个模型加独立锁，同一模型并发 ``ensure_model`` 只会真正下载一次。
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        cache_root: str | Path | None = None,
        logger: logging.Logger | None = None,
        show_progress: bool | None = None,
        auto_create_dirs: bool = True,
        allow_download: bool = True,
    ) -> None:
        self._log = logger or _LOG
        self._allow_download = allow_download
        raw_config = self._read_yaml(config_path)
        config_file = _resolve_config_path(config_path)
        self._cfg: ModelManagerConfig = _parse_config(
            raw_config, config_file, Path(cache_root) if cache_root else None
        )
        self._show_progress = (
            self._cfg.network.show_progress if show_progress is None else bool(show_progress)
        )
        self._lock = threading.RLock()
        self._model_locks: dict[str, threading.RLock] = {}
        self._device_hint: str | None = None
        self._apply_environment()
        if auto_create_dirs:
            self._cfg.cache.root.mkdir(parents=True, exist_ok=True)
            self._cfg.cache.partial_root.mkdir(parents=True, exist_ok=True)
        self._log.debug(
            "ModelManager 就绪: config=%s cache=%s models=%d",
            self._cfg.config_path,
            self._cfg.cache.root,
            len(self._cfg.models),
        )

    # ------------------------------------------------------------------ #
    # 构造期辅助
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_yaml(config_path: str | Path | None) -> Mapping[str, Any]:
        if yaml is None:
            raise MissingDependencyError(
                "解析 models.yaml 需要 PyYAML，请执行: pip install pyyaml"
            )
        path = _resolve_config_path(config_path)
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:  # type: ignore[union-attr]
            raise ConfigError(f"models.yaml 语法错误 ({path}): {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"无法读取模型配置 {path}: {exc}") from exc
        if data is None:
            raise ConfigError(f"模型配置为空: {path}")
        return data

    def _apply_environment(self) -> None:
        """把配置写进第三方库读取的环境变量。

        ``huggingface_hub`` 在**模块导入时**读取 ``HF_ENDPOINT`` 等常量，因此这里
        设置 env，并在 hf 已被别处导入的情况下同步常量值，保证镜像配置真的生效。
        """
        net = self._cfg.network
        if net.hf_endpoint:
            os.environ["HF_ENDPOINT"] = net.hf_endpoint
            self._sync_loaded_hf_constants(net.hf_endpoint)
        if net.hf_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(int(net.timeout)))
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(int(net.etag_timeout)))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        if net.modelscope_endpoint:
            os.environ.setdefault("MODELSCOPE_DOMAIN", net.modelscope_endpoint)

    @staticmethod
    def _sync_loaded_hf_constants(endpoint: str) -> None:
        module = sys.modules.get("huggingface_hub")
        if module is None:
            return
        try:
            from huggingface_hub import constants  # noqa: PLC0415 - 延迟导入是刻意的

            if getattr(constants, "ENDPOINT", None) != endpoint:
                constants.ENDPOINT = endpoint  # type: ignore[misc]
                constants.HUGGINGFACE_CO_URL_TEMPLATE = (  # type: ignore[misc]
                    endpoint.rstrip("/") + "/{repo_id}/resolve/{revision}/{filename}"
                )
                _LOG.info("huggingface_hub 已被导入，已同步 ENDPOINT=%s", endpoint)
        except Exception as exc:  # pragma: no cover - 版本差异
            _LOG.debug("同步 huggingface_hub ENDPOINT 失败（忽略）: %s", exc)

    # ------------------------------------------------------------------ #
    # 只读属性与查询
    # ------------------------------------------------------------------ #
    @property
    def config(self) -> ModelManagerConfig:
        return self._cfg

    @property
    def cache_root(self) -> Path:
        return self._cfg.cache.root

    @property
    def device_hint(self) -> str:
        """粗粒度设备探测结果（供上层决定量化等级/线程数），不影响权重下载。"""
        if self._device_hint is None:
            self._device_hint = self._detect_device()
        return self._device_hint

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def list_models(self, *, required_only: bool = False) -> tuple[str, ...]:
        names = tuple(self._cfg.models)
        if required_only:
            return tuple(n for n in names if self._cfg.models[n].required)
        return names

    def get_spec(self, name: str) -> ModelSpec:
        try:
            return self._cfg.models[_validate_model_name(name)]
        except KeyError as exc:
            raise UnknownModelError(name, self.list_models()) from exc

    def model_dir(self, name: str) -> Path:
        """最终落盘目录（纯路径计算，不碰磁盘）。"""
        return self._cfg.cache.root / self.get_spec(name).directory_name

    def is_available(self, name: str) -> bool:
        return self.status(name).available

    # ------------------------------------------------------------------ #
    # 状态检查
    # ------------------------------------------------------------------ #
    def status(self, name: str, *, verify_hash: bool | None = None) -> ModelStatus:
        """检查单个模型本地状态，不触发任何网络请求。"""
        spec = self.get_spec(name)
        return self._inspect(spec, verify_hash=verify_hash)

    def status_report(self, *, verify_hash: bool | None = None) -> tuple[ModelStatus, ...]:
        return tuple(self._inspect(spec, verify_hash=verify_hash) for spec in self._cfg.models.values())

    def _inspect(self, spec: ModelSpec, *, verify_hash: bool | None = None) -> ModelStatus:
        dest = self._cfg.cache.root / spec.directory_name
        if not dest.is_dir():
            return ModelStatus(
                name=spec.name,
                kind=spec.kind,
                required=spec.required,
                state=ModelState.MISSING,
                path=dest,
                message="目录不存在，需要下载",
            )

        manifest = self._read_manifest(dest)
        size = _dir_size(dest)
        file_count = len(self._scan_files(dest, spec))
        source = str(manifest.get("source")) if manifest and manifest.get("source") else None
        revision = str(manifest.get("revision")) if manifest and manifest.get("revision") else None

        missing = self._missing_files(spec, dest)
        if missing:
            return ModelStatus(
                name=spec.name,
                kind=spec.kind,
                required=spec.required,
                state=ModelState.INCOMPLETE,
                path=dest,
                size_bytes=size,
                file_count=file_count,
                source=source,
                revision=revision,
                missing_files=missing,
                message=f"缺少关键文件: {', '.join(missing)}",
            )

        want_hash = self._cfg.network.verify_sha256 if verify_hash is None else verify_hash
        if want_hash and spec.sha256:
            mismatch = self._verify_hashes(spec, dest)
            if mismatch:
                return ModelStatus(
                    name=spec.name,
                    kind=spec.kind,
                    required=spec.required,
                    state=ModelState.CORRUPT,
                    path=dest,
                    size_bytes=size,
                    file_count=file_count,
                    source=source,
                    revision=revision,
                    message=f"哈希不符: {', '.join(mismatch)}",
                )

        return ModelStatus(
            name=spec.name,
            kind=spec.kind,
            required=spec.required,
            state=ModelState.PRESENT,
            path=dest,
            size_bytes=size,
            file_count=file_count,
            source=source,
            revision=revision,
            message="可用",
        )

    def _verify_hashes(self, spec: ModelSpec, root: Path) -> list[str]:
        mismatched: list[str] = []
        for rel_name, expected in spec.sha256.items():
            target = root / rel_name
            if not target.is_file():
                mismatched.append(rel_name)
                continue
            try:
                actual = _sha256_of(target)
            except OSError as exc:
                self._log.warning("哈希计算失败 %s: %s", target, exc)
                mismatched.append(rel_name)
                continue
            if actual.lower() != expected.lower():
                mismatched.append(rel_name)
        return mismatched

    def _missing_files(self, spec: ModelSpec, root: Path) -> tuple[str, ...]:
        if not spec.required_files:
            return ()
        missing: list[str] = []
        for pattern in spec.required_files:
            if not self._pattern_hits(root, pattern):
                missing.append(pattern)
        return tuple(missing)

    @staticmethod
    def _pattern_hits(root: Path, pattern: str) -> bool:
        """按 glob 判定关键文件是否存在且非空；一级不中再全树兜底。"""
        try:
            candidates = list(root.glob(pattern))
            if not candidates:
                candidates = list(root.rglob(pattern))
        except (OSError, ValueError):
            return False
        for path in candidates:
            try:
                if path.is_file() and path.stat().st_size > 0:
                    return True
            except OSError:
                continue
        return False

    # ------------------------------------------------------------------ #
    # 核心入口
    # ------------------------------------------------------------------ #
    def ensure_model(
        self,
        name: str,
        *,
        force: bool = False,
        verify_hash: bool | None = None,
        source: str | None = None,
        show_progress: bool | None = None,
    ) -> Path:
        """确保模型本地可用并返回权重目录。

        Args:
            name: ``models.yaml`` 中的模型名。
            force: 为 True 时忽略现有目录，重新下载（用于修复 CORRUPT/版本更新）。
            verify_hash: 是否做 sha256 校验；None 表示遵循配置 ``network.verify_sha256``。
            source: 强制指定下载源（``modelscope``/``huggingface``），None 表示按配置顺序回退。
            show_progress: 覆盖实例级进度条开关。

        Returns:
            模型权重目录的绝对路径。

        Raises:
            UnknownModelError: 模型名未配置。
            ModelDownloadError: 所有源都失败。
            ModelIntegrityError: 下载完成但关键文件缺失。
        """
        spec = self.get_spec(name)
        # 先校验下载源合法性：无论本地是否已就绪，传错源都应当立即报错，
        # 而不是出现"有时报错、有时静默返回"的不一致行为。
        order = self._source_order(spec, source)
        with self._model_lock(spec.name):
            if not force:
                current = self._inspect(spec, verify_hash=verify_hash)
                if current.available:
                    self._log.debug(
                        "模型已就绪: %s (%s, %.1f MB)", spec.name, current.path, current.size_bytes / 1048576
                    )
                    return current.path
                if current.state is ModelState.INCOMPLETE:
                    self._log.warning("模型目录不完整，将重新拉取: %s (%s)", spec.name, current.message)

            if not self._allow_download:
                # 只读模式：权重不在本地就直接判失败，让上层走降级分支。
                # 生产推理绝不能在用户按下"分析"的瞬间悄悄拉几个 GB 的权重。
                raise ModelDownloadError(
                    spec.name,
                    [
                        f"本地缺少权重且已禁用自动下载（cache={self._cfg.cache.root}）；"
                        f"请先运行: python scripts/download_models.py --models {spec.name}"
                    ],
                    0,
                )

            errors: list[str] = []
            for source_name in order:
                source_spec = spec.sources.get(source_name)
                if source_spec is None:
                    continue
                partial = self._new_partial_dir(spec.name)
                try:
                    self._download(spec, source_name, source_spec, partial, show_progress)
                    self._finalize(spec, source_name, source_spec, partial)
                    self._log.info(
                        "模型拉取成功: %s <- %s/%s@%s",
                        spec.name,
                        source_name,
                        source_spec.repo_id,
                        source_spec.revision,
                    )
                    return self._cfg.cache.root / spec.directory_name
                except Exception as exc:  # noqa: BLE001 - 单源失败要继续尝试下一源
                    detail = f"{source_name}({source_spec.repo_id}): {type(exc).__name__}: {exc}"
                    errors.append(detail)
                    self._log.warning("模型拉取失败，尝试下一源 -> %s", detail)
                    _rmtree(partial)
            raise ModelDownloadError(spec.name, errors, self._cfg.network.retries * max(1, len(order)))

    def ensure_all(
        self,
        names: Sequence[str] | None = None,
        *,
        continue_on_error: bool = True,
        force: bool = False,
        verify_hash: bool | None = None,
        source: str | None = None,
    ) -> DownloadReport:
        """批量确保模型可用，返回结构化报告而不是在第一个失败处中断。"""
        targets = tuple(names) if names else self.list_models()
        ok: dict[str, Path] = {}
        failed: dict[str, str] = {}
        skipped: list[str] = []
        for model_name in targets:
            try:
                ok[model_name] = self.ensure_model(
                    model_name, force=force, verify_hash=verify_hash, source=source
                )
            except ModelManagerError as exc:
                failed[model_name] = str(exc)
                if not continue_on_error:
                    raise
            except Exception as exc:  # noqa: BLE001 - 未知异常也要归档，不能中断整批
                failed[model_name] = f"{type(exc).__name__}: {exc}"
                if not continue_on_error:
                    raise
        return DownloadReport(ok=ok, failed=failed, skipped=tuple(skipped))

    def remove(self, name: str, *, manifest_only: bool = False) -> bool:
        """删除模型目录（或只删 manifest）。返回是否真的删了东西。"""
        spec = self.get_spec(name)
        dest = self._cfg.cache.root / spec.directory_name
        if not dest.exists():
            return False
        with self._model_lock(spec.name):
            if manifest_only:
                manifest = dest / self._cfg.cache.manifest_name
                if manifest.is_file():
                    manifest.unlink()
                    return True
                return False
            _rmtree(dest)
        self._log.info("已删除模型目录: %s", dest)
        return True

    def prune_partials(self, older_than_hours: int | None = None) -> int:
        """清理过期下载暂存目录，返回清理数量（对应"临时缓存须可清理"约束）。"""
        threshold_hours = (
            self._cfg.cache.partial_retention_hours if older_than_hours is None else older_than_hours
        )
        cutoff = time.time() - threshold_hours * 3600
        removed = 0
        partial_root = self._cfg.cache.partial_root
        if not partial_root.is_dir():
            return 0
        for child in partial_root.iterdir():
            if not child.is_dir():
                continue
            try:
                if child.stat().st_mtime >= cutoff:
                    continue
            except OSError:
                continue
            _rmtree(child)
            removed += 1
        if removed:
            self._log.info("已清理 %d 个过期下载暂存目录", removed)
        return removed

    # ------------------------------------------------------------------ #
    # 内部: 下载编排
    # ------------------------------------------------------------------ #
    def _model_lock(self, name: str) -> threading.RLock:
        with self._lock:
            lock = self._model_locks.get(name)
            if lock is None:
                lock = threading.RLock()
                self._model_locks[name] = lock
            return lock

    def _source_order(self, spec: ModelSpec, source: str | None) -> tuple[str, ...]:
        if source:
            requested = source.strip().lower()
            if requested not in spec.sources:
                raise ConfigError(
                    f"模型 {spec.name} 没有配置下载源 {requested!r}；可用: {', '.join(spec.sources)}"
                )
            return (requested,)
        ordered = [s for s in self._cfg.network.sources_order if s in spec.sources]
        ordered += [s for s in spec.sources if s not in ordered]
        return tuple(ordered)

    def _new_partial_dir(self, name: str) -> Path:
        stamp = f"{int(time.time() * 1000)}-{os.getpid()}"
        partial = self._cfg.cache.partial_root / f"{name}-{stamp}"
        partial.mkdir(parents=True, exist_ok=True)
        return partial

    def _download(
        self,
        spec: ModelSpec,
        source_name: str,
        source_spec: SourceSpec,
        dest: Path,
        show_progress: bool | None,
    ) -> None:
        """在 ``dest`` 内完成一次下载（不带重试，重试在外层统一做）。"""
        if source_name == "modelscope":
            fetch: Callable[[], None] = lambda: self._fetch_modelscope(spec, source_spec, dest)
        elif source_name == "huggingface":
            fetch = lambda: self._fetch_huggingface(spec, source_spec, dest)  # noqa: E731
        else:  # pragma: no cover - 配置层已校验
            raise ConfigError(f"不支持的下载源: {source_name}")

        progress_enabled = self._show_progress if show_progress is None else bool(show_progress)
        desc = f"[{source_name}] {spec.name}"
        with self._byte_progress(desc, spec.size_hint_bytes or None, dest, show=progress_enabled):
            with _socket_timeout(self._cfg.network.timeout):
                self._retry_call(f"{source_name}:{spec.name}", fetch)

    def _retry_call(self, label: str, func: Callable[[], Any]) -> Any:
        """指数退避 + 抖动重试。不可重试异常立刻抛出，保留原始 traceback。"""
        net = self._cfg.network
        total = max(1, net.retries)
        for attempt in range(1, total + 1):
            started = time.monotonic()
            try:
                return func()
            except BaseException as exc:  # noqa: BLE001 - 需要按类型分类
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                if attempt >= total or not _is_retryable(exc):
                    self._log.error(
                        "%s 第 %d/%d 次尝试失败且不可重试: %s: %s",
                        label,
                        attempt,
                        total,
                        type(exc).__name__,
                        exc,
                    )
                    raise
                delay = min(net.backoff_max, net.backoff_base ** attempt)
                delay += random.uniform(0.0, delay * net.backoff_jitter)
                self._log.warning(
                    "%s 第 %d/%d 次尝试失败（%.1fs 后重试）: %s: %s",
                    label,
                    attempt,
                    total,
                    delay,
                    type(exc).__name__,
                    exc,
                )
                time.sleep(delay)
            finally:
                self._log.debug(
                    "%s attempt=%d elapsed=%.2fs", label, attempt, time.monotonic() - started
                )
        raise ModelDownloadError(label, ["重试耗尽"], total)  # pragma: no cover - 理论不可达

    # -- modelscope ---------------------------------------------------- #
    def _fetch_modelscope(self, spec: ModelSpec, source_spec: SourceSpec, dest: Path) -> None:
        module = self._import_backend(
            ("modelscope.hub.snapshot_download", "modelscope"),
            hint="modelscope",
            install_hint="pip install modelscope",
        )
        snapshot_download = getattr(module, "snapshot_download", None)
        if snapshot_download is None:  # pragma: no cover - 极老版本兜底
            from modelscope import snapshot_download as snapshot_download  # type: ignore[no-redef]

        params = inspect.signature(snapshot_download).parameters
        if "local_dir" not in params:
            raise MissingDependencyError(
                "当前 modelscope 版本不支持 local_dir 参数，请升级: pip install -U modelscope"
            )

        kwargs: dict[str, Any] = {"model_id": source_spec.repo_id, "local_dir": str(dest)}
        if "revision" in params and source_spec.revision:
            kwargs["revision"] = source_spec.revision
        allow_key = next((k for k in ("allow_file_pattern", "allow_patterns") if k in params), None)
        ignore_key = next((k for k in ("ignore_file_pattern", "ignore_patterns") if k in params), None)
        allow = _effective_allow_patterns(spec, source_spec)
        if allow_key and allow:
            kwargs[allow_key] = list(allow)
        combined_ignore = _effective_ignore_patterns(spec, source_spec)
        if ignore_key and combined_ignore:
            kwargs[ignore_key] = list(combined_ignore)
        if "max_workers" in params:
            kwargs["max_workers"] = self._cfg.network.max_workers
        if "disable_tqdm" in params:
            kwargs["disable_tqdm"] = True
        if "timeout" in params and self._cfg.network.timeout:
            kwargs["timeout"] = self._cfg.network.timeout

        self._log.debug("modelscope snapshot_download %s kwargs=%s", source_spec.repo_id, kwargs)
        snapshot_download(**kwargs)

    # -- huggingface --------------------------------------------------- #
    def _fetch_huggingface(self, spec: ModelSpec, source_spec: SourceSpec, dest: Path) -> None:
        hub = self._import_backend(
            ("huggingface_hub",), hint="huggingface_hub", install_hint="pip install huggingface_hub"
        )
        endpoint = self._cfg.network.hf_endpoint
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        with self._hf_progress_suppressed():
            hf_api_cls = getattr(hub, "HfApi")
            api_kwargs: dict[str, Any] = {}
            if endpoint:
                api_kwargs["endpoint"] = endpoint
            api = hf_api_cls(**api_kwargs)

            files: list[str] | None = None
            try:
                files = list(
                    api.list_repo_files(
                        repo_id=source_spec.repo_id,
                        revision=source_spec.revision,
                        token=token,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - 镜像可能不支持 list，退化为 snapshot
                if not _is_retryable(exc):
                    raise
                self._log.warning(
                    "list_repo_files 失败（%s），退回 snapshot_download: %s", source_spec.repo_id, exc
                )

            selected = self._select_files(files, source_spec, spec) if files is not None else None

            if selected is None:
                snapshot_download = getattr(hub, "snapshot_download")
                snap_kwargs: dict[str, Any] = {
                    "repo_id": source_spec.repo_id,
                    "revision": source_spec.revision,
                    "local_dir": str(dest),
                    "max_workers": self._cfg.network.max_workers,
                }
                if token:
                    snap_kwargs["token"] = token
                if endpoint:
                    snap_kwargs["endpoint"] = endpoint
                allow = _effective_allow_patterns(spec, source_spec)
                ignore = _effective_ignore_patterns(spec, source_spec)
                if allow:
                    snap_kwargs["allow_patterns"] = list(allow)
                if ignore:
                    snap_kwargs["ignore_patterns"] = list(ignore)
                self._log.debug("hf snapshot_download %s", source_spec.repo_id)
                snapshot_download(**snap_kwargs)
                return

            if not selected:
                raise ModelIntegrityError(
                    spec.name, "按 allow/ignore 规则过滤后没有任何文件可下载"
                )
            self._log.info(
                "huggingface 逐文件下载 %s: %d 个文件", source_spec.repo_id, len(selected)
            )
            hf_hub_download = getattr(hub, "hf_hub_download")
            for filename in selected:
                hf_hub_download(
                    repo_id=source_spec.repo_id,
                    filename=filename,
                    revision=source_spec.revision,
                    local_dir=str(dest),
                    token=token,
                    endpoint=endpoint,
                )

    @staticmethod
    def _select_files(
        files: Sequence[str], source_spec: SourceSpec, spec: ModelSpec
    ) -> list[str]:
        """按 allow / ignore 规则过滤仓库文件列表；目录项与元数据文件直接剔除。"""
        allow = _effective_allow_patterns(spec, source_spec)
        ignore = _effective_ignore_patterns(spec, source_spec)
        selected: list[str] = []
        for rel in files:
            if not rel or rel.endswith("/"):
                continue
            if rel.startswith(".git") or rel == ".gitattributes":
                continue
            if allow and not any(fnmatch.fnmatch(rel, pattern) for pattern in allow):
                continue
            if ignore and any(fnmatch.fnmatch(rel, pattern) for pattern in ignore):
                continue
            selected.append(rel)
        return selected

    @staticmethod
    @contextlib.contextmanager
    def _hf_progress_suppressed() -> Iterator[None]:
        """关掉 hf 自带进度条，避免与我们的字节级进度条互相刷屏。"""
        enabled: bool | None = None
        try:
            from huggingface_hub.utils import (  # noqa: PLC0415
                are_progress_bars_disabled,
                disable_progress_bars,
                enable_progress_bars,
            )

            enabled = not are_progress_bars_disabled()
            if enabled:
                disable_progress_bars()
        except Exception:  # pragma: no cover - 版本差异/未安装
            yield
            return
        try:
            yield
        finally:
            if enabled:
                with contextlib.suppress(Exception):
                    enable_progress_bars()

    @staticmethod
    def _import_backend(modules: Sequence[str], *, hint: str, install_hint: str) -> ModuleType:
        last: Exception | None = None
        for module_name in modules:
            try:
                return importlib.import_module(module_name)
            except ImportError as exc:
                last = exc
        raise MissingDependencyError(f"{hint} 不可用，请执行: {install_hint}") from last

    # ------------------------------------------------------------------ #
    # 内部: 进度与扫描
    # ------------------------------------------------------------------ #
    @contextlib.contextmanager
    def _byte_progress(
        self, desc: str, total_hint: int | None, watch_dir: Path, *, show: bool = True
    ) -> Iterator[ProgressHandle]:
        """字节级进度条：用后台线程轮询目录大小，不依赖下载库的回调。

        这样 modelscope / huggingface / 未来新增的源共用同一套进度展示，
        且进度反映的是"真实落盘字节"，中断续传时也不会倒退。
        """
        if not show or _tqdm is None:
            yield _NullBar()
            return

        interval = self._cfg.network.progress_interval
        bar: Any = _tqdm(
            total=total_hint or None,
            desc=desc,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
            leave=True,
            mininterval=interval,
        )
        stop = threading.Event()
        state = {"last": -1}

        def _sync() -> None:
            size = _dir_size(watch_dir, include_metadata=True)
            if size == state["last"]:
                return
            state["last"] = size
            try:
                if bar.total is None or size > bar.total:
                    bar.total = size
                bar.n = size
                bar.refresh()
            except Exception:  # pragma: no cover - 终端关闭等
                pass

        def _watch() -> None:
            while not stop.wait(interval):
                _sync()

        thread = threading.Thread(target=_watch, name=f"zhiying-progress-{desc}", daemon=True)
        thread.start()
        try:
            yield bar
        finally:
            stop.set()
            thread.join(timeout=5.0)
            _sync()
            with contextlib.suppress(Exception):
                bar.close()

    def _scan_files(self, root: Path, spec: ModelSpec) -> list[dict[str, Any]]:
        """枚举权重目录内的文件（跳过库自身的元数据缓存目录）。"""
        entries: list[dict[str, Any]] = []
        try:
            candidates = sorted(root.rglob("*"))
        except OSError as exc:  # pragma: no cover
            self._log.warning("扫描目录失败 %s: %s", root, exc)
            return entries
        for path in candidates:
            if not path.is_file():
                continue
            rel_parts = path.relative_to(root).parts
            if any(part in _IGNORED_DIR_NAMES for part in rel_parts[:-1]):
                continue
            if rel_parts[-1] == self._cfg.cache.manifest_name:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            entries.append({"path": Path(*rel_parts).as_posix(), "size": size})
        return entries

    # ------------------------------------------------------------------ #
    # 内部: 落盘与 manifest
    # ------------------------------------------------------------------ #
    def _finalize(
        self, spec: ModelSpec, source_name: str, source_spec: SourceSpec, partial: Path
    ) -> None:
        """校验 -> 写 manifest -> 原子换入最终目录。"""
        missing = self._missing_files(spec, partial)
        if missing:
            raise ModelIntegrityError(spec.name, f"下载完成后仍缺少关键文件: {', '.join(missing)}")

        files = self._scan_files(partial, spec)
        if not files:
            raise ModelIntegrityError(spec.name, "下载目录为空")
        total = sum(int(item["size"]) for item in files)
        self._write_manifest(partial, spec, source_name, source_spec, files, total)

        dest = self._cfg.cache.root / spec.directory_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        trash: Path | None = None
        if dest.exists():
            trash = dest.with_name(f"{dest.name}.old-{int(time.time() * 1000)}")
            try:
                os.replace(dest, trash)
            except OSError as exc:
                _rmtree(partial)
                raise ModelIntegrityError(spec.name, f"无法替换旧模型目录: {exc}") from exc
        try:
            os.replace(partial, dest)
        except OSError:
            if trash is not None:
                with contextlib.suppress(OSError):
                    os.replace(trash, dest)
            raise
        if trash is not None:
            _rmtree(trash)
        self._log.debug("模型已就位: %s (%.1f MB)", dest, total / 1048576)

    def _write_manifest(
        self,
        root: Path,
        spec: ModelSpec,
        source_name: str,
        source_spec: SourceSpec,
        files: Sequence[Mapping[str, Any]],
        total: int,
    ) -> None:
        payload = {
            "schema": MANIFEST_SCHEMA,
            "name": spec.name,
            "kind": spec.kind,
            "license": spec.license,
            "source": source_name,
            "repo_id": source_spec.repo_id,
            "revision": source_spec.revision,
            "downloaded_at": _utcnow_iso(),
            "manager_version": __version__,
            "size_bytes": total,
            "file_count": len(files),
            "files": [dict(item) for item in files],
            "sha256": dict(spec.sha256) or None,
        }
        manifest = root / self._cfg.cache.manifest_name
        tmp = manifest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, manifest)

    def _read_manifest(self, root: Path) -> dict[str, Any] | None:
        manifest = root / self._cfg.cache.manifest_name
        if not manifest.is_file():
            return None
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._log.debug("manifest 读取失败 %s: %s", manifest, exc)
            return None
        return data if isinstance(data, dict) else None

    # ------------------------------------------------------------------ #
    # 展示辅助
    # ------------------------------------------------------------------ #
    def describe(self, name: str) -> dict[str, Any]:
        """给 CLI / Web 用的单模型摘要（不含密钥、不含内部绝对路径以外的敏感信息）。"""
        spec = self.get_spec(name)
        status = self._inspect(spec)
        return {
            "name": spec.name,
            "kind": spec.kind,
            "required": spec.required,
            "license": spec.license,
            "description": spec.description,
            "state": status.state.value,
            "path": str(status.path),
            "size_bytes": status.size_bytes,
            "size_gb": round(status.size_bytes / (1024**3), 2),
            "file_count": status.file_count,
            "source": status.source,
            "revision": status.revision,
            "sources": {
                key: {"repo_id": value.repo_id, "revision": value.revision}
                for key, value in spec.sources.items()
            },
            "message": status.message,
        }


# --------------------------------------------------------------------------- #
# 进程级单例
# --------------------------------------------------------------------------- #
_SINGLETON: ModelManager | None = None
_SINGLETON_LOCK = threading.Lock()


def get_model_manager(
    config_path: str | Path | None = None,
    **kwargs: Any,
) -> ModelManager:
    """获取进程级 ``ModelManager`` 单例。

    传入 ``config_path`` 或任意 kwargs 时强制新建实例（便于测试与多配置场景），
    否则复用首个实例——模型目录、锁与下载状态在全进程内共享，避免重复下载。
    """
    global _SINGLETON
    if config_path is not None or kwargs:
        return ModelManager(config_path, **kwargs)
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = ModelManager()
        return _SINGLETON


def reset_model_manager() -> None:
    """丢弃单例（测试用；不删除任何已下载文件）。"""
    global _SINGLETON
    with _SINGLETON_LOCK:
        _SINGLETON = None
