"""配置、路径与后端凭据管理。

普通设置可以保存为 JSON；API Key 只能存系统凭据库或环境变量。
配置加载不会下载模型，避免启动程序时产生难以察觉的网络请求。
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import VideoLensError


# 生成接口除了正文还包含角色、消息边界等控制标记；保留固定安全空间。
GENERATION_SAFETY_TOKENS = 512
MIN_GENERATION_INPUT_TOKENS = 256


class Provider(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    enabled: bool = False
    name: str = "自定义服务"
    base_url: str = ""
    model: str = ""
    key_ref: str = ""
    context_tokens: int = Field(default=8192, ge=1024, le=2_000_000)
    timeout_seconds: float = Field(default=60, ge=1, le=1800)


class Settings(BaseModel):
    """单位直接写在字段名中，避免把秒当毫秒、把字节当 MB。"""

    # 默认的 20 和反序列化后的 20.0 应具有相同指纹，统一按字段类型校验默认值。
    model_config = ConfigDict(extra="forbid", validate_default=True)
    config_path: str = ""
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1024, le=65535)
    data_dir: str = "data"
    models_dir: str = "models"
    runtime_dir: str = "runtime"
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    cloud_enabled: bool = False
    allow_model_download: bool = False
    allow_video_download: bool = True
    text_provider: Provider = Field(default_factory=Provider)
    vision_provider: Provider = Field(default_factory=Provider)
    local_enabled: bool = True
    local_base_url: str = "http://127.0.0.1:8081/v1"
    local_model: str = "Qwen3-4B-Q4_K_M"
    llama_server_path: str = ""
    llama_model_path: str = ""
    llama_gpu_layers: int = Field(default=0, ge=0, le=999)
    local_context_tokens: int = Field(default=8192, ge=1024, le=131072)
    asr_model: str = "Systran/faster-whisper-small"
    text_embedding_model: str = "BAAI/bge-m3"
    image_embedding_model: str = "google/siglip2-base-patch16-224"
    max_video_seconds: int = Field(default=7200, ge=1)
    max_video_bytes: int = Field(default=4_000_000_000, ge=1)
    subtitle_language: str = "zh"
    clip_target_seconds: float = Field(default=20, ge=5, le=60)
    clip_min_seconds: float = Field(default=10, ge=5, le=60)
    clip_max_seconds: float = Field(default=30, ge=5, le=60)
    frames_per_clip: int = Field(default=3, ge=1, le=5)
    frame_max_bytes: int = Field(default=200_000, ge=10_000)
    scene_threshold: float = Field(default=0.3, ge=0, le=1)
    analysis_concurrency: int = Field(default=1, ge=1, le=4)
    candidate_count: int = Field(default=20, ge=1, le=200)
    retrieval_top_k: int = Field(default=8, ge=1, le=50)
    rrf_constant: int = Field(default=60, ge=1)
    text_weight: float = Field(default=0.5, ge=0)
    image_weight: float = Field(default=0.3, ge=0)
    keyword_weight: float = Field(default=0.2, ge=0)
    chapter_seconds: int = Field(default=300, ge=30, le=3600)
    output_tokens: int = Field(default=1536, ge=128, le=8192)
    recent_messages: int = Field(default=8, ge=2, le=50)
    managed_media_days: float = Field(default=7, ge=0)
    failed_temp_hours: float = Field(default=24, ge=0)
    result_ttl_days: float = Field(default=0, ge=0)
    data_quota_gb: float = Field(default=50, ge=0.1)
    rate_limit_per_minute: int = Field(default=60, ge=1)
    summary_prompt: str = "请概括视频的核心内容、重要结论和时间线，每条结论都引用原始证据。"
    answer_prompt: str = "请仅依据提供的视频证据回答；信息不足时说明，不编造事实。"
    api_token: str = Field(default="", exclude=True)
    ui_username: str = "videolens"
    ui_password: str = Field(default="", exclude=True)

    @model_validator(mode="after")
    def validate_ranges(self):
        if not self.clip_min_seconds <= self.clip_target_seconds <= self.clip_max_seconds:
            raise ValueError("切片最小值、目标值、最大值必须按顺序递增")
        if self.text_weight + self.image_weight + self.keyword_weight <= 0:
            raise ValueError("检索权重不能全部为零")
        contexts = []
        if self.local_enabled:
            contexts.append(("本地生成模型", self.local_context_tokens))
        if self.cloud_enabled:
            if self.text_provider.enabled:
                contexts.append(("云端文本模型", self.text_provider.context_tokens))
            if self.vision_provider.enabled:
                contexts.append(("云端视觉模型", self.vision_provider.context_tokens))
        required = self.output_tokens + GENERATION_SAFETY_TOKENS + MIN_GENERATION_INPUT_TOKENS
        for name, capacity in contexts:
            if required > capacity:
                raise ValueError(f"{name}上下文过小：至少需要输出预算 {self.output_tokens} + 安全空间 {GENERATION_SAFETY_TOKENS} + 最小输入空间 {MIN_GENERATION_INPUT_TOKENS} = {required} tokens")
        # 首版本地服务不提供公网部署；远程接入需在后续显式配置 HTTPS。
        if self.host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("首版服务仅支持本机回环地址")
        return self

    def fingerprint(self) -> str:
        """仅对影响分析产物的设置取指纹，令牌、端口不会使分析缓存失效。"""
        data = self.model_dump(exclude={"config_path", "host", "port", "ui_username"})
        data["model_versions"] = self.model_versions()
        return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def model_versions(self) -> dict[str, str]:
        """同名模型升级权重也必须重建产物；快照目录的位置不代表模型版本。

        使用安装清单中的 revision 和逐文件校验值识别已安装权重，避免每次查询
        都重新读取数 GB 模型。文件完整性由部署校验负责，版本兼容性由此处负责。
        """
        path = Path(self.models_dir) / "model-manifest.json"
        if not path.is_file():
            return {}
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))["models"]
            result = {}
            for role, item in entries.items():
                identity = {"repo_id": item.get("repo_id"), "revision": item.get("revision"),
                            "files": sorted((file["path"], file["sha256"]) for file in item.get("files", []))}
                result[role] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            return result
        except (OSError, ValueError, KeyError, TypeError):
            return {"manifest": "invalid"}


def project_root() -> Path:
    return Path(os.environ.get("VIDEOLENS_ROOT", Path(__file__).resolve().parents[2]))


def atomic_json(path: Path, data: object) -> None:
    """先写临时文件再替换，避免断电留下半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_settings(config_path: str | Path | None = None) -> Settings:
    root = project_root()
    path = Path(config_path or os.environ.get("VIDEOLENS_CONFIG", root / "config.local.json")).resolve()
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data["config_path"] = str(path)
    for key in ("data_dir", "models_dir", "runtime_dir"):
        value = Path(data.get(key, key.removesuffix("_dir")))
        data[key] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
    settings = Settings(**data)
    settings.api_token = os.environ.get("VIDEOLENS_API_TOKEN", "")
    if not settings.api_token:
        token_file = Path(settings.data_dir) / ".access-token"
        token_file.parent.mkdir(parents=True, exist_ok=True)
        if not token_file.exists():
            # 独占创建保证两个首次启动的进程不会互相覆盖令牌。
            try:
                with token_file.open("x", encoding="utf-8") as stream:
                    stream.write(secrets.token_urlsafe(32))
                if os.name != "nt":
                    token_file.chmod(0o600)
            except FileExistsError:
                pass
        settings.api_token = token_file.read_text(encoding="utf-8").strip()
    settings.ui_password = settings.api_token
    return settings


class SecretStore:
    """系统凭据库不可用时拒绝明文落盘，用户仍可通过环境变量注入密钥。"""

    @staticmethod
    def get(ref: str) -> str | None:
        if not ref:
            return None
        key = "VIDEOLENS_KEY_" + "".join(c if c.isalnum() else "_" for c in ref).upper()
        if os.environ.get(key):
            return os.environ[key]
        try:
            import keyring

            return keyring.get_password("VideoLens", ref)
        except Exception:
            return None

    @staticmethod
    def set(ref: str, value: str) -> None:
        if not ref or len(ref) > 100 or not value:
            raise VideoLensError("INVALID_KEY", "凭据名称与密钥不能为空")
        try:
            import keyring

            keyring.set_password("VideoLens", ref, value)
        except Exception as exc:
            raise VideoLensError("KEYSTORE_UNAVAILABLE", "系统凭据库不可用，请使用后端环境变量配置密钥", 503) from exc
