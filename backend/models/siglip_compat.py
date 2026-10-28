"""本地 SigLIP 系列配置兼容：序列化完整配置，避免构造无效的上游默认配置。"""

from __future__ import annotations

import json
from pathlib import Path


def load_siglip_config(model_dir):
    path = Path(model_dir)
    model_type = json.loads((path / "config.json").read_text(encoding="utf-8")).get("model_type")
    if model_type == "siglip":
        from transformers import SiglipConfig, SiglipModel, SiglipProcessor

        base, model_class, processor_class = SiglipConfig, SiglipModel, SiglipProcessor
    elif model_type == "siglip2":
        from transformers import Siglip2Config, Siglip2Model, Siglip2Processor

        base, model_class, processor_class = Siglip2Config, Siglip2Model, Siglip2Processor
    else:
        raise ValueError(f"SigLIP 权重的 model_type 不受支持: {model_type!r}")

    class LocalConfig(base):
        def to_diff_dict(self):
            # 上游 recursive_diff_dict 会用词表 32000 构造 49406/49407 的默认配置。
            # 仅调整该实例的配置序列化，不修改权重、实际 token 参数或全局日志。
            return self.to_dict()

    config = LocalConfig.from_pretrained(str(path), local_files_only=True)
    return config, model_class, processor_class
