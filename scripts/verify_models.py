"""配置完成前真实加载三个必需模型；不接受降级结果。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import load_app_config
from backend.pipeline.factory import build_orchestrator


def main():
    orch = build_orchestrator(load_app_config())
    orch.close()
    print("三个必需本地模型已真实加载，云端配置完整。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
