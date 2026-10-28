"""启动预检；必需项缺失立即判定 BROKEN，不输出可降级结论。"""

import argparse, importlib.util, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import load_app_config
from backend.models.manager import ModelManager
from backend.models.required import REQUIRED_MODELS


def main(argv=None):
    p = argparse.ArgumentParser(description="知影必需环境检查")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    checks = []

    def check(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    check("Python", sys.version_info[:2] == (3, 13), sys.version.split()[0])
    for module in (
        "fastapi",
        "uvicorn",
        "yaml",
        "torch",
        "transformers",
        "sentencepiece",
        "faster_whisper",
        "av",
        "chromadb",
        "yt_dlp",
        "cv2",
    ):
        check(module, importlib.util.find_spec(module) is not None)
    try:
        cfg = load_app_config()
        check("配置", True)
        from backend.media.ffmpeg_utils import resolve_binary

        for name in ("ffmpeg", "ffprobe"):
            try:
                check(name, True, resolve_binary(name))
            except Exception as exc:
                check(name, False, str(exc))
        manager = ModelManager(
            config_path=cfg.root_dir / cfg.models.config_file,
            cache_root=cfg.path("models"),
            allow_download=False,
        )
        for name in REQUIRED_MODELS:
            status = manager.status(name)
            check(name, status.available, status.state.value)
        from backend.models.llm import LlmGateway
        from backend.models.vl import VlService

        LlmGateway.from_config(cfg).close()
        VlService.from_config(cfg).close()
        check("云端配置", True, "尚未验证网络调用")
    except Exception as exc:
        check("配置或模型", False, str(exc))
    result = {
        "verdict": "READY" if all(c["ok"] for c in checks) else "BROKEN",
        "checks": checks,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for c in checks:
            print(f"{'通过' if c['ok'] else '失败'} {c['name']} {c['detail']}")
        print(result["verdict"])
    return 0 if result["verdict"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
