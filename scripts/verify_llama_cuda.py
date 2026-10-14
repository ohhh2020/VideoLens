"""隔离验证 llama.cpp CUDA：真实生成通过且日志确认层卸载后才算 GPU 成功。

使用独立配置、运行目录和端口，不改变主配置；失败时保留已验证的 CPU 基线。
本脚本会下载固定版本的官方 CUDA 发行包，但不安装或修改显卡驱动。
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from videolens.bootstrap import install_runtime, start_local, stop_local
from videolens.config import atomic_json, load_settings


def main() -> int:
    import httpx

    settings = load_settings().model_copy(deep=True)
    folder = Path("data/verification/llama-cuda").resolve()
    settings.config_path = str(folder / "config.json")
    settings.runtime_dir = str(folder / "runtime")
    settings.local_base_url = "http://127.0.0.1:8082/v1"
    settings.llama_gpu_layers = 99
    atomic_json(Path(settings.config_path), settings.model_dump(exclude={"config_path"}))
    report = {"kind": "real_llama_cuda", "requested_gpu_layers": 99, "status": "failed"}
    started = time.monotonic()
    state = {}
    try:
        runtime = install_runtime(settings, "cuda", None, None, None)
        settings = load_settings(settings.config_path)
        report["runtime"] = {"release": runtime["release"], "variant": runtime["variant"]}
        # b11612 的默认日志只输出服务状态；级别 4 才包含模型设备和层卸载信息。
        # 仅此验收子进程提高日志级别，不开启会输出完整请求的无限调试日志。
        os.environ["LLAMA_ARG_LOG_VERBOSITY"] = "4"
        state = start_local(settings, owner="verify-cuda", wait_seconds=240)
        inference = time.monotonic()
        with httpx.Client(timeout=180, trust_env=False) as client:
            response = client.post(settings.local_base_url + "/chat/completions", json={
                "model": settings.local_model, "messages": [{"role": "user", "content": "请只回答中文词语：测试成功"}],
                "max_tokens": 48, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"].get("content", "")
        log = (Path(settings.runtime_dir) / "llama-server.log").read_text(encoding="utf-8", errors="replace")
        layers = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log)
        assert answer.strip(), "GPU 服务没有返回可见生成内容"
        assert layers and int(layers[-1][0]) > 0, "日志未确认真实 GPU 层卸载，不能用 CPU 回退冒充 CUDA 成功"
        report.update(status="passed", answer=answer, offloaded_layers=int(layers[-1][0]),
                      total_layers=int(layers[-1][1]), generation_seconds=round(time.monotonic() - inference, 3))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if state.get("started"):
            stop_local(settings, owner="verify-cuda")
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        atomic_json(Path("data/verification/llama-cuda.json"), report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
