"""真实验收：项目内环境、独立 GGUF 导入/启停/删除、免登录报告 HTTP。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from videolens.api import create_app
from videolens.bootstrap import MODEL_SPECS, read_json
from videolens.config import atomic_json, load_settings
from videolens.service import VideoService


def main():
    root = Path(__file__).resolve().parents[1]
    settings = load_settings().model_copy(deep=True)
    original = read_json(Path(settings.models_dir) / "model-manifest.json")["models"]["local"]
    source = Path(original["snapshot"]) / MODEL_SPECS["local"]["filename"]
    base = root / "data/verification/simple-manager"
    settings.data_dir = str(base / "data")
    settings.models_dir = str(base / "models")
    settings.runtime_dir = str(base / "runtime")
    settings.config_path = str(base / "config.json")
    settings.local_base_url = "http://127.0.0.1:8083/v1"
    settings.local_enabled = True
    settings.cloud_enabled = False
    atomic_json(Path(settings.config_path), settings.model_dump())
    service = VideoService(settings)
    report = {"python": sys.executable, "cases": []}
    assert Path(sys.prefix).resolve().is_relative_to(root / "envs")
    with TestClient(create_app(settings, service), headers={"Authorization": "Bearer " + settings.api_token}) as client:
        assert client.get("/app/").status_code == 200
        assert not client.get("/app/config").json().get("auth_required")
        submitted = client.post("/api/v1/local-models/import", json={"path": str(source)})
        assert submitted.status_code == 202, submitted.text
        manager = client.app.state.local_models
        manager.thread.join(180)
        job = manager.jobs[submitted.json()["id"]]
        assert job["status"] == "succeeded", job
        identity = job["model_id"]
        report["cases"].append({"name": "real_gguf_import", "status": "passed", "bytes": job["bytes"]})
        started = client.post(f"/api/v1/local-models/{identity}/start", json={"gpu_layers": settings.llama_gpu_layers, "context": 8192})
        assert started.is_success, started.text
        # 真实请求 llama.cpp，确认不只是进程存在或健康接口可访问。
        response = service.gateway.generate([{"role": "user", "content": "只回答：知影模型已就绪。"}])
        assert response.strip(), "真实模型未返回正文"
        report["cases"].append({"name": "real_llama_generation", "status": "passed", "response": response})
        assert client.delete(f"/api/v1/local-models/{identity}").status_code == 409
        assert client.post("/api/v1/local-models/stop").is_success
        assert client.delete(f"/api/v1/local-models/{identity}").is_success
        assert source.is_file()
        report["cases"].append({"name": "running_guard_stop_delete_original_preserved", "status": "passed"})
    # 新环境真实流水线的报告通过同一新版免登录下载路由验收。
    reading = load_settings().model_copy(deep=True)
    reading.data_dir = str(root / "data/verification/simplified-data")
    service = VideoService(reading)
    with TestClient(create_app(reading, service)) as client:
        video = next(v for v in service.list_videos() if service.get_video(v["id"]).get("report"))
        exported = client.get(f"/app/videolens-export/{video['id']}/markdown")
        assert exported.is_success and "attachment" in exported.headers["content-disposition"]
        assert (Path(reading.data_dir) / "exports" / (video["id"] + ".md")).read_text(encoding="utf-8") == exported.text
        report["cases"].append({"name": "anonymous_markdown_export", "status": "passed", "characters": len(exported.text)})
    log = base / "logs/llama-server.log"
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines and all(re.match(r"\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\]\[(INFO|WARNING|ERROR)\]\[llama.cpp\]:", line) for line in lines)
    report["cases"].append({"name": "real_llama_log_format", "status": "passed", "lines": len(lines)})
    report["status"] = "passed"
    atomic_json(root / "data/verification/simple-manager.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
