"""部署辅助工具的离线回归测试；不安装包、不访问网络、不终止任何真实进程。

运行：conda run -n VideoLens python -m pytest -q scripts/test_bootstrap.py
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from videolens.bootstrap import (
    extract_checked, managed_process, sha256_file, stop_local, update_config, verify_models,
)


def settings_for(folder: Path):
    return SimpleNamespace(config_path=str(folder / "config.json"), runtime_dir=str(folder / "runtime"),
                           models_dir=str(folder / "models"))


def test_streamed_hash_handles_binary(tmp_path):
    path = tmp_path / "中文模型.bin"
    content = b"\x00\xff\x80VideoLens" * 10000
    path.write_bytes(content)
    assert sha256_file(path) == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("entry", ["../outside.txt", "C:/outside.txt", "..\\outside.txt"])
def test_zip_does_not_escape_destination(tmp_path, entry):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr(entry, "unexpected")
    with pytest.raises(RuntimeError, match="不安全"):
        extract_checked(archive, tmp_path / "target")
    assert not (tmp_path / "outside.txt").exists()


def test_zip_preserves_regular_nested_files(tmp_path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("bin/中文说明.txt", "正常".encode("utf-8"))
    extract_checked(archive, tmp_path / "target")
    assert (tmp_path / "target/bin/中文说明.txt").read_text(encoding="utf-8") == "正常"


def test_config_updates_merge_under_competing_writers(tmp_path):
    settings = settings_for(tmp_path)
    Path(settings.config_path).write_text('{"user_setting": "保留"}', encoding="utf-8")
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(update_config, settings, **{f"field_{i}": i}) for i in range(20)]
        for future in futures:
            future.result()
    data = json.loads(Path(settings.config_path).read_text(encoding="utf-8"))
    assert data["user_setting"] == "保留"
    assert all(data[f"field_{i}"] == i for i in range(20))


def test_wrong_process_identity_is_never_adopted(tmp_path):
    current = psutil.Process(os.getpid())
    record = {"pid": os.getpid(), "create_time": current.create_time(),
              "executable": str(tmp_path / "different.exe"), "model": "unrelated.gguf"}
    assert managed_process(record) is None


def test_stop_does_not_delete_another_owner_record(tmp_path):
    settings = settings_for(tmp_path)
    path = Path(settings.runtime_dir) / "llama-process.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"owner":"another-startup"}', encoding="utf-8")
    assert stop_local(settings, owner="this-startup")["status"] == "not_owned"
    assert path.exists()


def test_verification_detects_changed_model(tmp_path):
    settings = settings_for(tmp_path)
    model = Path(settings.models_dir) / "snapshot" / "model.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"valid")
    manifest = {"models": {"asr": {"snapshot": str(model.parent), "revision": "fixed",
                "files": [{"path": "model.bin", "bytes": 5, "sha256": sha256_file(model)}]}}}
    (Path(settings.models_dir) / "model-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_models(settings, ["asr"])["asr"]["status"] == "verified"
    model.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="哈希不匹配"):
        verify_models(settings, ["asr"])
