"""模型与 llama.cpp 部署工具。

默认使用固定版本下载地址和 SHA-256，安装后生成可复查的清单。
此模块不会在 import 时下载文件或启动进程，只有用户执行子命令才有副作用。
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from .config import Settings, atomic_json, load_settings

if TYPE_CHECKING:
    import psutil


# 版本号来自上游发布信息。模型 revision 是不可变的提交号，不使用会改变的 main。
MODEL_SPECS = {
    "asr": {"repo_id": "Systran/faster-whisper-small",
            "revision": "536b0662742c02347bc0e980a01041f333bce120",
            "patterns": ["*.json", "model.bin", "vocabulary.*", "README.md", "LICENSE*"]},
    "text": {"repo_id": "BAAI/bge-m3", "revision": "5617a9f61b028005a4858fdac845db406aefb181",
             "patterns": ["*.json", "*.safetensors", "pytorch_model.bin", "tokenizer.*", "sentencepiece.bpe.model",
                          "1_Pooling/*", "README.md", "LICENSE*"]},
    "image": {"repo_id": "google/siglip2-base-patch16-224",
              "revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
              "patterns": ["*.json", "*.safetensors", "*.model", "README.md", "LICENSE*"]},
    "local": {"repo_id": "Qwen/Qwen3-4B-GGUF", "revision": "bc640142c66e1fdd12af0bd68f40445458f3869b",
              "patterns": ["Qwen3-4B-Q4_K_M.gguf", "README.md", "LICENSE*"],
              "filename": "Qwen3-4B-Q4_K_M.gguf",
              "sha256": "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"},
}
LLAMA_RELEASE = "b11612"
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-8.1.2-essentials_build.zip"
FFMPEG_SHA256 = "db580001caa24ac104c8cb856cd113a87b0a443f7bdf47d8c12b1d740584a2ec"
RUNTIME_PACKAGES = {
    "cpu": [
        ("llama-b11612-bin-win-cpu-x64.zip", "b8456ccb1be98365a5a2227efa5c495508a6a3ebce881c6189a1dbeaaa170b69"),
    ],
    "cuda": [
        ("llama-b11612-bin-win-cuda-12.4-x64.zip", "4c6d8afd23475f6eded74c4a3019a6746aab83183a3158f3fc4de1cb7c4f5f12"),
        ("cudart-llama-bin-win-cuda-12.4-x64.zip", "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
    ],
}


def sha256_file(path: Path) -> str:
    """按 8MiB 块计算校验值，2.5GB 模型也不会一次占用几 GB 内存。"""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path, default: dict | None = None) -> dict:
    """不存在时使用默认值；损坏的清单不静默覆盖，以免误认为部署已通过。"""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else (default or {})


def update_config(settings: Settings, **values) -> None:
    """只修改部署得到的路径，保留用户已经设置的供应商和其他选项。"""
    from filelock import FileLock

    path = Path(settings.config_path)
    lock = Path(settings.runtime_dir) / "config-write.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    # 两个部署进程可能同时完成模型/运行时下载；先锁定再读取，避免后写者覆盖前者路径。
    with FileLock(str(lock), timeout=60):
        data = read_json(path)
        data.update(values)
        atomic_json(path, data)


def download_models(settings: Settings, names: list[str]) -> dict:
    """下载选定角色，核对上游 LFS 哈希，并逐角色原子保存本地清单。"""
    from huggingface_hub import HfApi, snapshot_download

    cache = Path(settings.models_dir)
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "model-manifest.json"
    manifest = read_json(manifest_path, {"schema": 1, "models": {}})
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    for name in names:
        spec = MODEL_SPECS[name]
        print(f"下载模型 [{name}] {spec['repo_id']} @ {spec['revision']}\n缓存目录：{cache}", flush=True)
        # HfApi 返回每个大文件的官方 LFS 摘要，避免只计算本地哈希而没有对照来源。
        info = HfApi().model_info(spec["repo_id"], revision=spec["revision"], files_metadata=True)
        hashes = {}
        for sibling in info.siblings or []:
            lfs = sibling.lfs
            if lfs:
                hashes[sibling.rfilename] = getattr(lfs, "sha256", None) or (lfs.get("sha256") if isinstance(lfs, dict) else None)
        selected_bytes = sum((getattr(item, "size", 0) or 0) for item in info.siblings or []
                             if any(fnmatch.fnmatch(item.rfilename, p) for p in spec["patterns"]))
        print(f"选定文件约 {selected_bytes / 1024 ** 3:.2f} GiB；已缓存文件自动复用。", flush=True)
        snapshot = Path(snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"],
                                         cache_dir=str(cache), allow_patterns=spec["patterns"], max_workers=4))
        files = []
        for path in sorted(snapshot.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(snapshot).as_posix()
            digest = sha256_file(path)
            expected = hashes.get(relative)
            if name == "local" and relative == spec["filename"]:
                expected = spec["sha256"]
            if expected and digest != expected:
                raise RuntimeError(f"模型校验失败：{relative}，请清理该缓存文件后重试。")
            files.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size,
                          "upstream_sha256": expected})
        if not any(item["path"].endswith((".bin", ".safetensors", ".gguf")) for item in files):
            raise RuntimeError(f"模型 {name} 没有下载到权重文件；不能把配置文件下载完成当成模型安装成功。")
        entry = {"repo_id": spec["repo_id"], "revision": spec["revision"],
                 "snapshot": str(snapshot.resolve()), "files": files}
        if name == "local":
            model_file = snapshot / spec["filename"]
            if not model_file.is_file():
                raise RuntimeError("GGUF 模型缺失，下载未完成。")
            entry["model_file"] = str(model_file.resolve())
            update_config(settings, llama_model_path=entry["model_file"])
        # ASR 与文本权重可以由不同部署进程并行下载，保存时重读并合并当前角色。
        # 如果仅在函数开始读取一次，最后完成的进程会覆盖其他进程的新角色记录。
        from filelock import FileLock
        with FileLock(str(cache / ".manifest.lock"), timeout=60):
            manifest = read_json(manifest_path, {"schema": 1, "models": {}})
            manifest.setdefault("models", {})[name] = entry
            atomic_json(manifest_path, manifest)
        print(f"[{name}] 下载与校验完成，共 {len(files)} 个文件。", flush=True)
    return manifest


def verify_models(settings: Settings, names: list[str]) -> dict:
    """离线逐文件核对清单。此验证证明文件完整性，不代表推理质量测试。"""
    manifest = read_json(Path(settings.models_dir) / "model-manifest.json")
    results = {}
    for name in names:
        entry = manifest.get("models", {}).get(name)
        if not entry or not entry.get("files"):
            raise RuntimeError(f"模型 {name} 未安装或无校验清单。请执行 models --names {name}。")
        for file in entry["files"]:
            path = Path(entry["snapshot"]) / file["path"]
            if not path.is_file() or path.stat().st_size != file["bytes"] or sha256_file(path) != file["sha256"]:
                raise RuntimeError(f"模型文件缺失或哈希不匹配：{name}/{file['path']}")
        results[name] = {"status": "verified", "revision": entry["revision"], "files": len(entry["files"])}
    return results


def download_file(url: str, target: Path, expected_sha256: str) -> None:
    """断点续传下载；服务器忽略 Range 时从头重写，成功校验后才替换正式文件。"""
    import httpx

    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256.lower()):
        raise ValueError("必须提供 64 位 SHA-256 校验值。")
    if target.exists() and sha256_file(target) == expected_sha256.lower():
        print(f"复用已校验文件：{target.name}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    for attempt in range(3):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                              timeout=httpx.Timeout(60, connect=30)) as response:
                if response.status_code == 416:
                    temporary.unlink(missing_ok=True)
                    continue
                response.raise_for_status()
                resumed = response.status_code == 206 and offset > 0
                if resumed and not response.headers.get("content-range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("服务器返回了错误的续传位置。")
                written = offset if resumed else 0
                total = int(response.headers.get("content-length", 0)) + written
                last_print = 0.0
                with temporary.open("ab" if resumed else "wb") as output:
                    for chunk in response.iter_bytes(1024 * 1024):
                        output.write(chunk)
                        written += len(chunk)
                        if time.monotonic() - last_print >= 3:
                            print(f"{target.name}: {written / 1024 ** 2:.1f}/{total / 1024 ** 2:.1f} MiB", flush=True)
                            last_print = time.monotonic()
            actual = sha256_file(temporary)
            if actual != expected_sha256.lower():
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"下载 SHA-256 不匹配：{target.name}")
            temporary.replace(target)
            return
        except (httpx.HTTPError, OSError, RuntimeError) as exc:
            if attempt == 2:
                raise RuntimeError(f"下载失败，已尝试 3 次：{target.name}。检查代理或使用 --archive 导入本地包。") from exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"服务器拒绝续传：{target.name}，请重新执行。")


def extract_checked(archive: Path, destination: Path) -> None:
    """防止 ZIP 内的 ../、绝对路径或软链接把文件写出目标目录。"""
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            normalized = member.filename.replace("\\", "/")
            path = (base / normalized).resolve()
            if not path.is_relative_to(base) or ":" in normalized or ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise RuntimeError(f"运行时 ZIP 包含不安全的路径：{member.filename}")
        package.extractall(base)


def install_runtime(settings: Settings, variant: str, url: str | None, sha256: str | None, archive: str | None) -> dict:
    """CPU 发行包是默认基线；CUDA 包可选，只有真实推理通过后才应启用 GPU。"""
    if os.name != "nt" and not (url or archive):
        raise RuntimeError("自动发行包面向 Windows x64；其他系统请配置本机编译的 llama_server_path。")
    runtime = Path(settings.runtime_dir)
    destination = runtime / f"llama-{variant}"
    packages = []
    if url or archive:
        if not sha256:
            raise ValueError("自定义 URL/本地运行时包必须同时提供 --sha256。")
        if archive:
            package_path = Path(archive).resolve()
            if sha256_file(package_path) != sha256.lower():
                raise RuntimeError("本地运行时包 SHA-256 不匹配。")
        else:
            package_path = runtime / "downloads" / "llama-custom.zip"
            download_file(url, package_path, sha256)
        packages.append({"path": str(package_path), "sha256": sha256.lower(), "source": url or "local"})
    else:
        for filename, digest in RUNTIME_PACKAGES[variant]:
            source = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE}/{filename}"
            package_path = runtime / "downloads" / filename
            print(f"下载 llama.cpp {LLAMA_RELEASE}：{source}", flush=True)
            download_file(source, package_path, digest)
            packages.append({"path": str(package_path), "sha256": digest, "source": source})
    for package in packages:
        extract_checked(Path(package["path"]), destination)
    servers = sorted(destination.rglob("llama-server.exe" if os.name == "nt" else "llama-server"))
    if not servers:
        raise RuntimeError("运行时包中找不到 llama-server。")
    server = servers[0].resolve()
    # --version 实际加载运行库，可发现缺少 DLL，但不冒充模型推理测试。
    result = subprocess.run([str(server), "--version"], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=30)
    if result.returncode:
        raise RuntimeError("llama-server 不能运行，请检查 VC++ 运行库和对应 CUDA DLL。")
    manifest = {"schema": 1, "release": LLAMA_RELEASE if not (url or archive) else "custom",
                "variant": variant, "server_path": str(server), "packages": packages,
                "server_sha256": sha256_file(server), "version_output": (result.stdout + result.stderr).strip()}
    atomic_json(runtime / "runtime-manifest.json", manifest)
    update_config(settings, llama_server_path=str(server), llama_gpu_layers=0 if variant == "cpu" else settings.llama_gpu_layers)
    return manifest


def install_ffmpeg(settings: Settings) -> dict:
    """Conda FFmpeg 安装失败时使用官网列出的 Gyan 固定构建，不修改已有 Conda 包。"""
    if os.name != "nt":
        raise RuntimeError("此静态 FFmpeg 包仅适用于 Windows x64。")
    runtime = Path(settings.runtime_dir)
    archive = runtime / "downloads" / "ffmpeg-8.1.2-essentials_build.zip"
    download_file(FFMPEG_URL, archive, FFMPEG_SHA256)
    destination = runtime / "ffmpeg"
    extract_checked(archive, destination)
    ffmpeg = next(destination.rglob("ffmpeg.exe"), None)
    ffprobe = next(destination.rglob("ffprobe.exe"), None)
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg 发行包缺少 ffmpeg.exe 或 ffprobe.exe。")
    for program in (ffmpeg, ffprobe):
        subprocess.run([str(program), "-version"], capture_output=True, check=True, timeout=30)
    update_config(settings, ffmpeg=str(ffmpeg.resolve()), ffprobe=str(ffprobe.resolve()))
    result = {"source": FFMPEG_URL, "sha256": FFMPEG_SHA256, "ffmpeg": str(ffmpeg.resolve()),
              "ffprobe": str(ffprobe.resolve()), "license": "GPL 构建，完整许可及版本说明保留在解压目录"}
    atomic_json(runtime / "ffmpeg-manifest.json", result)
    return result


def managed_process(record: dict) -> psutil.Process | None:
    """PID 可能被系统重复使用；同时核对创建时间、可执行文件和模型参数才认为归本项目所有。"""
    import psutil

    try:
        process = psutil.Process(record["pid"])
        if abs(process.create_time() - record["create_time"]) > 0.1:
            return None
        if Path(process.exe()).resolve() != Path(record["executable"]).resolve():
            return None
        if record["model"] not in process.cmdline():
            return None
        return process
    except (KeyError, psutil.Error, OSError):
        return None


def start_local(settings: Settings, owner: str = "manual", wait_seconds: float = 180) -> dict:
    """串行化启动过程，避免两次双击同时抢端口、互相覆盖 PID 清单。"""
    from filelock import FileLock

    runtime = Path(settings.runtime_dir)
    runtime.mkdir(parents=True, exist_ok=True)
    with FileLock(str(runtime / "llama-start.lock"), timeout=wait_seconds + 30):
        return _start_local_locked(settings, owner, wait_seconds)


def _start_local_locked(settings: Settings, owner: str, wait_seconds: float) -> dict:
    """启动仅监听回环地址的受管本地模型；成功必须通过 /health 检查。"""
    import httpx
    import psutil

    if not settings.local_enabled:
        return {"status": "disabled", "started": False}
    runtime = Path(settings.runtime_dir)
    runtime.mkdir(parents=True, exist_ok=True)
    record_path = runtime / "llama-process.json"
    record = read_json(record_path)
    if record and managed_process(record):
        host, port = record.get("host", "127.0.0.1"), record.get("port", 8081)
        address = f"http://[{host}]:{port}/health" if host == "::1" else f"http://{host}:{port}/health"
        try:
            with httpx.Client(timeout=5, trust_env=False) as client:
                if client.get(address).status_code != 200:
                    raise RuntimeError("已有本地模型进程但健康检查未通过，请检查运行日志后停止并重启。")
        except httpx.HTTPError as exc:
            raise RuntimeError("已有本地模型进程无法通过健康检查，请检查运行日志后停止并重启。") from exc
        return {"status": "already_running", "started": False, "pid": record["pid"], "owner": record.get("owner")}
    server = Path(settings.llama_server_path or read_json(runtime / "runtime-manifest.json").get("server_path", ""))
    model_manifest = read_json(Path(settings.models_dir) / "model-manifest.json")
    model = Path(settings.llama_model_path or model_manifest.get("models", {}).get("local", {}).get("model_file", ""))
    if not server.is_file() or not model.is_file():
        raise RuntimeError("本地运行时或 GGUF 尚未部署。请依次执行 models --names local 和 runtime。")
    endpoint = urlparse(settings.local_base_url)
    if endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("受管 llama.cpp 仅允许监听本机回环地址。")
    host, port = endpoint.hostname, endpoint.port or 8081
    with socket.socket(socket.AF_INET6 if host == "::1" else socket.AF_INET) as probe:
        try:
            probe.bind((host, port))
        except OSError as exc:
            raise RuntimeError(f"本地模型端口 {port} 已被其他服务占用；不会停止或接管该进程。") from exc
    command = [str(server.resolve()), "--model", str(model.resolve()), "--host", host, "--port", str(port),
               "--alias", settings.local_model, "--ctx-size", str(getattr(settings, "local_context_tokens", 8192)),
               "--n-gpu-layers", str(settings.llama_gpu_layers), "--parallel", "1",
               "--jinja", "--chat-template-kwargs", '{"enable_thinking":false}']
    log_path = runtime / "llama-server.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    identity = psutil.Process(process.pid)
    record = {"pid": process.pid, "create_time": identity.create_time(), "executable": str(server.resolve()),
              "model": str(model.resolve()), "owner": owner, "host": host, "port": port}
    atomic_json(record_path, record)
    try:
        deadline = time.monotonic() + wait_seconds
        with httpx.Client(timeout=3, trust_env=False) as client:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server 提前退出，查看日志：{log_path}")
                try:
                    response = client.get(f"http://[{host}]:{port}/health" if host == "::1" else f"http://{host}:{port}/health")
                    if response.status_code == 200:
                        return {"status": "ready", "started": True, "pid": process.pid, "owner": owner}
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
        raise RuntimeError(f"本地模型在 {wait_seconds:g} 秒内未就绪，查看日志：{log_path}")
    except BaseException:
        stop_local(settings, owner)
        raise


def stop_local(settings: Settings, owner: str | None = None) -> dict:
    """只终止当前项目清单所指向、身份仍完全匹配的服务，绝不按名称批量杀进程。"""
    import psutil

    record_path = Path(settings.runtime_dir) / "llama-process.json"
    record = read_json(record_path)
    if owner is not None and record.get("owner") != owner:
        return {"status": "not_owned", "stopped": False}
    process = managed_process(record) if record else None
    if process:
        process.terminate()
        try:
            process.wait(timeout=10)
        except psutil.TimeoutExpired:
            # 再次核对 PID 身份，避免等待期间进程退出、PID 复用后误杀。
            if managed_process(record):
                process.kill()
                process.wait(timeout=5)
    record_path.unlink(missing_ok=True)
    return {"status": "stopped" if process else "not_running", "stopped": bool(process)}


def main(argv: list[str] | None = None) -> int:
    """命令行入口使用中文说明，失败返回非零状态，便于一键脚本停止。"""
    for output in (sys.stdout, sys.stderr):
        if hasattr(output, "reconfigure"):
            output.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="知影模型下载、完整性校验及本地生成服务管理")
    parser.add_argument("--config", help="自定义配置 JSON 路径")
    commands = parser.add_subparsers(dest="command", required=True)
    for command, description in (("models", "下载并验证全部或指定模型"), ("verify-models", "离线核对模型清单")):
        sub = commands.add_parser(command, help=description)
        sub.add_argument("--names", default="asr,text,image,local", help="逗号分隔：asr,text,image,local")
    runtime = commands.add_parser("runtime", help="部署固定版本 llama.cpp（默认 CPU）")
    runtime.add_argument("--variant", choices=["cpu", "cuda"], default="cpu")
    runtime.add_argument("--url", help="自定义发行包 URL，必须同时提供 SHA-256")
    runtime.add_argument("--archive", help="已经下载的本地 ZIP 文件")
    runtime.add_argument("--sha256", help="自定义包的完整 SHA-256")
    commands.add_parser("ffmpeg", help="安装固定版本 Windows 静态 FFmpeg，作为 Conda 安装失败的回退")
    start = commands.add_parser("start-local", help="启动受管本地模型并等待健康检查")
    start.add_argument("--owner", default="manual", help="启动者标识，自动脚本使用随机值")
    start.add_argument("--wait", type=float, default=180, help="模型加载最大等待秒数")
    stop = commands.add_parser("stop-local", help="仅停止本项目管理且身份匹配的模型服务")
    stop.add_argument("--owner", help="只停止由此启动者创建的服务")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        if args.command in {"models", "verify-models"}:
            names = [name.strip() for name in args.names.split(",") if name.strip()]
            if not names or set(names) - MODEL_SPECS.keys():
                raise ValueError("模型名称只能是 asr,text,image,local。")
            result = download_models(settings, names) if args.command == "models" else verify_models(settings, names)
            if args.command == "models":
                result = {"status": "installed", "models": names, "manifest": str(Path(settings.models_dir) / "model-manifest.json")}
        elif args.command == "runtime":
            result = install_runtime(settings, args.variant, args.url, args.sha256, args.archive)
        elif args.command == "ffmpeg":
            result = install_ffmpeg(settings)
        elif args.command == "start-local":
            result = start_local(settings, args.owner, args.wait)
        else:
            result = stop_local(settings, args.owner)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"部署操作未完成：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
