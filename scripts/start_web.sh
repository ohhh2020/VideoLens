#!/usr/bin/env bash
# 在用户可见的 Git Bash 中运行；Ctrl+C 停止服务，不启动后台进程。
set -e
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "请先执行 conda activate VideoLens，再运行 bash scripts/start_web.sh。"
  exit 2
fi
conda_prefix="$(cygpath -u "$CONDA_PREFIX")"
if [[ ! -f "$conda_prefix/python.exe" ]]; then
  echo "当前 Conda 环境中找不到 python.exe。"
  exit 2
fi
runtime_root="${VIDEOLENS_RUNTIME_ROOT:-$(dirname "$project_root")/VideoLens-runtime}"
export PYTHONNOUSERSITE=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
unset PYTHONHOME
export PYTHONPATH="$(cygpath -w "$project_root")"
export VIDEOLENS_RUNTIME_ROOT="$(cygpath -w "$runtime_root")"
export PYTHONPYCACHEPREFIX="$(cygpath -w "$runtime_root/cache/pycache")"
"$conda_prefix/python.exe" -m backend.web.console "$@"
