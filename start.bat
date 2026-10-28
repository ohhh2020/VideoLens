@echo off
setlocal
"%SystemRoot%\System32\chcp.com" 65001 >nul
cd /d "%~dp0"
if "%~1"=="--help" goto help
if "%~1"=="-h" goto help
if not defined CONDA_PREFIX (
  echo 请先运行 conda activate VideoLens，再启动网页。
  exit /b 2
)
set "VIDEOLENS_PYTHON=%CONDA_PREFIX%\python.exe"
if not exist "%VIDEOLENS_PYTHON%" (
  echo 当前激活的 Conda 环境中找不到 python.exe。
  exit /b 2
)
if not defined VIDEOLENS_RUNTIME_ROOT set "VIDEOLENS_RUNTIME_ROOT=%~dp0..\VideoLens-runtime"
set "PYTHONNOUSERSITE=1"
set "PYTHONHOME="
set "PYTHONPATH=%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPYCACHEPREFIX=%VIDEOLENS_RUNTIME_ROOT%\cache\pycache"
"%VIDEOLENS_PYTHON%" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,13) else 2)"
if errorlevel 1 (
  echo 当前环境必须使用 Python 3.13，请检查 conda activate 的环境。
  exit /b 2
)
"%VIDEOLENS_PYTHON%" -m backend.web.console %*
exit /b %ERRORLEVEL%

:help
echo 用法：先 conda activate VideoLens，再运行 start.bat [--host 本机地址] [--port 8000] [--reload]
echo --host: 可选 127.0.0.1、localhost、::1
echo --port: 网页服务端口
echo --reload: 开发时自动重载
echo 启动后可输入 help、status、stop、restart；Ctrl+C 也可关闭。
exit /b 0
