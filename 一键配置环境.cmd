@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "VIDEOLENS_SCRIPT_ROOT=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& { $s = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $env:VIDEOLENS_SCRIPT_ROOT 'scripts\setup.ps1'); & ([scriptblock]::Create($s)) -ProjectRoot $env:VIDEOLENS_SCRIPT_ROOT }"
if errorlevel 1 (
  echo 配置未完成，请查看上方错误和 docs\故障排查.md。
  pause
  exit /b 1
)
echo 环境配置完成，可以运行“一键启动.cmd”。
pause
