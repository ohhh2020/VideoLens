@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "VIDEOLENS_SCRIPT_ROOT=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& { $s = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $env:VIDEOLENS_SCRIPT_ROOT 'scripts\test.ps1'); & ([scriptblock]::Create($s)) -ProjectRoot $env:VIDEOLENS_SCRIPT_ROOT }"
set "VIDEOLENS_EXIT=%ERRORLEVEL%"
pause
exit /b %VIDEOLENS_EXIT%
