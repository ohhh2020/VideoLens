@echo off
rem ===================================================================
rem  ZhiYing launcher shell.
rem  This file is intentionally pure ASCII: cmd.exe parses .bat files with
rem  the system ANSI code page, so non-ASCII text here corrupts the script.
rem  All real logic and Chinese UI text live in start.ps1 (UTF-8 BOM).
rem ===================================================================
setlocal
cd /d "%~dp0"

set "PSH=powershell.exe"
where /q pwsh.exe && set "PSH=pwsh.exe"

rem --- BOM guard ---------------------------------------------------
rem  Windows PowerShell 5.1 decodes a BOM-less .ps1 with the ANSI code
rem  page and then reports bogus parser errors, so re-add the BOM if it
rem  ever goes missing (editors drop it silently). ASCII-only on purpose.
"%PSH%" -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$p = Join-Path '%~dp0' 'start.ps1'; if (Test-Path -LiteralPath $p) { $b = [IO.File]::ReadAllBytes($p); if ($b.Length -ge 3 -and $b[0] -eq 0xEF -and $b[1] -eq 0xBB -and $b[2] -eq 0xBF) { exit 0 }; $t = [IO.File]::ReadAllText($p, [Text.Encoding]::UTF8); [IO.File]::WriteAllText($p, $t, (New-Object Text.UTF8Encoding $true)) }" >nul 2>&1

"%PSH%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo   [launcher] exited with code %RC%
    pause
)
endlocal
exit /b %RC%