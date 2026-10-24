#Requires -Version 5.1
<#
    知影 ZhiYing · 一键启动器 / 环境检查
    ------------------------------------------------------------
    真正的实现在这里（PowerShell 对 UTF-8 中文零问题）。
    start.bat 只是个 ASCII 外壳，双击它即可。

    为什么不用纯 bat：cmd 按系统 ANSI 代码页解析 .bat 文件，
    UTF-8 中文字节会破坏命令行结构，GBK 中文字节里又混有
    0x7C/0x5E/0x5C 这类 cmd 保留字符（次字节），两条路都是坑。
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Command = '',

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
)

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

try { $Host.UI.RawUI.WindowTitle = '知影 ZhiYing' } catch { }

# ---------------------------------------------------------------- 编码
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
try { $OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

# ---------------------------------------------------------------- 路径
$Root = $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }
Set-Location -LiteralPath $Root
$Script:PreflightFile = Join-Path $env:TEMP 'zhiying_preflight.json'

# ---------------------------------------------------------------- 外观
$C_OK = 'Green'
$C_WARN = 'Yellow'
$C_BAD = 'Red'
$C_DIM = 'DarkGray'
$C_HEAD = 'Cyan'
$C_ACC = 'Magenta'

function Write-Rule {
    param([string]$Char = '=', [int]$Width = 66, [string]$Color = 'DarkGray')
    Write-Host ('  ' + ($Char * $Width)) -ForegroundColor $Color
}

function Write-Title {
    param([string]$Text, [string]$Color = 'Cyan')
    Write-Host ''
    Write-Rule
    Write-Host ("   " + $Text) -ForegroundColor $Color
    Write-Rule
    Write-Host ''
}

function Write-Step { param([string]$T) Write-Host ("   " + $T) -ForegroundColor $C_DIM }
function Write-Good { param([string]$T) Write-Host ("   [OK] " + $T) -ForegroundColor $C_OK }
function Write-Warn { param([string]$T) Write-Host ("   [!]  " + $T) -ForegroundColor $C_WARN }
function Write-Bad  { param([string]$T) Write-Host ("   [X]  " + $T) -ForegroundColor $C_BAD }
function Write-Info { param([string]$T) Write-Host ("        " + $T) -ForegroundColor $C_DIM }

function Show-ExitCode {
    param([string]$What = '完成')
    $code = $LASTEXITCODE
    if ($null -eq $code) { $code = 0 }
    if ($code -eq 0) { Write-Good ($What + '，退出码 0') }
    else { Write-Warn ($What + '，退出码 ' + $code) }
}

function Read-Answer {
    param([string]$Prompt, [string]$Default = '')
    $raw = $null
    try { $raw = Read-Host $Prompt } catch { return $Default }
    if ($null -eq $raw) { return $Default }
    $s = ([string]$raw).Trim()
    if ($s -eq '') { return $Default }
    return $s
}

function Pause-Any {
    Write-Host ''
    Write-Host '   按回车继续...' -ForegroundColor $C_DIM -NoNewline
    $null = Read-Host
}

# ---------------------------------------------------------------- Python 定位
function Find-Python {
    $cands = New-Object System.Collections.ArrayList

    $venv = Join-Path $Root '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venv) { $null = $cands.Add($venv) }

    foreach ($v in @('3.12', '3.11')) {
        try {
            $out = & py "-$v" -c "import sys;print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) { $null = $cands.Add(([string]$out).Trim()) }
        } catch { }
    }

    try {
        $out = & python -c "import sys;print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { $null = $cands.Add(([string]$out).Trim()) }
    } catch { }

    foreach ($c in $cands) {
        if (-not (Test-Path -LiteralPath $c)) { continue }
        try {
            $ver = & $c -c "import sys;print(sys.version.split()[0])" 2>$null
            if ($LASTEXITCODE -ne 0 -or -not $ver) { continue }
            $ok = & $c -c "import sys;print(1 if sys.version_info[:2]>=(3,11) else 0)" 2>$null
            if (([string]$ok).Trim() -eq '1') {
                return [pscustomobject]@{ Exe = $c; Version = ([string]$ver).Trim() }
            }
        } catch { }
    }
    return $null
}

$Script:Python = Find-Python
$Script:BrowserJobs = New-Object System.Collections.ArrayList

# ---------------------------------------------------------------- 预检
function Get-Preflight {
    param([switch]$Force)

    if (-not $Force -and $Script:Report) { return $Script:Report }
    if (-not $Script:Python) { return $null }

    $pfPath = Join-Path $Root 'scripts\preflight.py'
    if (-not (Test-Path -LiteralPath $pfPath)) {
        return [pscustomobject]@{ verdict = 'UNKNOWN'; checks = @(); errors = @('scripts\preflight.py 不存在') }
    }

    $raw = & $Script:Python.Exe $pfPath --json 2>$null
    $text = ($raw | Out-String)
    try {
        $rep = $text | ConvertFrom-Json
    } catch {
        return [pscustomobject]@{ verdict = 'UNKNOWN'; checks = @(); errors = @('预检输出无法解析: ' + $_.Exception.Message) }
    }
    $Script:Report = $rep
    return $rep
}

function Get-Check {
    param($Report, [string]$Key)
    if (-not $Report) { return $null }
    foreach ($c in $Report.checks) { if ($c.key -eq $Key) { return $c } }
    return $null
}

function Find-FreePort {
    param([int]$Start = 8000, [int]$Span = 50)
    for ($p = $Start; $p -lt ($Start + $Span); $p++) {
        $listener = $null
        try {
            $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $p)
            $listener.Start()
            $listener.Stop()
            return $p
        } catch {
            if ($listener) { try { $listener.Stop() } catch { } }
        }
    }
    return $Start
}

function Get-WebPort {
    $rep = Get-Preflight
    $c = Get-Check $rep 'PORT'
    $base = 8000
    if ($c -and $c.detail) {
        $m = [regex]::Match([string]$c.detail, '(\d{2,5})')
        if ($m.Success) { $base = [int]$m.Groups[1].Value }
    }
    return (Find-FreePort -Start $base)
}

# ---------------------------------------------------------------- 闸门
function Confirm-Gate {
    $rep = Get-Preflight
    if (-not $rep) { return $true }

    switch ($rep.verdict) {
        'READY'   { return $true }
        'UNKNOWN' { return $true }
        'BROKEN' {
            Clear-Host
            Write-Title '环境未通过预检 —— 启动大概率失败' $C_BAD
            Show-PreflightDetail
            Write-Host ''
            Write-Bad '先修复上面标 [X] 的项'
            Write-Host ''
            Write-Step '常见修法：'
            Write-Info '缺依赖           pip install -r requirements.txt'
            Write-Info '数据库损坏       回菜单选 [9] 初始化数据库'
            Write-Info '配置写错         检查 configs\app_config.yaml'
            Write-Info '目录没权限       把项目放到用户可写路径'
            Write-Info '缺 Python 包之外  python clients\cli.py doctor 看细项'
            Pause-Any
            return $false
        }
        'DEGRADED' {
            Clear-Host
            Write-Title '环境可用，但有降级项' $C_WARN
            foreach ($c in $rep.checks) {
                if ($c.state -in @('MISS', 'WARN', 'FAIL')) {
                    $glyph = switch ($c.state) { 'MISS' { '[ ]' } 'WARN' { '[!]' } 'FAIL' { '[X]' } }
                    Write-Host ("   " + $glyph + " " + $c.key.PadRight(10) + $c.detail) -ForegroundColor $C_WARN
                    if ($c.note) { Write-Info $c.note }
                }
            }
            Write-Host ''
            Write-Step '降级不等于崩溃：缺模型的通道会被跳过或改用替代算法，'
            Write-Step '但结果质量下降（检索只剩字面匹配、关键帧没有画面描述等）。'
            Write-Host ''
            if ($env:ZY_AUTOYES) { return $true }
            Write-Step '3 秒后自动继续（缺核心组件时按 Ctrl+C 中断，回菜单选 [8] 下载）'
            Start-Sleep -Seconds 3
            return $true
        }
        default { return $true }
    }
}

# ---------------------------------------------------------------- 展示
function Show-PreflightDetail {
    $rep = Get-Preflight
    if (-not $rep) { Write-Bad '预检不可用'; return }
    foreach ($c in $rep.checks) {
        $color = switch ($c.state) {
            'OK'   { $C_OK }
            'SKIP' { $C_DIM }
            'MISS' { $C_WARN }
            'WARN' { $C_WARN }
            'FAIL' { $C_BAD }
            default { 'White' }
        }
        $glyph = switch ($c.state) {
            'OK' { '[OK]' }
            'SKIP' { '[-] ' }
            'MISS' { '[ ] ' }
            'WARN' { '[!] ' }
            'FAIL' { '[X] ' }
            default { '[?] ' }
        }
        $line = "   " + $glyph + " " + $c.key.PadRight(11) + [string]$c.detail
        Write-Host $line -ForegroundColor $color
        if ($c.note -and $c.state -in @('MISS', 'WARN', 'FAIL')) {
            Write-Host ("        -> " + $c.note) -ForegroundColor $C_DIM
        }
    }
    Write-Host ''
    $v = [string]$rep.verdict
    $color = switch ($v) { 'READY' { $C_OK } 'DEGRADED' { $C_WARN } default { $C_BAD } }
    Write-Host ("   结论：" + $v + "    （耗时 " + [string]$rep.elapsed_ms + " ms）") -ForegroundColor $color
}

function Show-StatusLine {
    $rep = Get-Preflight
    $v = if ($rep) { [string]$rep.verdict } else { 'UNKNOWN' }
    $color = switch ($v) { 'READY' { $C_OK } 'DEGRADED' { $C_WARN } default { $C_BAD } }
    $py = if ($Script:Python) { $Script:Python.Version } else { '?' }
    Write-Host '   ' -NoNewline
    Write-Host '环境 ' -NoNewline -ForegroundColor $C_DIM
    Write-Host $v -NoNewline -ForegroundColor $color
    Write-Host ('     Python ' + $py) -NoNewline -ForegroundColor $C_DIM
    Write-Host ('     端口 ' + (Get-WebPort)) -ForegroundColor $C_DIM
}

# ---------------------------------------------------------------- 动作
function Invoke-Py {
    param([string[]]$PyArgs)
    & $Script:Python.Exe @PyArgs
    return $LASTEXITCODE
}

function Open-BrowserLater {
    param([string]$Url, [int]$Port = 0, [int]$TimeoutS = 120)
    # 不能靠固定延时：服务要先 import gradio（约 6 秒）、再装配模型（约 15 秒）
    # 才真正 listen。固定等 7 秒会让浏览器撞上 ERR_CONNECTION_REFUSED。
    # 改成后台轮询端口，一 listen 立刻开页面。
    try {
        if ($Port -le 0) { $Port = ([uri]$Url).Port }
        if ($Port -le 0) { $Port = 80 }
        $cmd = @(
            "`$deadline = (Get-Date).AddSeconds($TimeoutS)"
            "while ((Get-Date) -lt `$deadline) {"
            "  try {"
            "    `$c = New-Object Net.Sockets.TcpClient"
            "    `$c.Connect('127.0.0.1', $Port)"
            "    `$c.Close()"
            "    Start-Sleep -Milliseconds 500"
            "    Start-Process '$Url'"
            "    break"
            "  } catch { Start-Sleep -Milliseconds 400 }"
            "}"
        ) -join '; '
        Start-Process -FilePath 'powershell.exe' `
            -ArgumentList '-NoProfile', '-WindowStyle', 'Hidden', '-Command', $cmd `
            -WindowStyle Hidden | Out-Null
    } catch { }
}

function Start-WebApp {
    if (-not $Script:Python) { Show-NoPython; return }
    $gate = Confirm-Gate
    if ($gate -eq $false) { return }
    if ("$gate" -eq 'models') { Start-ModelMenu; return }

    $port = Get-WebPort
    $url = "http://127.0.0.1:$port"
    Clear-Host
    Write-Title '启动 知影 Web 应用'
    Write-Step ('界面    ' + $url + '   （根地址直接打开界面）')
    Write-Step ('接口文档 ' + $url + '/docs')
    Write-Step ('REST     ' + $url + '/api')
    Write-Host ''
    Write-Step '首次启动要加载界面组件和模型，约 20~25 秒，请勿关闭窗口。'
    Write-Step '浏览器会在服务就绪后自动打开（现在打开会显示“拒绝连接”，属正常）。'
    Write-Step 'Ctrl+C 停止服务。'
    Write-Host ''
    Write-Rule
    Write-Host ''

    Open-BrowserLater -Url $url -Port $port
    try {
        $null = Invoke-Py @('-m', 'api.main', '--host', '127.0.0.1', '--port', [string]$port)
    } finally {
        Write-Host ''
        Write-Warn '服务已退出'
    }
    Pause-Any
}

function Start-ApiOnly {
    if (-not $Script:Python) { Show-NoPython; return }
    $gate = Confirm-Gate
    if ($gate -eq $false) { return }
    if ("$gate" -eq 'models') { Start-ModelMenu; return }

    $port = Get-WebPort
    Clear-Host
    Write-Title '启动 API 服务（不打开浏览器）'
    Write-Step ('REST    http://127.0.0.1:' + $port + '/api')
    Write-Step ('MCP     http://127.0.0.1:' + $port + '/api/mcp')
    Write-Step ('文档    http://127.0.0.1:' + $port + '/docs')
    Write-Host ''
    Write-Rule
    Write-Host ''
    try {
        $null = Invoke-Py @('-m', 'api.main', '--host', '127.0.0.1', '--port', [string]$port)
    } finally {
        Write-Warn '服务已退出'
    }
    Pause-Any
}

function Start-GradioApp {
    if (-not $Script:Python) { Show-NoPython; return }
    $gate = Confirm-Gate
    if ($gate -eq $false) { return }
    if ("$gate" -eq 'models') { Start-ModelMenu; return }

    $port = Find-FreePort -Start 7860
    $url = "http://127.0.0.1:$port"
    Clear-Host
    Write-Title '启动 Gradio 界面'
    Write-Step ('地址  ' + $url)
    Write-Host ''
    Write-Rule
    Write-Host ''
    Open-BrowserLater -Url $url -Port $port
    try {
        $null = Invoke-Py @('clients\gradio_ui.py', '--host', '127.0.0.1', '--port', [string]$port)
    } finally {
        Write-Warn 'Gradio 已退出'
    }
    Pause-Any
}

function Start-Analyze {
    param([string]$Target)

    if (-not $Script:Python) { Show-NoPython; return }

    if ([string]::IsNullOrWhiteSpace($Target)) {
        Clear-Host
        Write-Title '分析视频'
        Write-Step '本地文件：把文件拖进本窗口，或粘贴完整路径'
        Write-Step '在线视频：粘贴网址（B 站 / YouTube 等，需 yt-dlp）'
        Write-Host ''
        $Target = Read-Host '   视频路径或链接'
        if ([string]::IsNullOrWhiteSpace($Target)) { return }
    }

    Write-Host ''
    $null = Invoke-Py @('clients\cli.py', 'analyze', $Target, '--report')
    Write-Host ''
    Show-ExitCode '分析结束'
    Pause-Any
}

function Start-Ask {
    param([string]$VideoId, [string]$Question)

    if (-not $Script:Python) { Show-NoPython; return }

    if ([string]::IsNullOrWhiteSpace($Question)) {
        Clear-Host
        Write-Title '对已索引视频提问'
        & $Script:Python.Exe 'clients\cli.py' 'list' 2>$null
        Write-Host ''
        $VideoId = Read-Host '   视频 ID'
        if ([string]::IsNullOrWhiteSpace($VideoId)) { return }
        $Question = Read-Host '   你的问题'
        if ([string]::IsNullOrWhiteSpace($Question)) { return }
    }

    Write-Host ''
    $null = Invoke-Py @('clients\cli.py', 'ask', $VideoId, $Question)
    Write-Host ''
    Pause-Any
}

function Start-Library {
    if (-not $Script:Python) { Show-NoPython; return }
    Clear-Host
    Write-Title '视频库'
    & $Script:Python.Exe 'clients\cli.py' 'list' 2>$null
    Write-Host ''
    & $Script:Python.Exe 'clients\cli.py' 'stats' 2>$null
    Write-Host ''
    Write-Rule '-'
    $id = Read-Host '   输入视频 ID 可导出 Markdown 报告（直接回车跳过）'
    if (-not [string]::IsNullOrWhiteSpace($id)) {
        Write-Host ''
        & $Script:Python.Exe 'clients\cli.py' 'export' $id.Trim()
    }
    Write-Host ''
    Pause-Any
}

function Show-Check {
    param([switch]$NoPause)

    if (-not $Script:Python) { Show-NoPython; return }

    Clear-Host
    Write-Title '启动预检（3 秒，不加载模型）'
    Show-PreflightDetail

    Write-Host ''
    Write-Title '完整自检（依赖版本 + 模型清单）'
    & $Script:Python.Exe 'scripts\preflight.py' 2>$null | Out-Null
    & $Script:Python.Exe 'clients\cli.py' 'doctor'
    Write-Host ''
    if (-not $NoPause) { Pause-Any }
}

function Start-ModelMenu {
    if (-not $Script:Python) { Show-NoPython; return }

    while ($true) {
        Clear-Host
        Write-Title '模型权重'
        & $Script:Python.Exe 'scripts\download_models.py' '--status' 2>$null
        Write-Host ''
        Write-Step '[1] 核心两件套   turbo 1.6G + bge-m3 2.3G     完整链路'
        Write-Step '[2] 加图像检索   + siglip2-base 1.6G          以文搜画面'
        Write-Step '[3] 加视觉理解   + qwen2-vl-2b 2.0G           画面描述'
        Write-Step '[4] 全量         + qwen2.5-7b 4.7G            本地大模型，CPU 慢'
        Write-Step '[5] 自定义模型名（逗号分隔）'
        Write-Step '[6] 校验已有权重（sha256）'
        Write-Step '[0] 返回'
        Write-Host ''
        $sel = Read-Answer '   请选择' '0'

        $list = $null
        switch ($sel) {
            '1' { $list = 'faster-whisper-large-v3-turbo,bge-m3' }
            '2' { $list = 'faster-whisper-large-v3-turbo,bge-m3,siglip2-base' }
            '3' { $list = 'faster-whisper-large-v3-turbo,bge-m3,qwen2-vl-2b-gguf' }
            '4' {
                Write-Host ''
                Write-Warn '全量约 10.6 GB，CPU 机器下载与首次加载都慢，可 Ctrl+C 中断'
                Pause-Any
                Write-Host ''
                & $Script:Python.Exe 'scripts\download_models.py' '--all'
                Write-Host ''
                Show-ExitCode '下载结束'
                $Script:Report = $null
                Pause-Any
                continue
            }
            '5' { $list = Read-Host '   模型名（如 siglip2-base,qwen2-vl-2b-gguf）' }
            '6' {
                Write-Host ''
                & $Script:Python.Exe 'scripts\download_models.py' '--verify'
                Write-Host ''
                Show-ExitCode '校验结束'
                $Script:Report = $null
                Pause-Any
                continue
            }
            '0' { return }
            default {
                Write-Warn '无效输入'
                Start-Sleep -Seconds 1
                continue
            }
        }

        if (-not [string]::IsNullOrWhiteSpace($list)) {
            Write-Host ''
            Write-Step ('正在下载：' + $list)
            Write-Host ''
            & $Script:Python.Exe 'scripts\download_models.py' '--models' $list
            Write-Host ''
            Show-ExitCode '下载结束'
            $Script:Report = $null
            Pause-Any
        }
    }
}

function Start-InitDb {
    if (-not $Script:Python) { Show-NoPython; return }
    Clear-Host
    Write-Title '初始化 / 修复数据库'
    Write-Step '创建缺失的表与目录，已有数据不会被删除。'
    Write-Host ''
    & $Script:Python.Exe 'scripts\init_db.py'
    Write-Host ''
    Show-ExitCode '完成'
    $Script:Report = $null
    Pause-Any
}

function Show-NoPython {
    Clear-Host
    Write-Title '没找到可用的 Python' $C_BAD
    Write-Step '知影需要 Python 3.11 或 3.12。'
    Write-Host ''
    Write-Host '   情况 A：还没装' -ForegroundColor $C_HEAD
    Write-Info 'https://www.python.org/downloads/windows/'
    Write-Info '安装时勾选 "Add python.exe to PATH"'
    Write-Host ''
    Write-Host '   情况 B：装了但不在 PATH' -ForegroundColor $C_HEAD
    Write-Info '在项目根目录建虚拟环境，本脚本会优先用它：'
    Write-Info '  py -3.11 -m venv .venv'
    Write-Info '  .venv\Scripts\python.exe -m pip install -r requirements.txt'
    Write-Host ''
    Write-Host '   情况 C：装在别的盘' -ForegroundColor $C_HEAD
    Write-Info '把 Python 目录加进 PATH'
    Write-Host ''
    Write-Warn '不要用 Python 3.13 —— 部分依赖还没有预编译轮子'
    Pause-Any
}

function Show-Usage {
    Write-Host ''
    Write-Host '   知影 ZhiYing 启动器' -ForegroundColor $C_HEAD
    Write-Host ''
    Write-Step 'start.bat                       交互菜单'
    Write-Step 'start.bat web                   启动 Web（自动开浏览器）'
    Write-Step 'start.bat serve                 只启动 API'
    Write-Step 'start.bat gradio                Gradio 界面'
    Write-Step 'start.bat check                 环境检查'
    Write-Step 'start.bat analyze "路径/URL"    直接分析'
    Write-Step 'start.bat ask ID "问题"         直接提问'
    Write-Step 'start.bat models                下载模型'
    Write-Step 'start.bat init                  初始化数据库'
    Write-Step 'start.bat help                  本帮助'
    Write-Host ''
}

# ---------------------------------------------------------------- 菜单
function Show-Menu {
    Clear-Host
    Write-Host ''
    Write-Host '  ================================================================' -ForegroundColor $C_HEAD
    Write-Host '    知影 ZhiYing  ·  本地优先的影片理解与检索' -ForegroundColor $C_HEAD
    Write-Host '  ================================================================' -ForegroundColor $C_HEAD
    Show-StatusLine
    Write-Host '  ================================================================' -ForegroundColor $C_HEAD
    Write-Host ''
    Write-Host '    [1] 一键启动（Web 应用，自动打开浏览器）' -ForegroundColor White
    Write-Host '    [2] 只启动 API 服务（Obsidian / 远程客户端）' -ForegroundColor Gray
    Write-Host '    [3] 启动 Gradio 界面（备用，支持拖拽上传）' -ForegroundColor Gray
    Write-Host ''
    Write-Host '    [4] 分析视频（本地文件或链接）' -ForegroundColor Gray
    Write-Host '    [5] 对已索引视频提问' -ForegroundColor Gray
    Write-Host '    [6] 查看视频库 / 导出报告' -ForegroundColor Gray
    Write-Host ''
    Write-Host '    [7] 环境检查（详细）' -ForegroundColor Gray
    Write-Host '    [8] 下载 / 补全模型权重' -ForegroundColor Gray
    Write-Host '    [9] 初始化 / 修复数据库' -ForegroundColor Gray
    Write-Host ''
    Write-Host '    [R] 重新预检' -ForegroundColor DarkGray
    Write-Host '    [0] 退出' -ForegroundColor DarkGray
    Write-Host ''
}

# ---------------------------------------------------------------- 主流程
if (-not $Script:Python) {
    if ($Command -in @('help', '-h', '--help')) { Show-Usage; exit 0 }
    Show-NoPython
    exit 1
}

switch ($Command.ToLower()) {
    'web'     { Start-WebApp; exit 0 }
    'serve'   { Start-ApiOnly; exit 0 }
    'gradio'  { Start-GradioApp; exit 0 }
    'check'   { Show-Check -NoPause; exit 0 }
    'analyze' { if ($Rest.Count -gt 0) { Start-Analyze -Target $Rest[0] } else { Start-Analyze }; exit 0 }
    'ask'     {
        if ($Rest.Count -ge 2) { Start-Ask -VideoId $Rest[0] -Question $Rest[1] }
        elseif ($Rest.Count -eq 1) { Start-Ask -VideoId '' -Question $Rest[0] }
        else { Start-Ask }
        exit 0
    }
    'models'  { Start-ModelMenu; exit 0 }
    'init'    { Start-InitDb; exit 0 }
    'help'    { Show-Usage; exit 0 }
    '-h'      { Show-Usage; exit 0 }
    '--help'  { Show-Usage; exit 0 }
    ''        { }
    default {
        Write-Bad ('未知参数：' + $Command)
        Show-Usage
        Pause-Any
        exit 2
    }
}

# 交互菜单
$null = Get-Preflight -Force

$running = $true
while ($running) {
    Show-Menu
    $choice = Read-Answer '   请输入序号后回车' '0'

    switch ($choice) {
        '1' { Start-WebApp }
        '2' { Start-ApiOnly }
        '3' { Start-GradioApp }
        '4' { Start-Analyze }
        '5' { Start-Ask }
        '6' { Start-Library }
        '7' { Show-Check }
        '8' { Start-ModelMenu }
        '9' { Start-InitDb }
        'r' { $null = Get-Preflight -Force; Write-Good '预检已刷新'; Start-Sleep -Milliseconds 600 }
        'R' { $null = Get-Preflight -Force; Write-Good '预检已刷新'; Start-Sleep -Milliseconds 600 }
        '0' { $running = $false }
        'q' { $running = $false }
        'Q' { $running = $false }
        ''  { }
        default {
            Write-Warn ('无效输入：' + $choice)
            Start-Sleep -Seconds 1
        }
    }
}

Write-Host ''
Write-Good '再见'
exit 0
