# 共享脚本：通过 UTF-8 读取再执行，兼容 Windows PowerShell 5.1。
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:ANONYMIZED_TELEMETRY = 'False'
$env:HF_HUB_DISABLE_TELEMETRY = '1'

function Find-Conda {
    # 不把个人电脑的 D:\Environment 路径写死到程序，优先采用 conda 自身的位置。
    $candidates = @($env:CONDA_EXE)
    $command = Get-Command conda -ErrorAction SilentlyContinue
    if ($command) { $candidates += $command.Source }
    foreach ($base in @($env:USERPROFILE, $env:ProgramData, $env:LOCALAPPDATA)) {
        if ($base) {
            $candidates += (Join-Path $base 'miniconda3\Scripts\conda.exe')
            $candidates += (Join-Path $base 'anaconda3\Scripts\conda.exe')
            $candidates += (Join-Path $base 'miniconda3\condabin\conda.bat')
        }
    }
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) { return $candidate }
    }
    throw '未找到 Miniconda。请先安装 Miniconda，或在 Miniconda Prompt 中运行脚本；也可设置 CONDA_EXE 为 conda.exe 的完整路径。'
}

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "命令失败（退出码 $LASTEXITCODE）：$Program $($Arguments -join ' ')" }
}

function Get-VideoLensEnvironment {
    param([string]$Conda)
    $json = & $Conda env list --json
    if ($LASTEXITCODE -ne 0) { throw '无法读取 conda 环境列表。' }
    $info = ($json -join "`n") | ConvertFrom-Json
    return @($info.envs | Where-Object { (Split-Path $_ -Leaf) -eq 'VideoLens' }) | Select-Object -First 1
}
