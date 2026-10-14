# 共享脚本：通过 UTF-8 读取再执行，兼容 Windows PowerShell 5.1。
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:ANONYMIZED_TELEMETRY = 'False'
$env:HF_HUB_DISABLE_TELEMETRY = '1'

# 所有应用依赖与缓存均按项目定位，绝不按同名环境误选外部目录。
$script:VideoLensRoot = [IO.Path]::GetFullPath($ProjectRoot)
$script:VideoLensPrefix = Join-Path $script:VideoLensRoot 'envs/VideoLens'
$locations = @{
    TEMP='cache/tmp'; TMP='cache/tmp'; TMPDIR='cache/tmp';
    CONDA_PKGS_DIRS='cache/conda/pkgs'; CONDA_ENVS_PATH='envs';
    PIP_CACHE_DIR='cache/pip'; HF_HOME='cache/huggingface';
    HF_HUB_CACHE='models/huggingface'; HF_XET_CACHE='cache/huggingface/xet';
    TORCH_HOME='cache/torch'; SENTENCE_TRANSFORMERS_HOME='models/sentence-transformers';
    GRADIO_TEMP_DIR='cache/gradio'; XDG_CACHE_HOME='cache';
    CUDA_CACHE_PATH='cache/cuda'; MPLCONFIGDIR='cache/matplotlib';
    NUMBA_CACHE_DIR='cache/numba'; TRITON_CACHE_DIR='cache/triton';
    HOME='cache/home'; USERPROFILE='cache/home'; APPDATA='cache/appdata'; LOCALAPPDATA='cache/localappdata'
}
foreach ($entry in $locations.GetEnumerator()) {
    $directory = Join-Path $script:VideoLensRoot $entry.Value
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    [Environment]::SetEnvironmentVariable($entry.Key, $directory, 'Process')
}
$env:VIDEOLENS_ROOT = $script:VideoLensRoot
$env:PYTHONNOUSERSITE = '1'
$env:PIP_CONFIG_FILE = 'NUL'
$env:CONDARC = Join-Path $script:VideoLensRoot 'cache/condarc'
if (-not (Test-Path -LiteralPath $env:CONDARC)) {
    [IO.File]::WriteAllText($env:CONDARC, "channels:`n  - conda-forge`nchannel_priority: strict`n", [Text.UTF8Encoding]::new($false))
}
function Find-Conda {
    $candidate = Join-Path $script:VideoLensRoot 'runtime/miniconda3/Scripts/conda.exe'
    if (Test-Path -LiteralPath $candidate) { return $candidate }
    throw '未找到项目内 Miniconda。请先运行一键配置环境。'
}

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "命令失败（退出码 $LASTEXITCODE）：$Program $($Arguments -join ' ')" }
}

function Get-VideoLensEnvironment {
    param([string]$Conda)
    if (Test-Path -LiteralPath (Join-Path $script:VideoLensPrefix 'python.exe')) { return $script:VideoLensPrefix }
    return $null
}

function Write-ProjectLog {
    param([Parameter(Position=0)][string]$Message, [string]$Level='INFO')
    foreach ($line in ($Message -split "`r?`n")) {
        Write-Host ('[{0}][{1}][启动/配置]: {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $line)
    }
}
