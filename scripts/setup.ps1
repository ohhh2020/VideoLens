param(
    [string]$ProjectRoot = (Split-Path $PSScriptRoot -Parent),
    [switch]$SkipModels,
    [switch]$SkipRuntime,
    [switch]$Minimal,
    [switch]$CheckOnly,
    [switch]$SkipVerification,
    [ValidateSet('3.13', '3.12')][string]$PythonVersion = '3.13'
)
. ([scriptblock]::Create((Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $ProjectRoot 'scripts/common.ps1'))))
Set-Location -LiteralPath $ProjectRoot
if (-not $CheckOnly) {
    . ([scriptblock]::Create((Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $ProjectRoot 'scripts/portable-conda.ps1'))))
    $conda = Install-ProjectConda -Root $ProjectRoot
} else { $conda = Find-Conda }
$environment = Get-VideoLensEnvironment -Conda $conda
if ($CheckOnly) {
    Write-ProjectLog "只检查模式：Conda=$conda；VideoLens=$environment"
    if (-not $environment) { throw '尚未创建 VideoLens 环境。正常配置会创建它；检查模式不执行安装。' }
    Invoke-Checked $conda @('run','--no-capture-output','-p',$script:VideoLensPrefix,'python','scripts/smoke_environment.py')
    return
}
$backupDir = Join-Path $ProjectRoot 'data/environment-backups'
New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
if ($environment) {
    Write-ProjectLog "发现已有 VideoLens 环境：$environment；先保存环境清单。"
    $snapshot = & $conda env export -p $script:VideoLensPrefix
    if ($LASTEXITCODE -ne 0) { throw '环境备份失败，已停止安装。' }
    $backup = Join-Path $backupDir (('VideoLens-{0}.yml' -f (Get-Date -Format 'yyyyMMdd-HHmmss')))
    [IO.File]::WriteAllText($backup, ($snapshot -join "`n"), [Text.UTF8Encoding]::new($false))
} else {
    Write-ProjectLog '创建 VideoLens 环境，首次安装需要联网。'
    Invoke-Checked $conda @('create','-y','-p',$script:VideoLensPrefix,'-c','conda-forge',"python=$PythonVersion",'pip')
}
# 用户显式指定 -PythonVersion 时才调整已存在环境；默认执行保留兼容的现有版本。
if ($environment -and $PSBoundParameters.ContainsKey('PythonVersion')) {
    $installedPython = & $conda run -p $script:VideoLensPrefix python -c 'import sys; print(chr(46).join(map(str,sys.version_info[:2])))'
    if ($LASTEXITCODE -ne 0) { throw '无法读取已有环境的 Python 版本。' }
    if (($installedPython -join '').Trim() -ne $PythonVersion) {
        Invoke-Checked $conda @('install','-y','-p',$script:VideoLensPrefix,'-c','conda-forge',"python=$PythonVersion",'pip')
    }
}
# 先装 Python 项目，确保 Conda 媒体包失败时还能调用独立静态包回退。
$extras = if ($Minimal) { '.[dev]' } else { '.[heavy,dev]' }
Write-ProjectLog "安装 Python 依赖：$extras"
$installArgs = @('run','--no-capture-output','-p',$script:VideoLensPrefix,'python','scripts/install_dependencies.py')
if ($Minimal) { $installArgs += '--minimal' }
& $conda @installArgs
if ($LASTEXITCODE -ne 0) {
    # 计划允许兼容性优先退回 3.12；先已导出原环境，不删除环境目录。
    if ($LASTEXITCODE -eq 42 -and $PythonVersion -eq '3.13') {
        Write-ProjectLog -Level 'WARNING' '依赖明确要求其他 Python 版本，按兼容方案切换为 Python 3.12 后重试。'
        Invoke-Checked $conda @('install','-y','-p',$script:VideoLensPrefix,'-c','conda-forge','python=3.12','pip')
        Invoke-Checked $conda $installArgs
    } else { throw '依赖安装失败。检查网络、代理及磁盘空间后重新执行；已有模型和成功安装的依赖可复用。' }
}
$ffmpegReady = $false
& $conda run --no-capture-output -p $script:VideoLensPrefix python scripts/smoke_environment.py
$ffmpegReady = ($LASTEXITCODE -eq 0)
if (-not $ffmpegReady) {
    # Conda post-link 失败可能回滚；不反复执行同一个失败事务。
    $environment = Get-VideoLensEnvironment -Conda $conda
    $historyPath = Join-Path $environment 'conda-meta/history'
    $historyBackup = if (Test-Path -LiteralPath $historyPath) { [IO.File]::ReadAllBytes($historyPath) } else { $null }
    & $conda install -y -p $script:VideoLensPrefix -c conda-forge 'ffmpeg>=7'
    if ($LASTEXITCODE -ne 0) {
        Write-ProjectLog -Level 'WARNING' 'Conda FFmpeg 安装失败，使用 FFmpeg 官网推荐的 Gyan 固定版本构建。'
        if ($historyBackup -and -not (Test-Path -LiteralPath $historyPath)) {
            [IO.File]::WriteAllBytes($historyPath, $historyBackup)
            Write-ProjectLog -Level 'WARNING' 'Conda 回滚删除了环境 history，已恢复安装前的原始文件。'
        }
        Invoke-Checked $conda @('run','--no-capture-output','-p',$script:VideoLensPrefix,'python','-m','videolens.bootstrap','ffmpeg')
    }
}
Invoke-Checked $conda @('run','--no-capture-output','-p',$script:VideoLensPrefix,'python','scripts/smoke_environment.py')
if (-not $Minimal) {
    if (-not $SkipModels) { Invoke-Checked $conda @('run','--no-capture-output','-p',$script:VideoLensPrefix,'python','-m','videolens.bootstrap','models') }
    if (-not $SkipRuntime) { Invoke-Checked $conda @('run','--no-capture-output','-p',$script:VideoLensPrefix,'python','-m','videolens.bootstrap','runtime') }
}
$fullVerification = (-not $Minimal -and -not $SkipModels -and -not $SkipRuntime -and -not $SkipVerification)
if ($fullVerification) {
    Write-ProjectLog '完整部署需要真实 CPU 推理通过；现在下载短小公开语音样本并逐项验证模型。'
    Invoke-Checked $conda @('run','--no-capture-output','-p',$script:VideoLensPrefix,'python','scripts/download_test_audio.py')
    Invoke-Checked $conda @('run','--no-capture-output','-p',$script:VideoLensPrefix,'python','scripts/verify_real_models.py','--audio','data/fixtures/public-audio/funasr_zh.wav','--output','data/verification/setup-real-models.json')
} else {
    Write-ProjectLog -Level 'WARNING' '本次选择了部分配置/跳过验证：只确认基础环境，不能据此判断所有本地模型已经通过真实推理。'
}
$freeze = & $conda run -p $script:VideoLensPrefix python -m pip freeze
if ($LASTEXITCODE -ne 0) { throw '依赖锁定清单生成失败。' }
[IO.File]::WriteAllText((Join-Path $backupDir 'requirements-installed.txt'), ($freeze -join "`n"), [Text.UTF8Encoding]::new($false))
if ($fullVerification) { Write-ProjectLog '完整配置与真实 CPU 模型验证通过。已记录实际安装版本，可以双击“一键启动.cmd”。' }
else { Write-ProjectLog '基础/部分配置完成。已记录实际安装版本；完整模型验收需运行未跳过步骤的配置脚本。' }
