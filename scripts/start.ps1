param([string]$ProjectRoot = (Split-Path $PSScriptRoot -Parent), [switch]$NoBrowser, [switch]$SkipLocal, [switch]$CheckOnly)
. ([scriptblock]::Create((Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $ProjectRoot 'scripts/common.ps1'))))
Set-Location -LiteralPath $ProjectRoot
$conda = Find-Conda
if (-not (Get-VideoLensEnvironment $conda)) { throw '未找到 VideoLens 环境，请先运行“一键配置环境.cmd”。' }
$localStarted = $false
$localOwner = [Guid]::NewGuid().ToString('N')
try {
    Invoke-Checked $conda @('run','--no-capture-output','-n','VideoLens','python','scripts/smoke_environment.py')
    if ($CheckOnly) { Write-Host '启动前环境检查通过；检查模式不启动网页或模型进程。'; return }
    if (-not $SkipLocal) {
        # 模型尚未部署时允许主程序启动，问答页面会明确显示基础摘录降级。
        & $conda run --no-capture-output -n VideoLens python -m videolens.bootstrap start-local --owner $localOwner
        $localStarted = ($LASTEXITCODE -eq 0)
        if (-not $localStarted) { Write-Warning '本地生成模型未就绪；继续启动网页，可在部署后重试。' }
    }
    $serveArgs = @('run','--no-capture-output','-n','VideoLens','python','-m','videolens.cli','serve')
    if (-not $NoBrowser) { $serveArgs += '--open-browser' }
    Invoke-Checked $conda $serveArgs
} finally {
    if ($localStarted) { & $conda run --no-capture-output -n VideoLens python -m videolens.bootstrap stop-local --owner $localOwner }
}
