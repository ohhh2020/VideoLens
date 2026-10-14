param([string]$ProjectRoot = (Split-Path $PSScriptRoot -Parent), [switch]$RealModels)
. ([scriptblock]::Create((Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $ProjectRoot 'scripts/common.ps1'))))
Set-Location -LiteralPath $ProjectRoot
$conda = Find-Conda
Invoke-Checked $conda @('run','--no-capture-output','-p',$script:VideoLensPrefix,'python','scripts/smoke_environment.py')
# FFmpeg 可能安装在项目 runtime，而不在全局 PATH。把已验证的配置传给媒体
# 测试，防止一键测试因找不到命令而跳过本机其实具备条件的真实集成测试。
$mediaTools = & $conda run --no-capture-output -p $script:VideoLensPrefix python -c 'import json; from videolens.config import load_settings; s=load_settings(); print(json.dumps(dict(ffmpeg=s.ffmpeg,ffprobe=s.ffprobe)))'
if ($LASTEXITCODE -ne 0) { throw '读取媒体测试配置失败。' }
$mediaConfig = ($mediaTools -join "`n") | ConvertFrom-Json
$env:VIDEOLENS_TEST_FFMPEG = $mediaConfig.ffmpeg
$env:VIDEOLENS_TEST_FFPROBE = $mediaConfig.ffprobe
Invoke-Checked $conda @('run','--no-capture-output','-p',$script:VideoLensPrefix,'python','-m','pytest','-q','-m','not real_model and not cloud')
Invoke-Checked $conda @('run','--no-capture-output','-p',$script:VideoLensPrefix,'python','-m','pytest','-q','scripts/test_bootstrap.py')
if ($RealModels) {
    Invoke-Checked $conda @('run','--no-capture-output','-p',$script:VideoLensPrefix,'python','-m','videolens.bootstrap','verify-models')
    Invoke-Checked $conda @('run','--no-capture-output','-p',$script:VideoLensPrefix,'python','scripts/verify_real_models.py')
}
