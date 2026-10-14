# 固定安装包和摘要；静默安装仅放在本项目，不注册全局 Python，不修改 PATH。
function Install-ProjectConda {
    param([string]$Root)
    $destination = Join-Path $Root 'runtime/miniconda3'
    $installer = Join-Path $Root 'cache/downloads/Miniconda3-py313_25.7.0-2-Windows-x86_64.exe'
    New-Item -ItemType Directory -Force -Path (Split-Path $installer -Parent) | Out-Null
    $expected = '0505a3d4a7dab58ca621208e89efede4b6b8d817858b60d6eccc9303f2f3ebb0'
    if (-not (Test-Path -LiteralPath $installer)) {
        Write-ProjectLog '下载项目内 Miniconda（约 90 MB），不会安装到系统 Python。'
        Invoke-WebRequest -UseBasicParsing -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-py313_25.7.0-2-Windows-x86_64.exe' -OutFile $installer
    }
    if ((Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLower() -ne $expected) { throw 'Miniconda 安装包校验失败。' }
    if (-not (Test-Path -LiteralPath (Join-Path $destination 'Scripts/conda.exe'))) {
        $process = Start-Process -FilePath $installer -ArgumentList @('/InstallationType=JustMe','/RegisterPython=0','/AddToPath=0','/S',"/D=$destination") -Wait -PassThru -WindowStyle Hidden
        if ($process.ExitCode -ne 0) { throw '项目内 Miniconda 安装失败。' }
    }
    return (Join-Path $destination 'Scripts/conda.exe')
}
