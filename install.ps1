# Installs Sikio for the current user. Set $env:SIKIO_SOURCE to a wheel path or git URL to install that instead of PyPI.
$ErrorActionPreference = 'Stop'
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host 'Installing uv (https://astral.sh/uv) ...'
        irm https://astral.sh/uv/install.ps1 | iex
        $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv was installed but not found; open a new terminal and run this again.' }
    }
    $source = if ($env:SIKIO_SOURCE) { $env:SIKIO_SOURCE } else { 'sikio' }
    Write-Host 'Installing Sikio. The first time this downloads about 1.5 GB (Python and PyTorch), so it takes a while.'
    uv tool install --force --python 3.12 $source
    if ($LASTEXITCODE) { throw 'uv tool install failed.' }
    uv tool update-shell
    $bin = (uv tool dir --bin | Out-String).Trim()
    $target = Join-Path $bin 'sikio-tray.exe'
    if (-not (Test-Path $target)) { throw "sikio-tray.exe not found in $bin." }
    $lnk = Join-Path ([Environment]::GetFolderPath('Programs')) 'Sikio.lnk'
    $s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
    $s.TargetPath = $target
    $s.WorkingDirectory = $bin
    $s.Description = 'Sikio voice assistant'
    $s.Save()
} catch {
    Write-Host "Install failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
Write-Host ''
Write-Host 'Sikio is installed. Next:'
Write-Host '  - Start "Sikio" from the Start menu. The first start opens its settings.'
Write-Host '  - Run this installer again to upgrade.'
Write-Host '  - To uninstall: powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/OWNER/sikio/main/uninstall.ps1 | iex"'
