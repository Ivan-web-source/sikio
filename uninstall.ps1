# Removes Sikio (shortcut, Start with Windows entry, program). Keeps your settings.
$ErrorActionPreference = 'Stop'
try {
    Remove-Item (Join-Path ([Environment]::GetFolderPath('Programs')) 'Sikio.lnk') -ErrorAction SilentlyContinue
    Remove-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' -Name Sikio -ErrorAction SilentlyContinue
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { $env:Path = "$env:USERPROFILE\.local\bin;$env:Path" }
    $tools = (uv tool dir | Out-String).Trim(); $bin = (uv tool dir --bin | Out-String).Trim()
    $running = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -like "*$tools\sikio\*" -or $_.CommandLine -like "*$bin\sikio-tray.exe*" -or $_.CommandLine -like "*$bin\sikio.exe*") })
    if ($running) { Write-Host 'Stopping the running Sikio ...'; $running | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; Start-Sleep -Seconds 1 }
    uv tool uninstall sikio
    if ($LASTEXITCODE) { throw 'uv tool uninstall failed (is Sikio installed?).' }
} catch {
    Write-Host "Uninstall failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
Write-Host "Sikio is uninstalled. Your settings are kept in $env:APPDATA\sikio (delete that folder to remove them)."
