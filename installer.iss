; Online installer: ships only the two PowerShell scripts, which install uv and Sikio from PyPI.
; AppVersion must match `version` in pyproject.toml.
[Setup]
AppId={{B6E1C6A4-5D0B-4F3E-9A57-3C1D8E2F7A41}
AppName=Sikio
AppVersion=0.1.0
AppPublisher=Albertus Ivan Wijaya
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\Sikio
DisableDirPage=yes
DisableProgramGroupPage=yes
LicenseFile=LICENSE
SetupIconFile=sikio.ico
UninstallDisplayIcon={app}\sikio.ico
OutputDir=dist
OutputBaseFilename=Sikio-Setup

[Files]
Source: "install.ps1"; DestDir: "{tmp}"; Flags: dontcopy
Source: "install.ps1"; DestDir: "{app}"
Source: "uninstall.ps1"; DestDir: "{app}"
Source: "sikio.ico"; DestDir: "{app}"

[Run]
; The first start opens the settings window.
Filename: "{userprograms}\Sikio.lnk"; Description: "Start Sikio"; Flags: postinstall shellexec skipifsilent

[UninstallRun]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\uninstall.ps1"""; RunOnceId: "UninstallSikio"; Flags: waituntilterminated

[Code]
// Runs the install before Setup copies anything or registers the uninstaller, so a failure
// leaves no Apps-list entry behind. A returned message aborts Setup (exit code 8 when silent).
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
begin
  ExtractTemporaryFile('install.ps1');
  // Visible console, so the uv download progress shows.
  if (not Exec('powershell.exe', '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{tmp}\install.ps1') + '"',
      '', SW_SHOWNORMAL, ewWaitUntilTerminated, ResultCode)) or (ResultCode <> 0) then
    Result := 'Sikio could not be installed. Check your internet connection and run this installer again.';
end;
