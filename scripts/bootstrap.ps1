param(
    [string]$PythonExe = "python",
    [string]$VenvName = ".venv",
    [switch]$EnableCuda
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venvPath = Join-Path $root $VenvName
$venvPython = Join-Path $venvPath "Scripts/python.exe"

if (-not (Test-Path $venvPath)) {
    & $PythonExe -m venv $venvPath
}

& $venvPython -m pip install --upgrade pip

if ($EnableCuda) {
    & $venvPython -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
} else {
    & $venvPython -m pip install torch torchvision
}

& $venvPython -m pip install -r requirements.txt

Write-Host "Bootstrap complete."
Write-Host "Activate with: ./$VenvName/Scripts/Activate.ps1"
