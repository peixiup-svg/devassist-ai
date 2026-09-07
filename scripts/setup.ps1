$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VirtualEnvironment = Join-Path $ProjectRoot ".venv"

if (-not (Test-Path -LiteralPath $VirtualEnvironment)) {
    python -m venv $VirtualEnvironment
}

$PythonExecutable = Join-Path $VirtualEnvironment "Scripts\python.exe"
& $PythonExecutable -m pip install --upgrade pip
& $PythonExecutable -m pip install -e "${ProjectRoot}[dev]"

Write-Host "Environment ready: $PythonExecutable"
Write-Host "In VS Code, select .venv as the Python interpreter and press F5."
