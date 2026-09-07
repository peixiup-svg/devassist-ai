$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExecutable = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Run scripts/setup.ps1 first."
}

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
& $PythonExecutable -m uvicorn devassist.api.app:app --host 127.0.0.1 --port 8000 --reload

