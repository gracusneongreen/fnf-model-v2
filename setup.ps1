$ErrorActionPreference = "Stop"

Write-Host "Setting up FNF Model V2..." -ForegroundColor Cyan
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install Python 3.11 or 3.12 and enable the Python Launcher."
}

py -3.12 -m venv .venv
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Virtual environment creation failed."
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host "Setup complete. Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
