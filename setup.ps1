# Windows setup (PowerShell):  powershell -ExecutionPolicy Bypass -File setup.ps1
# Needs Python 3.10-3.13 and, for GPU training, an NVIDIA driver.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
Write-Host "Done. Check it with: .\.venv\Scripts\python -m pytest tests"
