$ErrorActionPreference = 'Stop'

$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Virtual environment not found. Run: py -m venv .venv'
}
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw 'nvidia-smi was not found. Install/update the NVIDIA driver first.'
}

Write-Host 'Detected NVIDIA GPU:'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# The official PyTorch wheel index currently publishes the matching Windows
# Python 3.12 CUDA 13.0 build. The wheel bundles the CUDA runtime.
& $python -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu130
& $python -m pip install --upgrade torch-geometric

& $python -c @'
import torch
print('PyTorch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA verification failed. Use the official PyTorch selector for a driver-compatible build.')
print('GPU:', torch.cuda.get_device_name(0))
'@

