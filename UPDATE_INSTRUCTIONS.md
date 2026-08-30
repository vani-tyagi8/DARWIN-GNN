# Updating an existing installation

This archive contains source/config/documentation only. It does not contain `.venv`,
`data`, or `runs`, so those existing directories are preserved.

From PowerShell, extract the update over the existing project:

```powershell
Expand-Archive -Force `
  -Path "$HOME\Downloads\devign-imbalance-research-v2.zip" `
  -DestinationPath "C:\devign-imbalance-project-clean"

Set-Location "C:\devign-imbalance-project-clean"
& ".\.venv\Scripts\python.exe" -m pip install -e ".[dev]"
& ".\.venv\Scripts\python.exe" -m pytest -q
```

Then enable CUDA and prepare Devign:

```powershell
& ".\scripts\setup_windows_gpu.ps1"
& ".\.venv\Scripts\python.exe" scripts\prepare_public_datasets.py --dataset devign
```

