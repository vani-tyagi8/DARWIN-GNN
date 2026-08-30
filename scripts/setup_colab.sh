#!/usr/bin/env bash
set -euo pipefail

# Colab already supplies a CUDA-enabled PyTorch build. Installing this project must
# not replace it with a CPU wheel.
python -m pip install -q numpy pandas matplotlib ijson requests pyyaml scikit-learn pytest torch-geometric
python -m pip install -q --no-deps -e .
python - <<'PY'
import torch
import torch_geometric
print("PyTorch:", torch.__version__)
print("PyG:", torch_geometric.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
