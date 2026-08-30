# Project status

## Verified locally

- Modern PyTorch/PyG package installation in an isolated environment.
- Multi-edge GGNN and reference-faithful dual Conv/max-pooling classification head.
- Baseline and random-oversampling one-epoch runs on real public graph artifacts.
- Checkpoint, configuration, metrics, and per-sample prediction persistence.
- Precision, Recall, F1, MCC, PR-AUC, and confusion-matrix computation.
- Deterministic over/undersampling, focal loss, class-weighted BCE, and leakage checks.
- Eight automated tests.
- Resumable, checksum-verified public dataset acquisition from Zenodo.
- Streaming conversion and disk-backed indexed loading for multi-gigabyte Big-Vul data.

## Deliberately not claimed

The 994-function epicosy sample is AST-only. Its smoke-run metrics are software QA,
not research findings. Final findings require approved, provenance-recorded composite
graphs for Devign, ReVeal, and Big-Vul and completion of the frozen experiment matrix.

The local verification environment installed CPU PyTorch. `setup_windows_gpu.ps1`
installs and verifies the official CUDA wheel on the user's NVIDIA laptop. Google
Colab's existing CUDA-enabled PyTorch is preserved by `setup_colab.sh`.
