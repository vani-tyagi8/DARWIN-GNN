from __future__ import annotations

import numpy as np

STRATEGIES = {"baseline", "random_oversampling", "random_undersampling",
              "class_weighted", "focal_loss"}


def sampled_indices(labels, strategy: str, seed: int) -> list[int]:
    """Return graph indices; validation/test data must never call this function."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    labels = np.asarray(labels, dtype=np.int64)
    indices = np.arange(labels.size)
    if strategy not in {"random_oversampling", "random_undersampling"}:
        return indices.tolist()
    class_zero = indices[labels == 0]
    class_one = indices[labels == 1]
    if not len(class_zero) or not len(class_one):
        raise ValueError("sampling requires both classes")
    rng = np.random.default_rng(seed)
    if strategy == "random_oversampling":
        target = max(len(class_zero), len(class_one))
        selected = np.concatenate([
            rng.choice(class_zero, target, replace=len(class_zero) < target),
            rng.choice(class_one, target, replace=len(class_one) < target),
        ])
    else:
        target = min(len(class_zero), len(class_one))
        selected = np.concatenate([
            rng.choice(class_zero, target, replace=False),
            rng.choice(class_one, target, replace=False),
        ])
    rng.shuffle(selected)
    return selected.tolist()

