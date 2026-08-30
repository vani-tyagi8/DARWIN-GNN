from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss operating on logits."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if alpha is not None and not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        loss = (1 - p_t).pow(self.gamma) * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()


def make_loss(strategy: str, labels: Tensor, focal_gamma: float = 2.0,
              focal_alpha: float | None = None) -> nn.Module:
    if strategy == "focal_loss":
        return FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    if strategy == "class_weighted":
        positives = labels.sum().item()
        negatives = labels.numel() - positives
        if positives == 0:
            raise ValueError("class-weighted loss requires at least one positive sample")
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives]))
    return nn.BCEWithLogitsLoss()

