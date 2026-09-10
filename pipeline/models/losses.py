"""Differentiable losses for four-class BraTS segmentation."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F


def soft_dice_loss(logits: Tensor, targets: Tensor, smooth: float = 1e-6) -> Tensor:
    if logits.ndim != 5 or targets.ndim != 4:
        raise ValueError("Expected logits (N,C,X,Y,Z) and targets (N,X,Y,Z).")
    if logits.shape[0] != targets.shape[0] or logits.shape[2:] != targets.shape[1:]:
        raise ValueError("Logits and targets have incompatible shapes.")
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(targets.long(), num_classes=logits.shape[1]).permute(0, 4, 1, 2, 3)
    probabilities = probabilities[:, 1:]
    one_hot = one_hot[:, 1:].to(dtype=probabilities.dtype)
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3, 4))
    denominator = probabilities.sum(dim=(0, 2, 3, 4)) + one_hot.sum(dim=(0, 2, 3, 4))
    dice = (2 * intersection + smooth) / (denominator + smooth)
    return 1 - dice.mean()


class DiceCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.5,
        cross_entropy_weight: float = 0.5,
        class_weights: Tensor | None = None,
    ) -> None:
        super().__init__()
        if dice_weight < 0 or cross_entropy_weight < 0 or dice_weight + cross_entropy_weight <= 0:
            raise ValueError("Loss weights must be nonnegative and not both zero.")
        self.dice_weight = dice_weight
        self.cross_entropy_weight = cross_entropy_weight
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        dice = soft_dice_loss(logits, targets)
        cross_entropy = F.cross_entropy(logits, targets.long(), weight=self.class_weights)
        total = self.dice_weight + self.cross_entropy_weight
        return (self.dice_weight * dice + self.cross_entropy_weight * cross_entropy) / total
