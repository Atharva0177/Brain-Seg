import pytest
import torch

from pipeline.models.losses import DiceCrossEntropyLoss, soft_dice_loss


def test_loss_is_finite_and_has_gradient() -> None:
    logits = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
    targets = torch.randint(0, 4, (1, 8, 8, 8))
    loss = DiceCrossEntropyLoss()(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_perfect_confident_logits_have_low_dice_loss() -> None:
    targets = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    targets[:, 1:3, 1:3, 1:3] = 1
    targets[:, 0, 0, 0] = 2
    targets[:, 0, 0, 1] = 3
    logits = torch.full((1, 4, 4, 4, 4), -10.0)
    logits.scatter_(1, targets.unsqueeze(1), 10.0)
    assert soft_dice_loss(logits, targets) < 1e-3


def test_invalid_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="not both zero"):
        DiceCrossEntropyLoss(dice_weight=0, cross_entropy_weight=0)
