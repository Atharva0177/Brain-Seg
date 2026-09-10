import torch
from torch import nn

from pipeline.models.inference import (
    gaussian_importance_map,
    generate_windows,
    sliding_window_logits,
)


class IdentityLogits(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat((inputs[:, :1], inputs[:, :1], inputs[:, :1], inputs[:, :1]), dim=1)


def test_windows_cover_edge_and_gaussian_map_is_positive() -> None:
    windows = generate_windows((10, 11, 12), (6, 6, 6), overlap=0.5)
    assert any(window.starts == (4, 5, 6) for window in windows)
    importance = gaussian_importance_map((6, 6, 6))
    assert tuple(importance.shape) == (6, 6, 6)
    assert bool(torch.all(importance > 0))


def test_sliding_window_reconstructs_constant_field() -> None:
    inputs = torch.ones(1, 1, 9, 10, 11)
    output = sliding_window_logits(
        IdentityLogits(), inputs, window_size=(6, 6, 6), overlap=0.5, amp=False
    )
    assert tuple(output.shape) == (1, 4, 9, 10, 11)
    assert torch.allclose(output, torch.ones_like(output))
