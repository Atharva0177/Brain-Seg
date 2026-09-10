import torch

from pipeline.models.attention_unet3d import AttentionUNet3D
from pipeline.models.unet3d import UNetConfig


def test_attention_unet_forward_preserves_spatial_shape() -> None:
    model = AttentionUNet3D(UNetConfig(channels=(4, 8, 16), num_res_units=1))
    outputs = model(torch.randn(1, 4, 17, 19, 21))
    assert outputs.shape == (1, 4, 17, 19, 21)


def test_attention_unet_backward_produces_gradients() -> None:
    model = AttentionUNet3D(UNetConfig(channels=(4, 8, 16), num_res_units=1))
    loss = model(torch.randn(1, 4, 16, 16, 16)).square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
