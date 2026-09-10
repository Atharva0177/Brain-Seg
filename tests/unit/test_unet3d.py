import pytest
import torch

from pipeline.models.unet3d import UNet3D, UNetConfig


def test_unet_forward_preserves_spatial_shape() -> None:
    model = UNet3D(UNetConfig(channels=(4, 8, 16), num_res_units=1))
    inputs = torch.randn(1, 4, 17, 19, 21)
    outputs = model(inputs)
    assert outputs.shape == (1, 4, 17, 19, 21)


def test_unet_rejects_wrong_channel_count() -> None:
    model = UNet3D(UNetConfig(channels=(4, 8, 16), num_res_units=1))
    with pytest.raises(ValueError, match="Expected input"):
        model(torch.randn(1, 3, 16, 16, 16))


def test_unet_config_rejects_invalid_channels() -> None:
    with pytest.raises(ValueError, match="at least two"):
        UNet3D(UNetConfig(channels=(8,)))
