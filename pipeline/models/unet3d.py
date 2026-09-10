"""Configurable baseline 3D U-Net for BraTS segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class UNetConfig:
    in_channels: int = 4
    out_channels: int = 4
    channels: tuple[int, ...] = (16, 32, 64, 128)
    num_res_units: int = 2
    negative_slope: float = 0.01

    def validate(self) -> None:
        if self.in_channels < 1 or self.out_channels < 2:
            raise ValueError("UNet requires at least one input and two output channels.")
        if len(self.channels) < 2 or any(channel < 1 for channel in self.channels):
            raise ValueError("UNet channels must contain at least two positive widths.")
        if self.num_res_units < 1:
            raise ValueError("num_res_units must be positive.")


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, negative_slope: float) -> None:
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )


class DoubleConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, units: int, negative_slope: float
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(units):
            layers.append(
                ConvNormAct(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    negative_slope,
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class UNet3D(nn.Module):
    """3D U-Net returning multiclass logits shaped (N, 4, X, Y, Z)."""

    def __init__(self, config: UNetConfig = UNetConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        widths = config.channels
        self.encoders = nn.ModuleList(
            [
                DoubleConv(
                    config.in_channels if index == 0 else widths[index],
                    width,
                    config.num_res_units,
                    config.negative_slope,
                )
                for index, width in enumerate(widths)
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Conv3d(widths[index], widths[index + 1], 2, stride=2)
                for index in range(len(widths) - 1)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                DoubleConv(
                    2 * widths[index - 1],
                    widths[index - 1],
                    config.num_res_units,
                    config.negative_slope,
                )
                for index in range(len(widths) - 1, 0, -1)
            ]
        )
        self.upsamples = nn.ModuleList(
            [
                nn.ConvTranspose3d(widths[index], widths[index - 1], kernel_size=2, stride=2)
                for index in range(len(widths) - 1, 0, -1)
            ]
        )
        self.head = nn.Conv3d(widths[0], config.out_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != self.config.in_channels:
            raise ValueError(
                f"Expected input (N, {self.config.in_channels}, X, Y, Z), got {tuple(inputs.shape)}"
            )
        skips: list[Tensor] = []
        current = inputs
        for index, encoder in enumerate(self.encoders):
            current = encoder(current)
            skips.append(current)
            if index < len(self.downsamples):
                current = self.downsamples[index](current)

        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips[:-1])):
            current = upsample(current)
            if current.shape[2:] != skip.shape[2:]:
                current = F.interpolate(
                    current, size=skip.shape[2:], mode="trilinear", align_corners=False
                )
            current = decoder(torch.cat((current, skip), dim=1))
        return self.head(current)
