"""Attention U-Net candidate for 3D BraTS segmentation."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .unet3d import DoubleConv, UNetConfig


class AttentionGate3D(nn.Module):
    """Additive attention gate using decoder context to filter skip features."""

    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.skip_projection = nn.Conv3d(skip_channels, inter_channels, 1, bias=False)
        self.gate_projection = nn.Conv3d(gate_channels, inter_channels, 1, bias=False)
        self.norm = nn.InstanceNorm3d(inter_channels, affine=True)
        self.activation = nn.LeakyReLU(0.01, inplace=True)
        self.attention = nn.Sequential(nn.Conv3d(inter_channels, 1, 1), nn.Sigmoid())

    def forward(self, skip: Tensor, gate: Tensor) -> Tensor:
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode="trilinear", align_corners=False)
        score = self.skip_projection(skip) + self.gate_projection(gate)
        score = self.attention(self.activation(self.norm(score)))
        return skip * score


class AttentionUNet3D(nn.Module):
    """3D Attention U-Net with the same four-class interface as UNet3D."""

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
        self.upsamples = nn.ModuleList(
            [
                nn.ConvTranspose3d(widths[index], widths[index - 1], 2, stride=2)
                for index in range(len(widths) - 1, 0, -1)
            ]
        )
        self.attention_gates = nn.ModuleList(
            [
                AttentionGate3D(
                    widths[index - 1], widths[index - 1], max(1, widths[index - 1] // 2)
                )
                for index in range(len(widths) - 1, 0, -1)
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
        self.head = nn.Conv3d(widths[0], config.out_channels, 1)

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

        for upsample, gate, decoder, skip in zip(
            self.upsamples, self.attention_gates, self.decoders, reversed(skips[:-1])
        ):
            current = upsample(current)
            filtered_skip = gate(skip, current)
            if current.shape[2:] != filtered_skip.shape[2:]:
                current = F.interpolate(
                    current, size=filtered_skip.shape[2:], mode="trilinear", align_corners=False
                )
            current = decoder(torch.cat((current, filtered_skip), dim=1))
        return self.head(current)
