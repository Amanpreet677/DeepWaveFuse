from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, out_ch)), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, out_ch)), out_ch),
            nn.GELU(),
        )
    def forward(self, x):
        return self.block(x)


class ReconstructionDecoder(nn.Module):
    """5-stage decoder: 16x16 -> 512x512 for 512x512 inputs."""
    def __init__(self, in_channels: int = 64, base: int = 128):
        super().__init__()
        self.c0 = ConvBlock(in_channels, base)
        self.c1 = ConvBlock(base, base // 2)
        self.c2 = ConvBlock(base // 2, base // 4)
        self.c3 = ConvBlock(base // 4, base // 8)
        self.c4 = ConvBlock(base // 8, base // 16)
        self.out = nn.Conv2d(base // 16, 3, 3, padding=1)

    def forward(self, x, target_size=(512, 512)):
        x = self.c0(x)
        for block in [self.c1, self.c2, self.c3, self.c4]:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.out(x))
