from __future__ import annotations
import torch
import torch.nn as nn


class DecisionMap(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        w = self.net(torch.cat([a, b], dim=1))
        return w, 1.0 - w
