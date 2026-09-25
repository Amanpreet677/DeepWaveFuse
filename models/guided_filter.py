from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableGuidedFilter(nn.Module):
    """Standard grayscale-guide guided filter applied per RGB channel."""
    def __init__(self, radius: int = 4, eps: float = 1e-3):
        super().__init__()
        self.radius = radius
        self.eps = eps

    def box(self, x: torch.Tensor) -> torch.Tensor:
        k = 2 * self.radius + 1
        return F.avg_pool2d(x, k, stride=1, padding=self.radius, count_include_pad=True)

    def forward(self, guide: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        g = 0.299 * guide[:, 0:1] + 0.587 * guide[:, 1:2] + 0.114 * guide[:, 2:3]
        mean_g = self.box(g)
        mean_s = self.box(src)
        corr_gg = self.box(g * g)
        var_g = corr_gg - mean_g * mean_g
        a = (self.box(g * src) - mean_g * mean_s) / (var_g + self.eps)
        b = mean_s - a * mean_g
        mean_a = self.box(a)
        mean_b = self.box(b)
        return mean_a * g + mean_b
