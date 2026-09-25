from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricCrossAttention(nn.Module):
    """Bidirectional cross-attention on VGG-19 relu5 features."""
    def __init__(self, channels: int = 512, heads: int = 8):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.attn_ab = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.attn_ba = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm_a1 = nn.LayerNorm(channels)
        self.norm_b1 = nn.LayerNorm(channels)
        self.ffn_a = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels))
        self.ffn_b = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels))
        self.norm_a2 = nn.LayerNorm(channels)
        self.norm_b2 = nn.LayerNorm(channels)

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)

    def _image(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], h, w)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = a.shape
        ta, tb = self._tokens(a), self._tokens(b)
        ab, _ = self.attn_ab(ta, tb, tb)
        ba, _ = self.attn_ba(tb, ta, ta)
        xa = self.norm_a1(ta + ab)
        xb = self.norm_b1(tb + ba)
        xa = self.norm_a2(xa + self.ffn_a(xa))
        xb = self.norm_b2(xb + self.ffn_b(xb))
        return self._image(xa, H, W), self._image(xb, H, W)


class FocusAttention(nn.Module):
    """Produces paired soft focus weights from projected deep features."""
    def __init__(self, channels: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, 16, 1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(torch.cat([a, b], dim=1))
        wa = torch.sigmoid(logits)
        wb = 1.0 - wa
        return wa, wb
