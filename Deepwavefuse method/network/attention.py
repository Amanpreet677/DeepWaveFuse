"""
DeepWaveFuse V2
Multi-Head Cross Attention

Author: Amanpreet
"""

import torch
import torch.nn as nn


class CrossAttention(nn.Module):

    def __init__(
        self,
        channels=512,
        heads=8
    ):

        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(channels)

        self.ffn = nn.Sequential(

            nn.Linear(
                channels,
                channels * 4
            ),

            nn.GELU(),

            nn.Linear(
                channels * 4,
                channels
            )

        )

        self.norm2 = nn.LayerNorm(channels)

    def forward(self, feature_a, feature_b):

        B, C, H, W = feature_a.shape

        query = feature_a.flatten(2).transpose(1, 2)

        key = feature_b.flatten(2).transpose(1, 2)

        value = feature_b.flatten(2).transpose(1, 2)

        attention, _ = self.attention(
            query,
            key,
            value
        )

        x = self.norm1(
            query + attention
        )

        ffn = self.ffn(x)

        x = self.norm2(
            x + ffn
        )

        x = x.transpose(1, 2).reshape(
            B,
            C,
            H,
            W
        )

        return x


if __name__ == "__main__":

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = CrossAttention().to(device)

    x1 = torch.randn(
        1,
        512,
        14,
        14,
        device=device
    )

    x2 = torch.randn(
        1,
        512,
        14,
        14,
        device=device
    )

    output = model(
        x1,
        x2
    )

    print("=" * 60)
    print("DeepWaveFuse Multi-Head Cross Attention")
    print("=" * 60)

    print("Input A :", x1.shape)
    print("Input B :", x2.shape)
    print("Output  :", output.shape)
