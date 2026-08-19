"""
DeepWaveFuse
Adaptive Frequency Fusion

Author : Amanpreet
"""

import torch
import torch.nn as nn


class AdaptiveWeightGenerator(nn.Module):

    def __init__(self, channels=64):

        super().__init__()

        self.weight = nn.Sequential(

            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False
            ),

            nn.Sigmoid()

        )

    def forward(self, ll_a, ll_b):

        feature = torch.cat(
            [ll_a, ll_b],
            dim=1
        )

        alpha = self.weight(feature)

        fused = alpha * ll_a + (1.0 - alpha) * ll_b

        return fused


class HighFrequencyFusion(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, feat_a, feat_b):

        mask = torch.abs(feat_a) >= torch.abs(feat_b)

        return torch.where(
            mask,
            feat_a,
            feat_b
        )


class AdaptiveFusion(nn.Module):

    def __init__(self, channels=64):

        super().__init__()

        self.low = AdaptiveWeightGenerator(channels)
        self.high = HighFrequencyFusion()

    def forward(self, coeff_a, coeff_b):

        fused = {

            "LL": self.low(
                coeff_a["LL"],
                coeff_b["LL"]
            ),

            "LH": self.high(
                coeff_a["LH"],
                coeff_b["LH"]
            ),

            "HL": self.high(
                coeff_a["HL"],
                coeff_b["HL"]
            ),

            "HH": self.high(
                coeff_a["HH"],
                coeff_b["HH"]
            )

        }

        return fused
if __name__ == "__main__":

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = AdaptiveFusion(channels=64).to(device)

    coeff_a = {
        "LL": torch.randn(1, 64, 7, 7, device=device),
        "LH": torch.randn(1, 64, 7, 7, device=device),
        "HL": torch.randn(1, 64, 7, 7, device=device),
        "HH": torch.randn(1, 64, 7, 7, device=device),
    }

    coeff_b = {
        "LL": torch.randn(1, 64, 7, 7, device=device),
        "LH": torch.randn(1, 64, 7, 7, device=device),
        "HL": torch.randn(1, 64, 7, 7, device=device),
        "HH": torch.randn(1, 64, 7, 7, device=device),
    }

    fused = model(coeff_a, coeff_b)

    print("=" * 60)
    print("DeepWaveFuse Final Fusion")
    print("=" * 60)

    for key, value in fused.items():
        print(f"{key} : {value.shape}")
