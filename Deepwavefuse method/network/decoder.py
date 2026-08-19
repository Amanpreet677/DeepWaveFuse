import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, in_channels=64):

        super().__init__()

        self.decoder = nn.Sequential(

            # 14 → 28
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 28 → 56
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 56 → 112
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 112 → 224
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # RGB
            nn.Conv2d(8, 3, kernel_size=3, padding=1),

            nn.Sigmoid()

        )

    def forward(self, x):

        return self.decoder(x)


if __name__ == "__main__":

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = Decoder().to(device)

    x = torch.randn(
        1,
        64,
        14,
        14,
        device=device
    )

    y = model(x)

    print("=" * 60)
    print("DeepWaveFuse Decoder")
    print("=" * 60)
    print("Input :", x.shape)
    print("Output:", y.shape)
