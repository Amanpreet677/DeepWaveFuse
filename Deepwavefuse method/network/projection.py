import torch
import torch.nn as nn


class FeatureProjection(nn.Module):

    def __init__(self, in_channels=512, out_channels=64):
        super().__init__()

        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.project(x)


if __name__ == "__main__":

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = FeatureProjection().to(device)

    x = torch.randn(1, 512, 14, 14).to(device)

    y = model(x)

    print("=" * 60)
    print("Feature Projection")
    print("=" * 60)
    print("Input :", x.shape)
    print("Output:", y.shape)