import torch
import torch.nn as nn


class DecisionMap(nn.Module):

    def __init__(self, channels=64):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(channels, 32, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, 1),

            nn.Sigmoid()

        )

    def forward(self, feature):

        weight = self.net(feature)

        fused = feature * weight

        return fused, weight


if __name__ == "__main__":

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = DecisionMap().to(device)

    x = torch.randn(
        1,
        64,
        14,
        14,
        device=device
    )

    fused, weight = model(x)

    print("=" * 60)
    print("DeepWaveFuse Decision Map")
    print("=" * 60)

    print("Input :", x.shape)
    print("Weight:", weight.shape)
    print("Output:", fused.shape)