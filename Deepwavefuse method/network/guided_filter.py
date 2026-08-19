import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedFilter(nn.Module):

    def __init__(self, radius=3):
        super().__init__()

        self.radius = radius

    def forward(self, guide, src):

        kernel = 2 * self.radius + 1

        guide_mean = F.avg_pool2d(
            guide,
            kernel,
            stride=1,
            padding=self.radius
        )

        src_mean = F.avg_pool2d(
            src,
            kernel,
            stride=1,
            padding=self.radius
        )

        output = src + (guide - guide_mean)

        output = F.avg_pool2d(
            output,
            kernel,
            stride=1,
            padding=self.radius
        )

        return output


if __name__ == "__main__":

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = GuidedFilter().to(device)

    guide = torch.randn(
        1,
        64,
        14,
        14,
        device=device
    )

    src = torch.randn(
        1,
        64,
        14,
        14,
        device=device
    )

    out = model(
        guide,
        src
    )

    print("=" * 60)
    print("Differentiable Guided Filter")
    print("=" * 60)
    print(out.shape)
