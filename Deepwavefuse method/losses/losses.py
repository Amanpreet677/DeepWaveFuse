import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class L1Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, prediction, target):
        return self.loss(prediction, target)


class SSIMLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):

        mu_x = F.avg_pool2d(prediction, 3, 1, 1)
        mu_y = F.avg_pool2d(target, 3, 1, 1)

        sigma_x = F.avg_pool2d(prediction * prediction, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(prediction * target, 3, 1, 1) - mu_x * mu_y

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim = (
            (2 * mu_x * mu_y + c1) *
            (2 * sigma_xy + c2)
        ) / (
            (mu_x ** 2 + mu_y ** 2 + c1) *
            (sigma_x + sigma_y + c2)
        )

        return 1.0 - ssim.mean()


class PerceptualLoss(nn.Module):

    def __init__(self):

        super().__init__()

        vgg = vgg19(weights=None)

        state_dict = torch.load(
            "./pretrained/vgg19-dcbb9e9d.pth",
            map_location="cpu"
        )

        vgg.load_state_dict(state_dict)

        self.network = vgg.features[:16]

        for p in self.network.parameters():
            p.requires_grad = False

        self.network.eval()

    def forward(self, prediction, target):

        f1 = self.network(prediction)
        f2 = self.network(target)

        return F.l1_loss(f1, f2)


class TotalLoss(nn.Module):

    def __init__(
        self,
        alpha=1.0,
        beta=0.5,
        gamma=0.2
    ):

        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.l1 = L1Loss()
        self.ssim = SSIMLoss()
        self.perceptual = PerceptualLoss()

    def forward(
        self,
        fused,
        image_a,
        image_b
    ):

        l1_a = self.l1(fused, image_a)
        l1_b = self.l1(fused, image_b)

        ssim_a = self.ssim(fused, image_a)
        ssim_b = self.ssim(fused, image_b)

        perceptual_a = self.perceptual(fused, image_a)
        perceptual_b = self.perceptual(fused, image_b)

        l1 = 0.5 * (l1_a + l1_b)

        ssim = 0.5 * (ssim_a + ssim_b)

        perceptual = 0.5 * (
            perceptual_a +
            perceptual_b
        )

        total = (
            self.alpha * l1 +
            self.beta * ssim +
            self.gamma * perceptual
        )

        return {
            "total": total,
            "l1": l1,
            "ssim": ssim,
            "perceptual": perceptual
        }


if __name__ == "__main__":

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    criterion = TotalLoss().to(device)

    fused = torch.rand(
        1,
        3,
        224,
        224,
        device=device
    )

    image_a = torch.rand(
        1,
        3,
        224,
        224,
        device=device
    )

    image_b = torch.rand(
        1,
        3,
        224,
        224,
        device=device
    )

    loss = criterion(
        fused,
        image_a,
        image_b
    )

    print("=" * 60)
    print("DeepWaveFuse Loss")
    print("=" * 60)

    for key, value in loss.items():
        print(f"{key:12s}: {value.item():.6f}")
