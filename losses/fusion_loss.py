from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image_ops import focus_map_pair, gradient_magnitude, to_gray


def ssim_index(x: torch.Tensor, y: torch.Tensor, window: int = 11) -> torch.Tensor:
    pad = window // 2
    mu_x = F.avg_pool2d(x, window, 1, pad)
    mu_y = F.avg_pool2d(y, window, 1, pad)
    sigma_x = F.avg_pool2d(x*x, window, 1, pad) - mu_x*mu_x
    sigma_y = F.avg_pool2d(y*y, window, 1, pad) - mu_y*mu_y
    sigma_xy = F.avg_pool2d(x*y, window, 1, pad) - mu_x*mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2*mu_x*mu_y+c1)*(2*sigma_xy+c2)) / ((mu_x*mu_x+mu_y*mu_y+c1)*(sigma_x+sigma_y+c2)+1e-8)
    return ssim.clamp(0, 1).mean()


class FusionLoss(nn.Module):
    def __init__(self, lambda_pseudo=1.0, lambda_ssim=1.0, lambda_grad=0.5, lambda_wavelet=0.5, lambda_perceptual=0.1):
        super().__init__()
        self.l1 = lambda_pseudo
        self.lssim = lambda_ssim
        self.lgrad = lambda_grad
        self.lwave = lambda_wavelet
        self.lperc = lambda_perceptual

    def _pseudo_target(self, a, b):
        wa, wb = focus_map_pair(a, b, kernel=5)
        target = wa*a + wb*b
        return target.detach(), wa.detach(), wb.detach()

    def forward(self, fused, a, b, target=None, wavelet_image=None):
        if target is None:
            ref, wa, wb = self._pseudo_target(a, b)
        else:
            ref = target
            wa, wb = focus_map_pair(a, b, kernel=5)

        l1 = F.l1_loss(fused, ref)
        lssim = 1.0 - ssim_index(fused, ref)

        gf = gradient_magnitude(fused)
        ga = gradient_magnitude(a)
        gb = gradient_magnitude(b)
        desired_grad = torch.maximum(ga, gb)
        lgrad = F.l1_loss(gf, desired_grad)

        # Frequency/detail surrogate: compare local high-frequency residual energy.
        hf_f = fused - F.avg_pool2d(fused, 3, 1, 1)
        hf_a = a - F.avg_pool2d(a, 3, 1, 1)
        hf_b = b - F.avg_pool2d(b, 3, 1, 1)
        desired_hf = torch.where(hf_a.abs() >= hf_b.abs(), hf_a, hf_b)
        lwave = F.l1_loss(hf_f, desired_hf.detach())

        # Lightweight perceptual proxy using local gradient + luminance structure.
        lperc = F.l1_loss(to_gray(fused), to_gray(ref))

        total = self.l1*l1 + self.lssim*lssim + self.lgrad*lgrad + self.lwave*lwave + self.lperc*lperc
        return {"total": total, "l1": l1, "ssim": lssim, "gradient": lgrad, "wavelet": lwave, "perceptual": lperc}
