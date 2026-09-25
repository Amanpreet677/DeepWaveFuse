from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from models.encoder import VGG19Encoder
from models.attention import SymmetricCrossAttention, FocusAttention
from models.wavelet import MeyerWaveletBranch
from models.decoder import ReconstructionDecoder
from models.guided_filter import DifferentiableGuidedFilter
from models.decision import DecisionMap


class DeepWaveFuse(nn.Module):
    """DeepWaveFuse: VGG-19 + cross attention + exact 3-level Meyer fusion + decoder."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = VGG19Encoder(
            cfg.vgg_weights,
            train_last_block=cfg.vgg_train_last_block,
            train_last_two_blocks=cfg.vgg_train_last_two_blocks,
        )
        self.cross = SymmetricCrossAttention(512, cfg.attention_heads)
        self.proj = nn.Sequential(
            nn.Conv2d(512, cfg.feature_channels, 1, bias=False),
            nn.GroupNorm(8, cfg.feature_channels),
            nn.GELU(),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(512, cfg.feature_channels, 1, bias=False),
            nn.GroupNorm(8, cfg.feature_channels),
            nn.GELU(),
        )
        self.focus = FocusAttention(cfg.feature_channels)
        self.decision = DecisionMap(cfg.feature_channels)
        self.wavelet = MeyerWaveletBranch(cfg.wavelet, cfg.wavelet_level, cfg.wavelet_mode, cfg.soft_high_frequency)
        self.semantic_fuse = nn.Sequential(
            nn.Conv2d(cfg.feature_channels * 3, cfg.feature_channels, 3, padding=1),
            nn.GroupNorm(8, cfg.feature_channels),
            nn.GELU(),
            nn.Conv2d(cfg.feature_channels, cfg.feature_channels, 3, padding=1),
            nn.GELU(),
        )
        self.decoder = ReconstructionDecoder(cfg.feature_channels, cfg.decoder_base)
        self.alpha = nn.Sequential(
            nn.Conv2d(cfg.feature_channels, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        self.refiner = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh(),
        ) if cfg.residual_refine else None
        self.guided = DifferentiableGuidedFilter(cfg.guided_radius, cfg.guided_eps)

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> dict[str, torch.Tensor]:
        target_size = image_a.shape[-2:]
        fa = self.encoder(image_a)
        fb = self.encoder(image_b)

        ca, cb = self.cross(fa["relu5"], fb["relu5"])
        pa, pb = self.proj(ca), self.proj(cb)
        skip_a, skip_b = self.skip_proj(fa["relu4"]), self.skip_proj(fb["relu4"])
        skip_a = F.interpolate(skip_a, size=pa.shape[-2:], mode="bilinear", align_corners=False)
        skip_b = F.interpolate(skip_b, size=pb.shape[-2:], mode="bilinear", align_corners=False)

        wa16, wb16 = self.focus(pa, pb)
        # Decision maps refine deep features before decoding.
        da, db = self.decision(pa + skip_a, pb + skip_b)
        fused_deep = da * pa + db * pb
        fused_deep = self.semantic_fuse(torch.cat([fused_deep, pa * wa16, pb * wb16], dim=1))

        # Convert attention to image space for the exact Meyer branch.
        wa_img = F.interpolate(wa16, size=target_size, mode="bilinear", align_corners=False)
        wb_img = 1.0 - wa_img
        wavelet_fused = self.wavelet(image_a, image_b, wa_img, wb_img).clamp(0, 1)

        deep_fused = self.decoder(fused_deep, target_size=target_size)
        alpha = F.interpolate(self.alpha(fused_deep), size=target_size, mode="bilinear", align_corners=False)
        final = alpha * wavelet_fused + (1.0 - alpha) * deep_fused

        if self.refiner is not None:
            residual = 0.1 * self.refiner(torch.cat([final, deep_fused], dim=1))
            final = (final + residual).clamp(0, 1)

        # Guided filter is used as a conservative residual correction.
        guided = self.guided(final, final)
        final = (0.75 * final + 0.25 * guided).clamp(0, 1)

        return {
            "image": final,
            "deep_image": deep_fused,
            "wavelet_image": wavelet_fused,
            "weight_a": wa_img,
            "weight_b": wb_img,
            "alpha": alpha,
        }
