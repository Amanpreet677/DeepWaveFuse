from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import vgg19


class VGG19Encoder(nn.Module):
    """ImageNet-pretrained VGG-19 feature extractor.

    relu4: output after VGG block 4 (512 channels, H/16, W/16)
    relu5: output after VGG block 5 (512 channels, H/32, W/32)
    """
    def __init__(self, weights_path: str, train_last_block: bool = True, train_last_two_blocks: bool = False):
        super().__init__()
        self.vgg = vgg19(weights=None)
        weights = Path(weights_path)
        if not weights.exists():
            raise FileNotFoundError(
                f"VGG-19 weights not found: {weights}. "
                "Place vgg19-dcbb9e9d.pth in pretrained/ or set vgg_weights in config."
            )
        state = torch.load(weights, map_location="cpu")
        self.vgg.load_state_dict(state)

        # Freeze everything first.
        for p in self.vgg.features.parameters():
            p.requires_grad = False

        # VGG feature blocks: block4 ~ indices 19-27, block5 ~ 28-36.
        if train_last_two_blocks:
            train_start = 19
        elif train_last_block:
            train_start = 28
        else:
            train_start = 999
        for idx, layer in enumerate(self.vgg.features):
            if idx >= train_start:
                for p in layer.parameters():
                    p.requires_grad = True

        self.features = self.vgg.features
        self.eval()
        # Keep trainable layers in training mode when parent model is train().

    def train(self, mode: bool = True):
        super().train(mode)
        # VGG batchnorm layers, if any, follow mode. This VGG configuration has BN.
        return self

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 26:       # relu4_4
                out["relu4"] = x
            elif i == 35:     # relu5_4
                out["relu5"] = x
        if "relu4" not in out or "relu5" not in out:
            raise RuntimeError("Unexpected VGG-19 layer layout; relu4/relu5 outputs were not produced.")
        return out
