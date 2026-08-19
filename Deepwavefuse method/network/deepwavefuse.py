import torch
import torch.nn as nn
import torch.nn.functional as F

from network.encoder import VGGEncoder
from network.attention import CrossAttention
from network.projection import FeatureProjection
from network.wavelet import WaveletTransform
from network.fusion import AdaptiveFusion
from network.idwt import InverseWavelet
from network.guided_filter import GuidedFilter
from network.decision_map import DecisionMap
from network.decoder import Decoder


class DeepWaveFuse(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = VGGEncoder()

        self.attention = CrossAttention(
            channels=512,
            heads=8
        )

        self.project = FeatureProjection(
            in_channels=512,
            out_channels=64
        )

        self.dwt = WaveletTransform()

        self.fusion = AdaptiveFusion(
            channels=64
        )

        self.idwt = InverseWavelet()

        self.guided = GuidedFilter()

        self.decision = DecisionMap(
            channels=64
        )

        self.decoder = Decoder(
            in_channels=64
        )

    def forward(
        self,
        image_a,
        image_b
    ):

        feat_a = self.encoder(image_a)
        feat_b = self.encoder(image_b)

        semantic = self.attention(
            feat_a["relu5"],
            feat_b["relu5"]
        )

        proj_a = self.project(
            feat_a["relu5"]
        )

        proj_b = self.project(
            feat_b["relu5"]
        )

        coeff_a = self.dwt(proj_a)

        coeff_b = self.dwt(proj_b)

        fused_coeff = self.fusion(
            coeff_a,
            coeff_b
        )

        feature = self.idwt(
            fused_coeff
        )

        semantic = F.interpolate(
            semantic,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        semantic = semantic[:, :64]

        feature = feature + semantic
        feature = self.guided(
            feature,
            feature
        )

        feature, weight = self.decision(
            feature
        )

        image = self.decoder(
            feature
        )

        return {
            "image": image,
            "feature": feature,
            "weight": weight
        }


if __name__ == "__main__":

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = DeepWaveFuse().to(device)

    image_a = torch.randn(
        1,
        3,
        224,
        224,
        device=device
    )

    image_b = torch.randn(
        1,
        3,
        224,
        224,
        device=device
    )

    output = model(
        image_a,
        image_b
    )

    print("=" * 60)
    print("DeepWaveFuse FINAL Network")
    print("=" * 60)
    print("Image   :", output["image"].shape)
    print("Feature :", output["feature"].shape)
    print("Weight  :", output["weight"].shape)
