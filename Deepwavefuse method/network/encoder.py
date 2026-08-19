import os
import torch
import torch.nn as nn
from torchvision import models


class VGGEncoder(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        vgg = models.vgg19(weights=None)

        if pretrained:
            weight_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "pretrained",
                "vgg19-dcbb9e9d.pth"
            )

            state_dict = torch.load(weight_path, map_location="cpu")
            vgg.load_state_dict(state_dict)

        self.features = vgg.features

    def forward(self, x):

        outputs = {}

        for i, layer in enumerate(self.features):

            x = layer(x)

            if i == 3:
                outputs["relu1"] = x

            elif i == 8:
                outputs["relu2"] = x

            elif i == 17:
                outputs["relu3"] = x

            elif i == 26:
                outputs["relu4"] = x

            elif i == 35:
                outputs["relu5"] = x

        return outputs


if __name__ == "__main__":

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = VGGEncoder().to(device)

    image = torch.randn(1, 3, 224, 224).to(device)

    features = model(image)

    print("=" * 60)
    print("DeepWaveFuse Encoder")
    print("=" * 60)

    for k, v in features.items():
        print(k, v.shape)
