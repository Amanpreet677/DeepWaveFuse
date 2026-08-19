import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletTransform(nn.Module):
    """
    DeepWaveFuse
    Differentiable Haar DWT
    Final Interface
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):

        ll = F.avg_pool2d(
            x,
            kernel_size=2,
            stride=2
        )

        lh = x[:, :, 0::2, 1::2] - ll

        hl = x[:, :, 1::2, 0::2] - ll

        hh = x[:, :, 1::2, 1::2] - ll

        return {

            "LL": ll,

            "LH": lh,

            "HL": hl,

            "HH": hh

        }


if __name__ == "__main__":

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = WaveletTransform().to(device)

    x = torch.randn(
        1,
        64,
        14,
        14,
        device=device
    )

    coeff = model(x)

    print("=" * 60)
    print("DeepWaveFuse Final Wavelet")
    print("=" * 60)

    for k, v in coeff.items():
        print(k, v.shape)
