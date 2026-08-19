import torch
import torch.nn as nn


class InverseWavelet(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, coeff):

        ll = coeff["LL"]
        lh = coeff["LH"]
        hl = coeff["HL"]
        hh = coeff["HH"]

        B, C, H, W = ll.shape

        output = torch.zeros(
            B,
            C,
            H * 2,
            W * 2,
            device=ll.device
        )

        output[:, :, 0::2, 0::2] = ll
        output[:, :, 0::2, 1::2] = ll + lh
        output[:, :, 1::2, 0::2] = ll + hl
        output[:, :, 1::2, 1::2] = ll + hh

        return output


if __name__ == "__main__":

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = InverseWavelet().to(device)

    coeff = {

        "LL": torch.randn(1,64,7,7,device=device),

        "LH": torch.randn(1,64,7,7,device=device),

        "HL": torch.randn(1,64,7,7,device=device),

        "HH": torch.randn(1,64,7,7,device=device)

    }

    out = model(coeff)

    print("="*60)
    print("DeepWaveFuse Final IDWT")
    print("="*60)

    print(out.shape)
