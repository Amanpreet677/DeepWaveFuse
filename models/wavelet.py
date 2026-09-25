from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn as nn

try:
    import pywt
except ImportError as exc:  # pragma: no cover
    pywt = None
    _PYWT_ERROR = exc
else:
    _PYWT_ERROR = None


def _require_pywt():
    if pywt is None:
        raise ImportError("PyWavelets is required for the exact Meyer (dmey) branch. Install with: pip install PyWavelets") from _PYWT_ERROR


def _pairwise_dwt(image: torch.Tensor, wavelet: str, level: int, mode: str):
    """Exact PyWavelets decomposition on a batch of CHW tensors.

    The transform itself is fixed/non-learned; gradients flow through the learned
    attention/fusion weights that operate on the returned coefficient tensors.
    """
    _require_pywt()
    # image is [C,H,W], float32 CPU numpy.
    arr = image.detach().cpu().numpy().transpose(1, 2, 0)
    coeffs = []
    for c in range(arr.shape[2]):
        coeffs.append(pywt.wavedec2(arr[:, :, c], wavelet=wavelet, mode=mode, level=level))
    return coeffs


def decompose_batch(x: torch.Tensor, wavelet: str = "dmey", level: int = 3, mode: str = "periodization"):
    """Return nested Python structures of CPU numpy arrays for each image/channel."""
    _require_pywt()
    return [_pairwise_dwt(sample, wavelet, level, mode) for sample in x]


def reconstruct_fused_batch(
    coeffs_a: list,
    coeffs_b: list,
    wa: torch.Tensor,
    wb: torch.Tensor,
    wavelet: str = "dmey",
    mode: str = "periodization",
    soft_high_frequency: bool = False,
) -> torch.Tensor:
    """Fuse 3-level Meyer coefficients and reconstruct an image batch.

    wa/wb are image-space attention maps. The LL map is downsampled to each
    coefficient resolution. Detail bands use max-absolute selection by default;
    optionally a soft logistic selector can be enabled.
    """
    _require_pywt()
    B = len(coeffs_a)
    out = []
    wa_cpu = wa.detach().cpu().numpy()
    wb_cpu = wb.detach().cpu().numpy()

    for i in range(B):
        channel_coeffs = []
        for c in range(3):
            ca = coeffs_a[i][c]
            cb = coeffs_b[i][c]
            # LL at the deepest level.
            wa_ll = torch.from_numpy(wa_cpu[i, 0]).float()[None, None]
            wb_ll = torch.from_numpy(wb_cpu[i, 0]).float()[None, None]
            ll_shape = ca[0].shape
            wa_ll = torch.nn.functional.interpolate(wa_ll, size=ll_shape, mode="bilinear", align_corners=False)[0, 0].numpy()
            wb_ll = torch.nn.functional.interpolate(wb_ll, size=ll_shape, mode="bilinear", align_corners=False)[0, 0].numpy()
            ll = (wa_ll * ca[0] + wb_ll * cb[0]) / (wa_ll + wb_ll + 1e-6)

            details = []
            # pywt order: [cA_n, (H_n,V_n,D_n), ..., (H_1,V_1,D_1)]
            for level_idx in range(1, len(ca)):
                a_triplet, b_triplet = ca[level_idx], cb[level_idx]
                # Use attention map downsampled to the band size as a smooth tie-break.
                band_triplet = []
                for a_band, b_band in zip(a_triplet, b_triplet):
                    if soft_high_frequency:
                        mean_abs = np.abs(a_band) - np.abs(b_band)
                        mask = 1.0 / (1.0 + np.exp(-8.0 * np.clip(mean_abs, -5.0, 5.0)))
                        f = mask * a_band + (1.0 - mask) * b_band
                    else:
                        f = np.where(np.abs(a_band) >= np.abs(b_band), a_band, b_band)
                    band_triplet.append(f)
                details.append(tuple(band_triplet))

            rec_coeffs = [ll] + details
            rec = pywt.waverec2(rec_coeffs, wavelet=wavelet, mode=mode)
            channel_coeffs.append(rec.astype(np.float32))
        rec = np.stack(channel_coeffs, axis=0)
        out.append(rec)

    arr = np.stack(out, axis=0)
    return torch.from_numpy(arr).to(wa.device)


class MeyerWaveletBranch(nn.Module):
    """Wrapper exposing exact 3-level discrete Meyer wavelet fusion."""
    def __init__(self, wavelet: str = "dmey", level: int = 3, mode: str = "periodization", soft_high_frequency: bool = False):
        super().__init__()
        if level != 3:
            raise ValueError("This publication configuration is fixed to a 3-level Meyer wavelet transform.")
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.soft_high_frequency = soft_high_frequency

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor) -> torch.Tensor:
        ca = decompose_batch(image_a, self.wavelet, self.level, self.mode)
        cb = decompose_batch(image_b, self.wavelet, self.level, self.mode)
        return reconstruct_fused_batch(ca, cb, wa, wb, self.wavelet, self.mode, self.soft_high_frequency)
