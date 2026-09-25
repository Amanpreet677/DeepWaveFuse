from __future__ import annotations

import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import vifp


# ============================================================
# Basic utilities
# ============================================================

def _gray(x):
    return cv2.cvtColor(
        x,
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32) / 255.0


def entropy(x):
    g = (_gray(x) * 255).astype(np.uint8)

    hist = np.bincount(
        g.ravel(),
        minlength=256
    ).astype(np.float64)

    p = hist / max(1, hist.sum())
    p = p[p > 0]

    return float(
        -(p * np.log2(p)).sum()
    )


def spatial_frequency(x):
    g = _gray(x)

    rf = np.diff(
        g,
        axis=0
    )

    cf = np.diff(
        g,
        axis=1
    )

    return float(
        np.sqrt(
            np.mean(rf ** 2) +
            np.mean(cf ** 2)
        )
    )


def average_gradient(x):
    g = _gray(x)

    gy, gx = np.gradient(g)

    return float(
        np.mean(
            np.sqrt(
                gx ** 2 +
                gy ** 2
            )
        )
    )


# ============================================================
# Mutual Information
# ============================================================

def _mi(a, b):
    x = cv2.cvtColor(
        a,
        cv2.COLOR_RGB2GRAY
    ).ravel()

    y = cv2.cvtColor(
        b,
        cv2.COLOR_RGB2GRAY
    ).ravel()

    if x.size != y.size:
        raise ValueError(
            "MI requires images with the same number "
            "of pixels."
        )

    h, _, _ = np.histogram2d(
        x,
        y,
        bins=256,
        range=[
            [0, 255],
            [0, 255]
        ]
    )

    pxy = h / max(
        1,
        h.sum()
    )

    px = pxy.sum(
        axis=1,
        keepdims=True
    )

    py = pxy.sum(
        axis=0,
        keepdims=True
    )

    den = px @ py

    mask = pxy > 0

    return float(
        np.sum(
            pxy[mask] *
            np.log2(
                (pxy[mask] + 1e-12) /
                (den[mask] + 1e-12)
            )
        )
    )


def mutual_information_fusion(
    fused,
    a,
    b
):
    """
    Total mutual information between the fused
    image and the two source images.

    MI(F,A) + MI(F,B)
    """

    return (
        _mi(fused, a) +
        _mi(fused, b)
    )


# ============================================================
# PSNR / SSIM
# ============================================================

def psnr(x, ref):
    mse = float(
        np.mean(
            (
                x.astype(np.float32) / 255.0 -
                ref.astype(np.float32) / 255.0
            ) ** 2
        )
    )

    return float(
        10.0 *
        np.log10(
            1.0 /
            max(mse, 1e-12)
        )
    )


def ssim_rgb(x, ref):
    return float(
        ssim(
            x,
            ref,
            channel_axis=2,
            data_range=255
        )
    )


# ============================================================
# QAB/F
# Xydeas-Petrović style gradient/orientation measure
# ============================================================

def _gradient_features(im):
    g = cv2.cvtColor(
        im,
        cv2.COLOR_RGB2GRAY
    ).astype(np.float64)

    gx = cv2.Sobel(
        g,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        g,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    magnitude = np.sqrt(
        gx * gx +
        gy * gy
    )

    orientation = np.arctan2(
        gy,
        gx
    )

    return magnitude, orientation


def _qabf_pair(
    fused_mag,
    fused_ori,
    source_mag,
    source_ori
):
    eps = 1e-12

    # --------------------------------------------------------
    # Gradient strength preservation
    # --------------------------------------------------------

    g_ratio = (
        np.minimum(
            fused_mag,
            source_mag
        ) /
        (
            np.maximum(
                fused_mag,
                source_mag
            ) + eps
        )
    )

    # --------------------------------------------------------
    # Gradient orientation preservation
    # --------------------------------------------------------

    delta = np.abs(
        fused_ori -
        source_ori
    )

    delta = np.minimum(
        delta,
        2.0 * np.pi - delta
    )

    o_ratio = (
        1.0 -
        delta / np.pi
    )

    o_ratio = np.clip(
        o_ratio,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Standard QAB/F-style quality components
    # --------------------------------------------------------

    q_g = g_ratio

    q_o = o_ratio

    # Weight according to source gradient strength.
    weight = source_mag

    numerator = np.sum(
        weight *
        q_g *
        q_o
    )

    denominator = np.sum(
        weight
    ) + eps

    return float(
        numerator /
        denominator
    )


def qabf(fused, a, b):
    """
    Gradient/orientation based QAB/F measure.

    This implementation evaluates preservation of
    gradient magnitude and orientation from both
    source images in the fused image.
    """

    fused_mag, fused_ori = _gradient_features(
        fused
    )

    a_mag, a_ori = _gradient_features(
        a
    )

    b_mag, b_ori = _gradient_features(
        b
    )

    qa = _qabf_pair(
        fused_mag,
        fused_ori,
        a_mag,
        a_ori
    )

    qb = _qabf_pair(
        fused_mag,
        fused_ori,
        b_mag,
        b_ori
    )

    wa = np.sum(a_mag)
    wb = np.sum(b_mag)

    return float(
        (
            wa * qa +
            wb * qb
        ) /
        (
            wa +
            wb +
            1e-12
        )
    )


# ============================================================
# VIF
# ============================================================

def vif(fused, a, b):
    """
    VIF-P based fusion quality.

    The source-wise VIF values are averaged so that
    the resulting score remains on the conventional
    0-1-ish fusion evaluation scale used by the project.
    """

    fused_gray = cv2.cvtColor(
        fused,
        cv2.COLOR_RGB2GRAY
    )

    a_gray = cv2.cvtColor(
        a,
        cv2.COLOR_RGB2GRAY
    )

    b_gray = cv2.cvtColor(
        b,
        cv2.COLOR_RGB2GRAY
    )

    va = float(
        vifp(
            a_gray,
            fused_gray
        )
    )

    vb = float(
        vifp(
            b_gray,
            fused_gray
        )
    )

    return float(
        np.mean(
            [
                va,
                vb
            ]
        )
    )


# ============================================================
# Image loading / resizing
# ============================================================

def _read_rgb(path):
    image = cv2.imread(
        str(path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {path}"
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


def _match_size(
    image,
    target_shape
):
    target_h, target_w = target_shape[:2]

    if image.shape[:2] == (
        target_h,
        target_w
    ):
        return image

    return cv2.resize(
        image,
        (target_w, target_h),
        interpolation=cv2.INTER_CUBIC
    )


# ============================================================
# Complete evaluation
# ============================================================

def evaluate_triplet(
    fused_path,
    a_path,
    b_path,
    ref_path=None
):

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    fused = _read_rgb(
        fused_path
    )

    a = _read_rgb(
        a_path
    )

    b = _read_rgb(
        b_path
    )

    # --------------------------------------------------------
    # Match source dimensions to fused image
    # --------------------------------------------------------

    target_shape = fused.shape[:2]

    a = _match_size(
        a,
        target_shape
    )

    b = _match_size(
        b,
        target_shape
    )

    # --------------------------------------------------------
    # Fusion metrics
    # --------------------------------------------------------

    out = {

        "MI":
            mutual_information_fusion(
                fused,
                a,
                b
            ),

        "VIF":
            vif(
                fused,
                a,
                b
            ),

        "QABF":
            qabf(
                fused,
                a,
                b
            ),

        "SD":
            float(
                np.std(
                    cv2.cvtColor(
                        fused,
                        cv2.COLOR_RGB2GRAY
                    )
                )
            ),

        "EN":
            entropy(
                fused
            ),

        "SF":
            spatial_frequency(
                fused
            ),

        "AG":
            average_gradient(
                fused
            ),
    }

    # --------------------------------------------------------
    # Reference-based metrics
    # --------------------------------------------------------

    if ref_path:

        ref = _read_rgb(
            ref_path
        )

        ref = _match_size(
            ref,
            fused.shape[:2]
        )

        out["PSNR"] = psnr(
            fused,
            ref
        )

        out["SSIM"] = ssim_rgb(
            fused,
            ref
        )

    return out
