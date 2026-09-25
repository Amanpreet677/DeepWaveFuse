from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def load_rgb(path: str, size: int | None = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def save_rgb(tensor: torch.Tensor, path: str) -> None:
    x = tensor.detach().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    arr = (x.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    # Dataset-local robust normalization; keeps [0,1] range.
    p2 = torch.quantile(x.flatten(1), 0.02, dim=1).view(-1, 1, 1, 1)
    p98 = torch.quantile(x.flatten(1), 0.98, dim=1).view(-1, 1, 1, 1)
    x = (x - p2) / (p98 - p2).clamp_min(1e-6)
    return x.clamp(0, 1)


def to_gray(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    g = to_gray(x)
    gx = g[..., :, 1:] - g[..., :, :-1]
    gy = g[..., 1:, :] - g[..., :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return torch.sqrt(gx.square() + gy.square() + 1e-8)


def focus_map_pair(a: torch.Tensor, b: torch.Tensor, kernel: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    ga = gradient_magnitude(a)
    gb = gradient_magnitude(b)
    pad = kernel // 2
    ea = F.avg_pool2d(ga.square(), kernel, 1, pad)
    eb = F.avg_pool2d(gb.square(), kernel, 1, pad)
    logits = 12.0 * (ea - eb)
    wa = torch.sigmoid(logits)
    wb = 1.0 - wa
    return wa, wb


def align_sift_affine(a_path: str, b_path: str, out_a: str, out_b: str) -> bool:
    a = cv2.imread(a_path, cv2.IMREAD_COLOR)
    b = cv2.imread(b_path, cv2.IMREAD_COLOR)
    if a is None or b is None:
        return False
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    ka, da = sift.detectAndCompute(gray_a, None)
    kb, db = sift.detectAndCompute(gray_b, None)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        cv2.imwrite(out_a, a)
        cv2.imwrite(out_b, b)
        return False
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(da, db, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good) < 8:
        cv2.imwrite(out_a, a)
        cv2.imwrite(out_b, b)
        return False
    pts_a = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_b = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffinePartial2D(pts_b, pts_a, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None:
        cv2.imwrite(out_a, a)
        cv2.imwrite(out_b, b)
        return False
    h, w = a.shape[:2]
    b_aligned = cv2.warpAffine(b, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite(out_a, a)
    cv2.imwrite(out_b, b_aligned)
    return True
