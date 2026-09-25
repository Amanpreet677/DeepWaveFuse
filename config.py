from dataclasses import dataclass
from pathlib import Path
import os
import torch

ROOT = Path(__file__).resolve().parent

@dataclass
class Config:
    # Reproducibility / runtime
    seed: int = 42
    device: str = "auto"
    amp: bool = True

    # Data
    train_root: str = "datasets/MFI-WHU"
    val_root: str | None = None
    target_root: str | None = None
    image_size: int = 512
    batch_size: int = 1
    grad_accum_steps: int = 8
    num_workers: int = 0
    pin_memory: bool = True

    # Model
    vgg_weights: str = "pretrained/vgg19-dcbb9e9d.pth"
    vgg_train_last_block: bool = True
    vgg_train_last_two_blocks: bool = False
    feature_channels: int = 64
    attention_heads: int = 8
    decoder_base: int = 128
    use_skip_relu4: bool = True

    # Exact wavelet path
    wavelet: str = "dmey"
    wavelet_level: int = 3
    wavelet_mode: str = "periodization"
    soft_high_frequency: bool = False

    # Training
    epochs: int = 100
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    grad_clip: float = 1.0

    # Loss weights
    lambda_pseudo: float = 1.0
    lambda_ssim: float = 1.0
    lambda_grad: float = 0.5
    lambda_wavelet: float = 0.5
    lambda_perceptual: float = 0.1

    # Focus / refinement
    focus_kernel: int = 5
    guided_radius: int = 4
    guided_eps: float = 1e-3
    residual_refine: bool = True

    # Outputs
    output_dir: str = "results"
    checkpoint_dir: str = "results/checkpoints"
    image_dir: str = "results/images"
    metric_dir: str = "results/metrics"


def resolve_device(cfg: Config) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dirs(cfg: Config) -> None:
    for p in [cfg.output_dir, cfg.checkpoint_dir, cfg.image_dir, cfg.metric_dir]:
        Path(p).mkdir(parents=True, exist_ok=True)


def as_abs(path: str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)
