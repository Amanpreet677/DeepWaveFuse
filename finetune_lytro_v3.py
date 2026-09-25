from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from config import Config, resolve_device, as_abs, ensure_dirs
from datasets.pair_dataset import MultiFocusPairDataset
from models.deepwavefuse import DeepWaveFuse
from trainer import Trainer
from utils.repro import seed_everything

cfg = Config()

# Lytro-specific experiment
cfg.train_root = as_abs("datasets/Lytro")
cfg.target_root = None
cfg.image_size = 520
cfg.batch_size = 1
cfg.num_workers = 0
cfg.epochs = 10
cfg.lr = 5e-6
cfg.device = "mps"

# Keep existing training settings
cfg.grad_accum_steps = 8

# Completely separate experiment outputs
cfg.output_dir = as_abs("results/lytro_finetune")
cfg.checkpoint_dir = as_abs("results/checkpoints_lytro_finetune_v3")
cfg.image_dir = as_abs("results/lytro_finetune_v3/images")
cfg.metric_dir = as_abs("results/lytro_finetune_v3/metrics")
cfg.vgg_weights = as_abs(cfg.vgg_weights)

ensure_dirs(cfg)
seed_everything(cfg.seed)

device = resolve_device(cfg)

print("=" * 70)
print("DEEPWAVEFUSE — LYTR0 FINE-TUNING")
print("=" * 70)
print("Device:", device)
print("Image size:", cfg.image_size)
print("Epochs:", cfg.epochs)
print("Batch size:", cfg.batch_size)
print("Gradient accumulation:", cfg.grad_accum_steps)

dataset = MultiFocusPairDataset(
    cfg.train_root,
    cfg.image_size,
    cfg.target_root
)

n_val = max(1, int(0.2 * len(dataset)))
n_train = len(dataset) - n_val

train_set, val_set = random_split(
    dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(cfg.seed)
)

train_loader = DataLoader(
    train_set,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=False
)

val_loader = DataLoader(
    val_set,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=False
)

print(f"Pairs: {len(dataset)} | Train: {n_train} | Val: {n_val}")

# Create fresh model
model = DeepWaveFuse(cfg)

# Load ONLY the trained model weights.
# Do NOT restore the old MFI-WHU optimizer/scheduler.
source = "results/checkpoints_lytro_finetune_v2/best_model.pth"

checkpoint = torch.load(
    source,
    map_location="cpu"
)

model.load_state_dict(checkpoint["model"], strict=True)

print("Initial weights loaded from:")
print(source)

# Fresh optimizer/scheduler through Trainer
trainer = Trainer(
    model,
    train_loader,
    val_loader,
    cfg,
    device,
    resume_path=None
)

print("Fresh Lytro optimizer/scheduler created.")
print("=" * 70)

trainer.fit()

print("=" * 70)
print("LYTRO FINE-TUNING COMPLETE")
print("Checkpoints:", cfg.checkpoint_dir)
print("=" * 70)
