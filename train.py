from __future__ import annotations
import argparse
from dataclasses import replace
from torch.utils.data import DataLoader, random_split
from config import Config, resolve_device, ensure_dirs, as_abs
from datasets.pair_dataset import MultiFocusPairDataset
from models.deepwavefuse import DeepWaveFuse
from trainer import Trainer
from utils.repro import seed_everything


def main():
    ap = argparse.ArgumentParser(description="Train DeepWaveFuse")
    ap.add_argument("--train-root", default=None)
    ap.add_argument("--target-root", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--resume", default=None)

    ap.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args()

    cfg = Config()
    if args.train_root: cfg.train_root = args.train_root
    if args.target_root: cfg.target_root = args.target_root
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.image_size: cfg.image_size = args.image_size
    if args.device: cfg.device = args.device

    cfg.train_root = as_abs(cfg.train_root)
    cfg.target_root = as_abs(cfg.target_root)
    cfg.vgg_weights = as_abs(cfg.vgg_weights)
    cfg.checkpoint_dir = as_abs(cfg.checkpoint_dir)
    cfg.output_dir = as_abs(cfg.output_dir)
    cfg.image_dir = as_abs(cfg.image_dir)
    cfg.metric_dir = as_abs(cfg.metric_dir)
    ensure_dirs(cfg)
    seed_everything(cfg.seed)
    device = resolve_device(cfg)
    print("Device:", device)

    dataset = MultiFocusPairDataset(cfg.train_root, cfg.image_size, cfg.target_root)
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=__import__('torch').Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=(cfg.pin_memory and device.type == 'cuda'))
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=(cfg.pin_memory and device.type == 'cuda'))
    print(f"Pairs: {len(dataset)} | Train: {n_train} | Val: {n_val} | Batch: {cfg.batch_size} | Accumulation: {cfg.grad_accum_steps}")

    model = DeepWaveFuse(cfg)
    trainer = Trainer(model, train_loader, val_loader, cfg, device, args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()
