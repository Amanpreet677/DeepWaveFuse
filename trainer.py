from __future__ import annotations
from pathlib import Path
import torch
from torch.cuda.amp import GradScaler
from contextlib import nullcontext


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg, device, resume_path=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=max(1, cfg.epochs), eta_min=cfg.min_lr)
        self.criterion = __import__("losses.fusion_loss", fromlist=["FusionLoss"]).FusionLoss(cfg.lambda_pseudo, cfg.lambda_ssim, cfg.lambda_grad, cfg.lambda_wavelet, cfg.lambda_perceptual).to(device)
        self.scaler = GradScaler(enabled=(device.type == "cuda" and cfg.amp))
        self.ckpt_dir = Path(cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = 0
        self.best = float("inf")
        if resume_path is not None:
            checkpoint = torch.load(resume_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            self.optim.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.start_epoch = int(checkpoint["epoch"])
            print(f"Resumed from epoch {self.start_epoch}")



    def _forward_loss(self, batch):
        a = batch["image_a"].to(self.device, non_blocking=True)
        b = batch["image_b"].to(self.device, non_blocking=True)
        target = batch.get("target")
        if isinstance(target, torch.Tensor):
            target = target.to(self.device, non_blocking=True)
            if target.numel() == 0:
                target = None
        out = self.model(a, b)
        losses = self.criterion(out["image"], a, b, target=target, wavelet_image=out["wavelet_image"])
        return losses, out

    def fit(self):
        best = self.best
        for epoch in range(self.start_epoch, self.cfg.epochs):
            self.model.train()
            total = 0.0
            self.optim.zero_grad(set_to_none=True)
            for step, batch in enumerate(self.train_loader):
                amp_ctx = torch.cuda.amp.autocast(enabled=(self.device.type == "cuda" and self.cfg.amp)) if self.device.type == "cuda" else nullcontext()
                with amp_ctx:
                    losses, _ = self._forward_loss(batch)
                    loss = losses["total"] / self.cfg.grad_accum_steps
                self.scaler.scale(loss).backward()
                if (step + 1) % self.cfg.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                    if self.device.type == "cuda":
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    self.optim.zero_grad(set_to_none=True)
                total += losses["total"].item()

            val = self.validate()
            self.scheduler.step()
            state = {"model": self.model.state_dict(), "optimizer": self.optim.state_dict(), "scheduler": self.scheduler.state_dict(), "epoch": epoch + 1, "config": vars(self.cfg)}
            torch.save(state, self.ckpt_dir / f"epoch_{epoch+1:03d}.pth")
            if val < best:
                best = val
                torch.save(state, self.ckpt_dir / "best_model.pth")
            print(f"Epoch {epoch+1:03d}/{self.cfg.epochs} | train={total/max(1,len(self.train_loader)):.6f} | val={val:.6f} | lr={self.optim.param_groups[0]['lr']:.2e}")

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total = 0.0
        for batch in self.val_loader:
            losses, _ = self._forward_loss(batch)
            total += losses["total"].item()
        return total / max(1, len(self.val_loader))
