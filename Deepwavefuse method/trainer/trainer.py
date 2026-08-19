import os
import torch

from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from losses.losses import TotalLoss


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        lr=1e-4,
        weight_decay=1e-4,
        epochs=100,
        checkpoint_dir="./checkpoints"
    ):

        self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device
        self.epochs = epochs

        self.criterion = TotalLoss().to(device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs
        )

        self.checkpoint_dir = checkpoint_dir

        os.makedirs(
            checkpoint_dir,
            exist_ok=True
        )

    def train_one_epoch(self, epoch):

        self.model.train()

        running_loss = 0.0

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.epochs}"
        )

        for batch in progress:

            image_a = batch["image_a"].to(self.device)
            image_b = batch["image_b"].to(self.device)

            self.optimizer.zero_grad()

            output = self.model(
                image_a,
                image_b
            )

            fused = output["image"]

            losses = self.criterion(
                fused,
                image_a,
                image_b
            )

            losses["total"].backward()

            self.optimizer.step()

            running_loss += losses["total"].item()

            progress.set_postfix(
                loss=f"{losses['total'].item():.6f}"
            )

        return running_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def validate(self):

        if len(self.val_loader) == 0:
            return 0.0

        self.model.eval()

        running_loss = 0.0

        for batch in self.val_loader:

            image_a = batch["image_a"].to(self.device)
            image_b = batch["image_b"].to(self.device)

            output = self.model(
                image_a,
                image_b
            )

            fused = output["image"]

            losses = self.criterion(
                fused,
                image_a,
                image_b
            )

            running_loss += losses["total"].item()

        return running_loss / max(1, len(self.val_loader))

    def save_checkpoint(
        self,
        epoch,
        best=False
    ):

        filename = (
            "best_model.pth"
            if best
            else f"epoch_{epoch+1}.pth"
        )

        torch.save(

            {
                "epoch": epoch,

                "model": self.model.state_dict(),

                "optimizer": self.optimizer.state_dict(),

                "scheduler": self.scheduler.state_dict()

            },

            os.path.join(
                self.checkpoint_dir,
                filename
            )

        )

    def train(self):

        print("=" * 60)
        print("Training Started")
        print("=" * 60)

        print("Train Loader :", len(self.train_loader))
        print("Validation Loader :", len(self.val_loader))
        print()

        if len(self.train_loader) == 0:

            print("ERROR : Empty Train Loader")

            return

        best_loss = float("inf")

        for epoch in range(self.epochs):

            train_loss = self.train_one_epoch(epoch)

            val_loss = self.validate()

            self.scheduler.step()

            print(

                f"Epoch {epoch+1:03d}"

                f" | Train {train_loss:.6f}"

                f" | Val {val_loss:.6f}"

            )

            self.save_checkpoint(epoch)

            if val_loss < best_loss:

                best_loss = val_loss

                self.save_checkpoint(
                    epoch,
                    best=True
                )

        print()
        print("=" * 60)
        print("Training Finished")
        print("=" * 60)
