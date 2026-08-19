import torch
from torch.utils.data import DataLoader, random_split

from network.deepwavefuse import DeepWaveFuse
from training.dataset import MultiFocusDataset
from trainer.trainer import Trainer


def main():

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("DeepWaveFuse Training")
    print("=" * 60)
    print("Device :", device)

    dataset = MultiFocusDataset(
        root_dir="./data/Lytro"
    )

    total = len(dataset)

    train_size = max(1, int(total * 0.8))
    val_size = total - train_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=2,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=2,
        shuffle=False
    )

    print("Dataset :", len(dataset))
    print("Train Loader :", len(train_loader))
    print("Validation Loader :", len(val_loader))

    model = DeepWaveFuse()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=5,
        lr=1e-4
    )

    trainer.train()


if __name__ == "__main__":
    main()
