import os

import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

from network.deepwavefuse import DeepWaveFuse


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def load_image(path):

    image = Image.open(path).convert("RGB")

    image = image.resize((224, 224))

    image = ToTensor()(image)

    image = image.unsqueeze(0)

    return image.to(device)


def save_image(tensor, path):

    tensor = tensor.squeeze(0).detach().cpu()

    image = ToPILImage()(tensor)

    image.save(path)


def main():

    model = DeepWaveFuse().to(device)

    checkpoint = torch.load(
        "./checkpoints/best_model.pth",
        map_location=device
    )

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    image_a = load_image(
        "./data/Lytro/A/001.jpg"
    )

    image_b = load_image(
        "./data/Lytro/B/001.jpg"
    )

    with torch.no_grad():

        output = model(
            image_a,
            image_b
        )

        fused = output["image"]

    os.makedirs(
        "./results",
        exist_ok=True
    )

    save_image(
        fused,
        "./results/fused.png"
    )

    print("=" * 60)
    print("DeepWaveFuse Inference")
    print("=" * 60)
    print("Input A :", "./data/Lytro/A/001.jpg")
    print("Input B :", "./data/Lytro/B/001.jpg")
    print("Saved   :", "./results/fused.png")


if __name__ == "__main__":
    main()
