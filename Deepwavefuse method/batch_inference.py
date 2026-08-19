import os

import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from network.deepwavefuse import DeepWaveFuse


DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def load_image(path):

    image = Image.open(path).convert("RGB")

    image = image.resize((224, 224))

    image = ToTensor()(image)

    return image.unsqueeze(0).to(DEVICE)


def save_image(tensor, path):

    tensor = tensor.squeeze(0).cpu()

    image = ToPILImage()(tensor)

    image.save(path)


def main():

    print("=" * 60)
    print("DeepWaveFuse Batch Inference")
    print("=" * 60)

    model = DeepWaveFuse().to(DEVICE)

    checkpoint = torch.load(
        "./checkpoints/best_model.pth",
        map_location=DEVICE
    )

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    folder_a = "./data/Lytro/A"
    folder_b = "./data/Lytro/B"

    output_dir = "./results"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    files = sorted(

        [

            f for f in os.listdir(folder_a)

            if f.lower().endswith(

                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".tif",
                    ".tiff"
                )

            )

        ]

    )

    print("Images Found :", len(files))
    print()

    with torch.no_grad():

        for filename in files:

            path_a = os.path.join(
                folder_a,
                filename
            )

            path_b = os.path.join(
                folder_b,
                filename
            )

            if not os.path.exists(path_b):

                print("Skipping :", filename)

                continue

            image_a = load_image(path_a)

            image_b = load_image(path_b)

            output = model(
                image_a,
                image_b
            )

            fused = output["image"]

            save_image(

                fused,

                os.path.join(
                    output_dir,
                    filename
                )

            )

            print("Saved :", filename)

    print()
    print("=" * 60)
    print("Batch Inference Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()