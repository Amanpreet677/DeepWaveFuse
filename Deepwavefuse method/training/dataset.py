import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MultiFocusDataset(Dataset):

    def __init__(
        self,
        root_dir,
        image_size=224,
        transform=None
    ):

        self.root_dir = root_dir

        self.focus_a = os.path.join(
            root_dir,
            "A"
        )

        self.focus_b = os.path.join(
            root_dir,
            "B"
        )

        self.files = sorted([
    f for f in os.listdir(self.focus_a)
    if f.lower().endswith(
        (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    )
])

        if transform is None:

            self.transform = transforms.Compose([

                transforms.Resize(
                    (image_size, image_size)
                ),

                transforms.ToTensor()

            ])

        else:

            self.transform = transform

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]

        img_a = Image.open(
            os.path.join(
                self.focus_a,
                filename
            )
        ).convert("RGB")

        img_b = Image.open(
            os.path.join(
                self.focus_b,
                filename
            )
        ).convert("RGB")

        img_a = self.transform(img_a)
        img_b = self.transform(img_b)

        return {

            "image_a": img_a,

            "image_b": img_b,

            "name": filename

        }


if __name__ == "__main__":

    dataset = MultiFocusDataset(
        root_dir="./data/Lytro"
    )

    print("=" * 60)
    print("DeepWaveFuse Dataset")
    print("=" * 60)
    print("Samples :", len(dataset))

    sample = dataset[0]

    print(sample["image_a"].shape)
    print(sample["image_b"].shape)
