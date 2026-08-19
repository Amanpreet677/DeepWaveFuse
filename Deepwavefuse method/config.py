"""
DeepWaveFuse Configuration
"""

from pathlib import Path
import torch

# ---------------------------------------------------
# Root
# ---------------------------------------------------

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------
# Device
# ---------------------------------------------------

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# ---------------------------------------------------
# Dataset
# ---------------------------------------------------

DATASET_DIR = ROOT / "datasets"

LYTRO_DIR = DATASET_DIR / "Lytro"

MFIWHU_DIR = DATASET_DIR / "MFI-WHU"

COCO_DIR = DATASET_DIR / "COCO2017"

VOC_DIR = DATASET_DIR / "VOC2012"

# ---------------------------------------------------
# Pretrained
# ---------------------------------------------------

PRETRAIN_DIR = ROOT / "pretrained"

VGG19_PATH = PRETRAIN_DIR / "vgg19-dcbb9e9d.pth"

# ---------------------------------------------------
# Results
# ---------------------------------------------------

RESULT_DIR = ROOT / "results"

CHECKPOINT_DIR = RESULT_DIR / "checkpoints"

IMAGE_DIR = RESULT_DIR / "images"

METRIC_DIR = RESULT_DIR / "metrics"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------
# Image
# ---------------------------------------------------

IMAGE_SIZE = 224

CHANNELS = 3

# ---------------------------------------------------
# Wavelet
# ---------------------------------------------------

WAVELET = "dmey"

LEVEL = 3

# ---------------------------------------------------
# Training
# ---------------------------------------------------

BATCH_SIZE = 8

EPOCHS = 100

LR = 1e-4

WEIGHT_DECAY = 1e-5

# ---------------------------------------------------
# Seed
# ---------------------------------------------------

SEED = 42

# ---------------------------------------------------

if __name__ == "__main__":

    print("=" * 40)
    print("DeepWaveFuse Configuration")
    print("=" * 40)

    print("Device :", DEVICE)
    print("Dataset :", DATASET_DIR)
    print("Weights :", VGG19_PATH)
    print("Wavelet :", WAVELET)
    print("Levels :", LEVEL)
    print("Batch :", BATCH_SIZE)
    print("Epochs :", EPOCHS)