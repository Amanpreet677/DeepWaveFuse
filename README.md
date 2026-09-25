# DeepWaveFuse

**DeepWaveFuse** is a deep learning framework for multi-focus image fusion that combines deep feature extraction, attention-based feature processing, and wavelet-domain decomposition for improved preservation of focused structures and image details.

> **Repository purpose:** This repository contains the research implementation and supporting code for DeepWaveFuse. Datasets, trained checkpoints, generated fused images, and experimental result files are intentionally excluded from the public code repository.

## Overview

Multi-focus image fusion aims to combine complementary information from images captured with different focus settings into a single image with a larger depth of field.

DeepWaveFuse follows a hybrid architecture that integrates:

- Deep feature extraction
- Attention-based feature processing
- Wavelet-based feature decomposition
- Feature fusion
- Image reconstruction
- Quantitative image-fusion evaluation

The implementation is organized into modular components so that the model, loss functions, datasets, evaluation metrics, and training pipeline can be studied and reproduced independently.

## Repository Structure

```text
DeepWaveFuse/
├── datasets/
│   ├── __init__.py
│   └── pair_dataset.py
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── metrics.py
│   └── qabf_reference.py
│
├── losses/
│   ├── __init__.py
│   ├── fusion_loss.py
│   └── fusion_loss_v6_wavelet.py
│
├── models/
│   ├── __init__.py
│   ├── attention.py
│   ├── decision.py
│   ├── decoder.py
│   ├── deepwavefuse.py
│   ├── encoder.py
│   ├── guided_filter.py
│   └── wavelet.py
│
├── scripts/
│   ├── train_mps.sh
│   └── train_rtx4050.sh
│
├── tools/
│   ├── check_environment.py
│   ├── inspect_model.py
│   └── prepare_pairs.py
│
├── utils/
│   ├── __init__.py
│   ├── image_ops.py
│   └── repro.py
│
├── config.py
├── train.py
├── trainer.py
├── inference.py
└── finetune_lytro_v3.py
```

## Main Components

### Model

The main DeepWaveFuse implementation is located in:

```text
models/deepwavefuse.py
```

Supporting modules include:

- `encoder.py` — deep feature extraction
- `attention.py` — attention-based feature processing
- `wavelet.py` — wavelet-domain decomposition and processing
- `decision.py` — feature/fusion decision operations
- `decoder.py` — reconstruction
- `guided_filter.py` — guided filtering utilities

### Training

The main training entry point is:

```bash
python3 train.py
```

The training procedure is implemented in:

```text
trainer.py
```

Configuration parameters are maintained in:

```text
config.py
```

### Inference

For image fusion inference:

```bash
python3 inference.py
```

Use the configuration and input/output paths defined by the implementation before running inference on your own dataset.

### Evaluation

Evaluation utilities are provided under:

```text
evaluation/
```

The implementation includes commonly used image-fusion metrics such as:

- PSNR
- SSIM
- QAB/F
- Mutual Information (MI)
- VIF
- Standard Deviation (SD)
- Entropy (EN)
- Spatial Frequency (SF)
- Average Gradient (AG)

The evaluation code is intended to calculate metrics from generated fused images rather than storing precomputed experimental results in this repository.

## Dataset

The code is designed to work with paired multi-focus image datasets.

**Datasets are not included in this repository.**

This keeps the repository lightweight and avoids redistributing datasets that may have their own licensing or access conditions.

After obtaining a dataset from its official source, arrange the files according to the dataset loader expected by:

```text
datasets/pair_dataset.py
```

## Installation

Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install torch torchvision torchaudio
pip install numpy scipy pandas opencv-python
pip install pillow matplotlib scikit-image scikit-learn
pip install PyWavelets tqdm
```

Depending on the hardware and PyTorch version, the appropriate PyTorch installation command may differ.

## Hardware

The code can be adapted to different compute environments.

For Apple Silicon systems with supported PyTorch MPS acceleration:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

For NVIDIA systems, CUDA availability can be checked with:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

The repository also includes example training scripts under:

```text
scripts/
```

## Reproducibility

For reproducible experiments, the project includes utilities for deterministic/random-seed configuration:

```text
utils/repro.py
```

Before reporting experimental results, run the evaluation code on the generated fused images and retain the exact configuration used for the experiment.

**This repository does not hard-code experimental result tables or metric values.**

## Research Use

This repository is intended to support academic research, experimentation, and reproducibility of the DeepWaveFuse implementation.

If you use the implementation in academic work, please cite the corresponding DeepWaveFuse publication when the final bibliographic information is available.

## Author

**Amanpreet Sharma**

Centre for Research Impact & Outcome  
Chitkara University Institute of Engineering and Technology  
Punjab, India

## Notes

- Training datasets are not included.
- Pretrained model weights are not included.
- Generated fused images are not included.
- Checkpoints are not included.
- Experimental CSV files and result tables are not included.
- Results should be generated independently using the provided evaluation code.

## License

A license file has not been added to this repository yet. Until a license is specified, the source code should not be assumed to carry an open-source license beyond the permissions explicitly granted by the repository owner.
