#!/usr/bin/env bash
set -euo pipefail
python train.py --device mps --train-root datasets/MFI-WHU --epochs 100 --batch-size 1 --image-size 512
