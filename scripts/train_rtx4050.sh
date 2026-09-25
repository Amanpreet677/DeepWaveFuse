#!/usr/bin/env bash
set -euo pipefail
python train.py --device cuda --train-root datasets/MFI-WHU --epochs 100 --batch-size 1 --image-size 512
