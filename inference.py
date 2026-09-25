from __future__ import annotations
import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from config import Config, resolve_device, as_abs
from models.deepwavefuse import DeepWaveFuse
from utils.image_ops import save_rgb


def load(path, size, device):
    img = Image.open(path).convert('RGB').resize((size, size), Image.Resampling.BICUBIC)
    return TF.to_tensor(img).unsqueeze(0).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--checkpoint', default='results/checkpoints/best_model.pth')
    ap.add_argument('--output', default='results/images/fused.png')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()
    cfg = Config()
    cfg.image_size = args.size
    cfg.device = args.device
    cfg.vgg_weights = as_abs(cfg.vgg_weights)
    device = resolve_device(cfg)
    model = DeepWaveFuse(cfg).to(device)
    state = torch.load(as_abs(args.checkpoint), map_location=device)
    model.load_state_dict(state['model'] if 'model' in state else state, strict=True)
    model.eval()
    a, b = load(args.a, args.size, device), load(args.b, args.size, device)
    with torch.no_grad():
        fused = model(a, b)['image']
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_rgb(fused, str(out))
    print('Saved:', out)

if __name__ == '__main__':
    main()
