from __future__ import annotations
import torch
from config import Config, resolve_device, as_abs
from models.deepwavefuse import DeepWaveFuse

cfg=Config()
cfg.vgg_weights=as_abs(cfg.vgg_weights)
device=resolve_device(cfg)
model=DeepWaveFuse(cfg).to(device).eval()
total=sum(p.numel() for p in model.parameters())
trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Device:', device)
print(f'Total parameters: {total/1e6:.3f} M')
print(f'Trainable parameters: {trainable/1e6:.3f} M')
with torch.no_grad():
    x=torch.rand(1,3,cfg.image_size,cfg.image_size,device=device)
    y=torch.rand_like(x)
    o=model(x,y)
print('Input :', x.shape)
print('Output:', o['image'].shape)
print('Wavelet:', o['wavelet_image'].shape)
print('Deep   :', o['deep_image'].shape)
