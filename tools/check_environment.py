from __future__ import annotations
import importlib
import platform
import sys
import torch

mods = ['numpy','PIL','cv2','pywt','scipy','skimage','pandas']
print('Python:', sys.version)
print('Platform:', platform.platform())
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
print('MPS available:', torch.backends.mps.is_available())
for m in mods:
    try:
        x=importlib.import_module(m)
        print(f'{m}: OK {getattr(x,"__version__","")}')
    except Exception as e:
        print(f'{m}: MISSING ({e})')
