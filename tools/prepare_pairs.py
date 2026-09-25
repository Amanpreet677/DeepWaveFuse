from __future__ import annotations
import argparse
from pathlib import Path
from utils.image_ops import align_sift_affine

EXT={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}

def main():
    ap=argparse.ArgumentParser(description='Align paired images using SIFT + affine transformation.')
    ap.add_argument('--input-root',required=True,help='root containing A/ and B/')
    ap.add_argument('--output-root',required=True,help='destination root containing aligned A/ and B/')
    args=ap.parse_args()
    src=Path(args.input_root); dst=Path(args.output_root)
    (dst/'A').mkdir(parents=True,exist_ok=True); (dst/'B').mkdir(parents=True,exist_ok=True)
    for a in sorted((src/'A').iterdir()):
        if a.suffix.lower() not in EXT: continue
        b=src/'B'/a.name
        if not b.exists(): continue
        ok=align_sift_affine(str(a),str(b),str(dst/'A'/a.name),str(dst/'B'/b.name))
        print(a.name, 'aligned' if ok else 'copied_without_transform')

if __name__=='__main__': main()
