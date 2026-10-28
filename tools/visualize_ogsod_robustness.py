#!/usr/bin/env python3
"""Create deterministic contact sheets for visual inspection of OGSOD corruptions."""
from __future__ import annotations
import argparse, random
from pathlib import Path
import cv2
import numpy as np

DATA_ROOT = Path("/opt/data/private/DATA/OGSOD-1.0")
ROBUST_ROOT = DATA_ROOT / "robustness_test"

def args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data-root",type=Path,default=DATA_ROOT); p.add_argument("--robustness-root",type=Path,default=ROBUST_ROOT); p.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"robustness_visualization"); p.add_argument("--num-samples",type=int,default=10); p.add_argument("--seed",type=int,default=3407); return p.parse_args()
def load(path):
    im=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(path)
    return im
def tile(im, title, size=(256,256)):
    canvas=cv2.resize(im,size,interpolation=cv2.INTER_AREA); cv2.rectangle(canvas,(0,0),(size[0],25),(0,0,0),-1); cv2.putText(canvas,title,(6,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA); return canvas
def main():
    a=args(); rgb=sorted((a.data_root/"rgb/images/test").glob("*")); random.Random(a.seed).shuffle(rgb); rgb=rgb[:min(a.num_samples,len(rgb))]; a.output.mkdir(parents=True,exist_ok=True)
    specs=[("Clean RGB",lambda p:a.data_root/"rgb/images/test"/p.name),("Clean SAR",lambda p:a.data_root/"sar/images/test"/p.name)]
    specs += [(f"Optical {s.title()}",lambda p,s=s:a.robustness_root/"optical_degradation"/s/"rgb/images/test"/p.name) for s in ("mild","moderate","severe")]
    specs += [(f"Misalign {s.title()}",lambda p,s=s:a.robustness_root/"modal_misalignment"/s/"sar/images/test"/p.name) for s in ("mild","moderate","severe")]
    specs += [(f"SAR Noise {s.title()}",lambda p,s=s:a.robustness_root/"sar_noise"/s/"sar/images/test"/p.name) for s in ("mild","moderate","severe")]
    for p in rgb:
        tiles=[tile(load(fn(p)),title) for title,fn in specs]
        columns=4
        blank=np.zeros_like(tiles[0])
        tiles.extend([blank.copy() for _ in range((-len(tiles)) % columns)])
        sheet=np.vstack([np.hstack(tiles[i:i+columns]) for i in range(0,len(tiles),columns)])
        out=a.output/f"{p.stem}.jpg"
        if not cv2.imwrite(str(out),sheet): raise RuntimeError(f"Could not write {out}")
    print(f"Saved {len(rgb)} visualizations to {a.output}")
if __name__=="__main__": main()
