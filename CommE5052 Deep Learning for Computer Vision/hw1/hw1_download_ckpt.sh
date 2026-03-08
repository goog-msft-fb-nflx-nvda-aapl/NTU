#!/bin/bash
pip install gdown -q
mkdir -p ckpt
gdown "https://drive.google.com/uc?id=1LQQC8zGSvO_r7c8450u-iGfwHPH1ok2E" -O ckpt/best_C.pth