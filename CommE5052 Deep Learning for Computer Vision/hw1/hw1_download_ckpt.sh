#!/bin/bash
pip install gdown -q

# Download P1 checkpoint (Setting C)
mkdir -p ckpt
gdown "https://drive.google.com/uc?id=1LQQC8zGSvO_r7c8450u-iGfwHPH1ok2E" -O ckpt/best_C.pth

# Download P2 checkpoint (DeepLabV3+)
mkdir -p ckpt_p2
gdown "https://drive.google.com/uc?id=PLACEHOLDER_P2_FILE_ID" -O ckpt_p2/best_deeplab.pth
