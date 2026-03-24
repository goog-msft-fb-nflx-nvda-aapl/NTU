#!/bin/bash
mkdir -p dust3r/checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    -O dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
