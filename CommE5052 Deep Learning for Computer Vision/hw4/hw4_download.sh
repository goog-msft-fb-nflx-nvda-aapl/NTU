#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Download DUSt3R checkpoint
mkdir -p "${SCRIPT_DIR}/checkpoints"
wget "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
    -O "${SCRIPT_DIR}/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

# Download MASt3R checkpoint
mkdir -p "${SCRIPT_DIR}/InstantSplat/mast3r/checkpoints"
wget "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
    -O "${SCRIPT_DIR}/InstantSplat/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

echo "Download complete."
