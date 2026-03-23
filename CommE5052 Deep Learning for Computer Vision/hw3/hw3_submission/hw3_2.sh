#!/bin/bash
# $1: path to folder containing test images
# $2: path to output json file
# $3: path to decoder_model.bin

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/p2/inference.py" \
    "$1" \
    "$2" \
    "$3" \
    --checkpoint "${SCRIPT_DIR}/p2/checkpoints/best_model.bin" \
    --lora_r 4 \
    --lora_alpha 32 \
    --max_new_tokens 50 \
    --batch_size 16
