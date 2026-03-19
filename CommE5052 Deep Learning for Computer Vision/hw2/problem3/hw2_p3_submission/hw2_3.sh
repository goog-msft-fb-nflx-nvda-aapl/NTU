#!/bin/bash

JSON_PATH=$1
INPUT_DIR=$2
OUTPUT_DIR=$3
CKPT_PATH=$4

cd stable-diffusion
pip install -e . -q
cd ..

python3 inference_controlnet.py \
    --json_path $JSON_PATH \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --ckpt $CKPT_PATH