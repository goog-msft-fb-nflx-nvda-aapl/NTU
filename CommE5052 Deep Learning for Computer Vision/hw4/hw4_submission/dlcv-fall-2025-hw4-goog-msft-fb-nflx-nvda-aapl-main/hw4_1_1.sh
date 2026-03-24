#!/bin/bash
# $1: index txt path
# $2: original image pair directory
# $3: model checkpoint path
# $4: output prediction file path

python3 dust3r_inference.py \
    --index_txt_path "$1" \
    --gt_npy_path "$1" \
    --data_root "$2" \
    --output_dir "$(dirname $4)" \
    --model_path "$3" \
    --eval_mode R \
    --save_pose_path "$4" \
    --seed 0 \
    --test_only
