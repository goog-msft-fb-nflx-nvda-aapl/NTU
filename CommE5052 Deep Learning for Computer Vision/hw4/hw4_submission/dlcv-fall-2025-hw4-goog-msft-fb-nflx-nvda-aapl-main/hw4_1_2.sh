#!/bin/bash
# $1: index txt path
# $2: original image pair directory
# $3: interpolated sequence directory
# $4: model checkpoint path
# $5: output prediction file path

python3 dust3r_inference.py \
    --index_txt_path "$1" \
    --gt_npy_path "$1" \
    --data_root "$2" \
    --interpolated_dir "$3" \
    --use_original_endpoints \
    --output_dir "$(dirname $5)" \
    --model_path "$4" \
    --eval_mode R \
    --save_pose_path "$5" \
    --seed 0 \
    --test_only
