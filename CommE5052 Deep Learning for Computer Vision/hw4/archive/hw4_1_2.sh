#!/bin/bash
# hw4_1_2.sh - Inference on Interpolated Sequences
# Usage: bash hw4_1_2.sh $1 $2 $3 $4 $5
# $1: Path to the TXT file containing the index
# $2: Path to the original image pair directory
# $3: Path to the interpolated sequence directory
# $4: Path to the model checkpoint
# $5: Path for the output prediction file

INDEX_TXT_PATH=$1
DATA_ROOT=$2
INTERPOLATED_DIR=$3
MODEL_PATH=$4
SAVE_POSE_PATH=$5

python3 dust3r_inference.py \
    --index_txt_path "${INDEX_TXT_PATH}" \
    --gt_npy_path "${INDEX_TXT_PATH}" \
    --data_root "${DATA_ROOT}" \
    --interpolated_dir "${INTERPOLATED_DIR}" \
    --output_dir "$(dirname ${SAVE_POSE_PATH})" \
    --eval_mode R \
    --model_path "${MODEL_PATH}" \
    --use_model Dust3R \
    --use_original_endpoints \
    --save_pose_path "${SAVE_POSE_PATH}" \
    --test_only \
    --seed 0
