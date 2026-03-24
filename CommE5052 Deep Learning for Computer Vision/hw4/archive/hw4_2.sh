#!/bin/bash
# hw4_2.sh - Sparse-View 3D Gaussian Splatting
# Usage: bash hw4_2.sh $1 $2
# $1: Path to folder containing test and train data
# $2: Path to output png files

DATA_PATH=$(readlink -f "$1")
OUTPUT_PATH=$(readlink -f "$2")
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
INSTANTSPLAT_DIR="${SCRIPT_DIR}/InstantSplat"
CKPT_PATH="${INSTANTSPLAT_DIR}/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
MODEL_PATH="${SCRIPT_DIR}/results/p2/final"
N_VIEWS=3
ITERATIONS=1000

mkdir -p "${MODEL_PATH}"
mkdir -p "${OUTPUT_PATH}"

cd "${INSTANTSPLAT_DIR}"

echo "Step 1: Geometry Initialization..."
python3 init_geo.py \
    -s "${DATA_PATH}" \
    -m "${MODEL_PATH}" \
    --ckpt_path "${CKPT_PATH}" \
    --n_views ${N_VIEWS} \
    --focal_avg \
    --co_vis_dsp \
    --conf_aware_ranking \
    --infer_video

echo "Step 2: Training..."
python3 train.py \
    -s "${DATA_PATH}" \
    -m "${MODEL_PATH}" \
    -r 1 \
    --n_views ${N_VIEWS} \
    --iterations ${ITERATIONS} \
    --pp_optimizer

echo "Step 3: Rendering..."
python3 render.py \
    -s "${DATA_PATH}" \
    -m "${MODEL_PATH}" \
    -r 1 \
    --n_views ${N_VIEWS} \
    --iterations ${ITERATIONS} \
    --eval

echo "Step 4: Copying output images..."
cp "${MODEL_PATH}/test/ours_${ITERATIONS}/renders/"*.png "${OUTPUT_PATH}/"

echo "Done. Output saved to ${OUTPUT_PATH}"
