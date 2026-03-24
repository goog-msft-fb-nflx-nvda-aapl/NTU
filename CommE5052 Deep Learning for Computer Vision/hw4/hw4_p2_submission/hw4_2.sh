#!/bin/bash
# Usage: bash hw4_2.sh <data_dir> <output_dir>
# <data_dir>: folder with images/ and sparse_3/{0,1}/
# <output_dir>: where rendered test PNGs are saved

DATA_DIR=$1
OUTPUT_DIR=$2

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
GS_DIR=~/vfx-final-project/gaussian-splatting
P2_WORK=/tmp/p2_hw4_work_$$
MODEL_OUT=/tmp/p2_hw4_model_$$

mkdir -p "$OUTPUT_DIR"
mkdir -p "$P2_WORK/data/sparse/0"
mkdir -p "$P2_WORK/data/sparse/1"

# Prepare COLMAP-format data
cp "$DATA_DIR/sparse_3/0/cameras.txt"  "$P2_WORK/data/sparse/0/"
cp "$DATA_DIR/sparse_3/0/images.txt"   "$P2_WORK/data/sparse/0/"
cp "$DATA_DIR/sparse_3/0/points3D.ply" "$P2_WORK/data/sparse/0/"
echo "# 3D point list" > "$P2_WORK/data/sparse/0/points3D.txt"

cp "$DATA_DIR/sparse_3/1/cameras.txt"  "$P2_WORK/data/sparse/1/"
cp "$DATA_DIR/sparse_3/1/images.txt"   "$P2_WORK/data/sparse/1/"
echo "# 3D point list" > "$P2_WORK/data/sparse/1/points3D.txt"

ln -sfn "$(realpath $DATA_DIR/images)" "$P2_WORK/data/images"

# Train 3DGS
cd "$GS_DIR"
PYTHONNOUSERSITE=1 conda run -n rl_hw3 python train.py \
    -s "$P2_WORK/data" \
    -m "$MODEL_OUT" \
    --iterations 7000 \
    --test_iterations 7000 \
    --save_iterations 7000 \
    --densification_interval 100 \
    --densify_until_iter 3000 \
    --densify_grad_threshold 0.0002 \
    --position_lr_init 0.00032 \
    --position_lr_final 0.0000032 \
    --opacity_lr 0.05 \
    --scaling_lr 0.005 \
    --eval

# Render test views
PYTHONNOUSERSITE=1 conda run -n rl_hw3 python "$REPO_DIR/render_test.py" \
    --model_path "$MODEL_OUT" \
    --test_sparse_dir "$P2_WORK/data/sparse/1" \
    --images_dir "$P2_WORK/data/images" \
    --output_dir "$OUTPUT_DIR" \
    --iteration 7000

# Cleanup
rm -rf "$P2_WORK" "$MODEL_OUT"
