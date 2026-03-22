#!/bin/bash
# GaussianOne: Drop images, get rendered video
# Usage: bash gaussianone.sh <input_images_folder> <output_name>

set -e

INPUT=$1
OUTPUT_NAME=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output/$OUTPUT_NAME"

if [ -z "$INPUT" ] || [ -z "$OUTPUT_NAME" ]; then
    echo "Usage: bash gaussianone.sh <input_images_folder> <output_name>"
    exit 1
fi

echo "==> [1/5] Running COLMAP on $INPUT"
python $SCRIPT_DIR/convert.py -s $INPUT

echo "==> [2/5] Training 3D Gaussian Splatting"
python $SCRIPT_DIR/train.py -s $INPUT --model_path $OUTPUT_DIR

echo "==> [3/5] Rendering novel views"
python $SCRIPT_DIR/render.py -m $OUTPUT_DIR --skip_test

echo "==> [4/5] Computing metrics (PSNR/SSIM/LPIPS)"
mkdir -p $OUTPUT_DIR/test
cp -r $OUTPUT_DIR/train/ours_30000 $OUTPUT_DIR/test/ours_30000
python $SCRIPT_DIR/metrics.py -m $OUTPUT_DIR

echo "==> [5/5] Generating output video"
W=$(python3 -c "from PIL import Image; import os; f=sorted(os.listdir('$OUTPUT_DIR/train/ours_30000/renders'))[0]; w,h=Image.open('$OUTPUT_DIR/train/ours_30000/renders/'+f).size; print(w if w%2==0 else w-1)")
H=$(python3 -c "from PIL import Image; import os; f=sorted(os.listdir('$OUTPUT_DIR/train/ours_30000/renders'))[0]; w,h=Image.open('$OUTPUT_DIR/train/ours_30000/renders/'+f).size; print(h if h%2==0 else h-1)")
ffmpeg -y -framerate 30 -i $OUTPUT_DIR/train/ours_30000/renders/%05d.png \
    -vf "scale=$W:$H" -c:v libx264 -pix_fmt yuv420p $OUTPUT_DIR/${OUTPUT_NAME}_render.mp4

echo ""
echo "Done! Output video: $OUTPUT_DIR/${OUTPUT_NAME}_render.mp4"
