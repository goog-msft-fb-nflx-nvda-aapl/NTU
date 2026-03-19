#!/bin/bash

# $1: annotation json file
# $2: images root folder
# $3: llava weight path
# $4: output pred json file

cd "$(dirname "$0")"

python infer_p1.py \
  --annotation_file "$1" \
  --images_root "$2" \
  --llava_weight_path "$3" \
  --pred_file "$4"