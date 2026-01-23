#!/bin/bash

# run.sh - Script to perform inference using trained model
# Usage: bash run.sh <base_model_path> <adapter_path> <input_json> <output_json>

BASE_MODEL_PATH=$1
ADAPTER_PATH=$2
INPUT_PATH=$3
OUTPUT_PATH=$4

# Check if all arguments are provided
if [ -z "$BASE_MODEL_PATH" ] || [ -z "$ADAPTER_PATH" ] || [ -z "$INPUT_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Error: Missing arguments"
    echo "Usage: bash run.sh <base_model_path> <adapter_path> <input_json> <output_json>"
    exit 1
fi

# Run inference
python3 ./code/predict.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --peft_path "$ADAPTER_PATH" \
    --test_data_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --max_new_tokens 128

echo "Inference completed. Output saved to $OUTPUT_PATH"