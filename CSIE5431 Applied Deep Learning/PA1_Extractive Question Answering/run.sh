#!/bin/bash

# Check arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: bash run.sh <context.json> <test.json> <output.csv>"
    exit 1
fi

CONTEXT_PATH=$1
TEST_PATH=$2
OUTPUT_PATH=$3

echo "Running inference..."
echo "Context: ${CONTEXT_PATH}"
echo "Test: ${TEST_PATH}"
echo "Output: ${OUTPUT_PATH}"

# Run inference script
python3 inference.py \
    --context_file "${CONTEXT_PATH}" \
    --test_file "${TEST_PATH}" \
    --paragraph_model ./paragraph_model \
    --span_model ./span_model_epoch_5 \
    --output_file "${OUTPUT_PATH}"

echo "Inference completed! Predictions saved to ${OUTPUT_PATH}"
