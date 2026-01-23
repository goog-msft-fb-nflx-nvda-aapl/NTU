#!/bin/bash

mkdir -p adapter_checkpoint

# Replace with your actual Google Drive file ID
FILE_ID="1whr81uneSOKDaC1g4GjhXv6M9iDRiLQw"

echo "Downloading checkpoint from Google Drive..."
gdown --id $FILE_ID -O adapter_checkpoint.zip

if [ $? -ne 0 ]; then
    echo "✗ Download failed!"
    exit 1
fi

echo "Extracting checkpoint..."
unzip -q adapter_checkpoint.zip -d adapter_checkpoint

if [ $? -ne 0 ]; then
    echo "✗ Extraction failed!"
    exit 1
fi

rm adapter_checkpoint.zip

# Verify files exist
if [ -f "adapter_checkpoint/adapter_config.json" ] && [ -f "adapter_checkpoint/adapter_model.safetensors" ]; then
    echo "✓ adapter_config.json found"
    echo "✓ adapter_model.safetensors found"
    echo "✓ Download successful!"
else
    echo "✗ Required files not found!"
    echo "Expected:"
    echo "  - adapter_checkpoint/adapter_config.json"
    echo "  - adapter_checkpoint/adapter_model.safetensors"
    exit 1
fi