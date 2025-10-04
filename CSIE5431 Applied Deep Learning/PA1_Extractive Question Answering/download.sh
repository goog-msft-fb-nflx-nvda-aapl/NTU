#!/bin/bash

# Download trained models from Google Drive
# Replace FILEID with your actual Google Drive file ID

echo "Downloading models..."

# Install gdown if not available
pip install gdown -q

# Download models (replace FILEID with your Google Drive file ID)
# To get file ID: Share your file -> Copy link -> Extract ID from URL
# Example: https://drive.google.com/file/d/FILEID/view?usp=sharing
gdown 'https://drive.google.com/uc?id=1QoKxYKhDRXhS869g0f1J1H3nTRqvb_va' -O models.zip

# Extract models
echo "Extracting models..."
unzip -q models.zip

# Verify extraction
if [ -d "paragraph_model" ] && [ -d "span_model_epoch_5" ]; then
    echo "Models downloaded and extracted successfully"
else
    echo "Error: Model directories not found"
    exit 1
fi

# Clean up
rm models.zip

echo "Download complete!"
