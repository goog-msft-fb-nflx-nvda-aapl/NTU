#!/bin/bash
# Download fine-tuned model weights

mkdir -p models/retriever
mkdir -p models/reranker

echo "Downloading fine-tuned retriever from Google Drive..."
# Replace RETRIEVER_GDRIVE_ID with your actual Google Drive file ID
gdown "https://drive.google.com/uc?id=RETRIEVER_GDRIVE_ID" -O retriever.zip
unzip -o retriever.zip -d models/retriever/
rm retriever.zip

echo "Downloading base reranker from HuggingFace..."
python -c "
from sentence_transformers import CrossEncoder
import os
os.makedirs('./models/reranker', exist_ok=True)
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
model.save_pretrained('./models/reranker')
print('Reranker downloaded.')
"

echo "Done. Models ready in ./models/"