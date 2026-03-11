# ADL HW3 - Retrieval-Augmented Generation
**Student ID:** R13921031

## Environment Setup

```bash
conda create -n rl_hw3 python=3.12 -y
conda activate rl_hw3
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.56.1 datasets==4.0.0 tqdm==4.67.1 \
    sentence-transformers==5.1.0 python-dotenv==1.1.1 \
    accelerate==1.10.1 gdown faiss-gpu-cu12==1.12.0
```

Create `.env` with your HuggingFace token:
```bash
echo 'hf_token="your_hf_token_here"' > .env
```

## Download Model Weights

```bash
bash download.sh
```

## Build Vector Database

```bash
python save_embeddings.py --retriever_model_path ./models/retriever --build_db
```

## Training

### Train Retriever
```bash
python code/train_retriever.py
```
- Base model: `intfloat/multilingual-e5-small`
- Loss: MultipleNegativesRankingLoss
- Epochs: 3, Batch size: 32
- Output: `./models/retriever`

### Train Reranker
```bash
python code/train_reranker.py
```
- Base model: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- Loss: BinaryCrossEntropyLoss
- Epochs: 3, Batch size: 32
- Note: Base model weights submitted — fine-tuned version underperformed (see report Q1)

## Inference

```bash
python inference_batch.py \
  --retriever_model_path ./models/retriever \
  --reranker_model_path ./models/reranker \
  --test_data_path ./data/test_open.txt
```

## Results

| Metric | Baseline | Fine-tuned | Target |
|--------|----------|------------|--------|
| Recall@10 | 0.7151 | **0.8725** | >0.780 ✅ |
| MRR@10 | 0.6335 | **0.7651** | >0.695 ✅ |
| Bi-Encoder CosSim | 0.3531 | **0.3798** | >0.340 ✅ |