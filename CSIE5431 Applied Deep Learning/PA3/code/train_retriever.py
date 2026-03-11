import json, torch, random
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from tqdm import tqdm
import os

# Load train data
train_path = "./data/train.txt"
corpus_path = "./data/corpus.txt"

# Load corpus
corpus = {}
with open(corpus_path) as f:
    for line in f:
        obj = json.loads(line)
        corpus[obj["id"]] = obj["text"]

# Load qrels
with open("./data/qrels.txt") as f:
    qrels = json.load(f)

# Build training examples
train_examples = []
with open(train_path) as f:
    for line in f:
        obj = json.loads(line)
        qid = obj["qid"]
        query = obj["rewrite"]
        evidences = obj["evidences"]
        labels = obj["retrieval_labels"]

        # Get positive passage from qrels
        pos_pids = [pid for pid, lab in qrels.get(qid, {}).items() if lab != 0]
        if not pos_pids:
            continue
        pos_text = corpus.get(pos_pids[0], "")
        if not pos_text:
            continue

        # Negatives from evidences
        neg_texts = [e for e, l in zip(evidences, labels) if l == 0]
        if not neg_texts:
            continue

        # Sample up to 7 negatives
        neg_texts = random.sample(neg_texts, min(7, len(neg_texts)))

        train_examples.append(InputExample(texts=[query, pos_text] + neg_texts))

print(f"Training examples: {len(train_examples)}")

model = SentenceTransformer("intfloat/multilingual-e5-small")

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="./models/retriever",
    show_progress_bar=True,
    checkpoint_save_steps=500,
    checkpoint_path="./models/retriever_ckpts"
)

print("Retriever saved to ./models/retriever")