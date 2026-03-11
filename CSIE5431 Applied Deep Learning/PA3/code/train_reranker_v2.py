import json, random, os, torch
import faiss
import sqlite3
import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

SEED = 42
random.seed(SEED)

# Load corpus
corpus = {}
with open("./data/corpus.txt") as f:
    for line in f:
        obj = json.loads(line)
        corpus[obj["id"]] = obj["text"]

with open("./data/qrels.txt") as f:
    qrels = json.load(f)

# Load retriever + faiss for hard negative mining
print("Loading retriever and FAISS index for hard negative mining...")
retriever = SentenceTransformer("./models/retriever", device="cuda")
index = faiss.read_index("./vector_database/passage_index.faiss")

conn = sqlite3.connect("./vector_database/passage_store.db")
cur = conn.cursor()

# Build training data with hard negatives
queries_list, passages_list, labels_list = [], [], []

with open("./data/train.txt") as f:
    train_data = [json.loads(line) for line in f if line.strip()]

print(f"Mining hard negatives for {len(train_data)} queries...")
batch_size = 256
for b_start in range(0, len(train_data), batch_size):
    batch = train_data[b_start:b_start+batch_size]
    batch_queries = ["query: " + obj["rewrite"] for obj in batch]

    q_embs = retriever.encode(batch_queries, convert_to_numpy=True,
                               normalize_embeddings=True, batch_size=batch_size)
    D, I = index.search(q_embs, 20)  # retrieve top-20 as hard negative pool

    need_rowids = set(int(rid) for row in I for rid in row.tolist())
    placeholders = ",".join(["?"] * len(need_rowids))
    rows = cur.execute(
        f"SELECT rowid, pid, text FROM passages WHERE rowid IN ({placeholders})",
        tuple(need_rowids)
    ).fetchall()
    rowid2pt = {rid: (pid, text) for (rid, pid, text) in rows}

    for i, obj in enumerate(batch):
        qid = obj["qid"]
        query = obj["rewrite"]
        evidences = obj["evidences"]
        label_list = obj["retrieval_labels"]

        pos_pids = [pid for pid, lab in qrels.get(qid, {}).items() if lab != 0]
        if not pos_pids:
            continue
        pos_text = corpus.get(pos_pids[0], "")
        if not pos_text:
            continue

        # Add positive
        queries_list.append(query)
        passages_list.append(pos_text)
        labels_list.append(1.0)

        # BM25 negatives
        bm25_negs = [e for e, l in zip(evidences, label_list) if l == 0]
        for neg in random.sample(bm25_negs, min(2, len(bm25_negs))):
            queries_list.append(query)
            passages_list.append(neg)
            labels_list.append(0.0)

        # Hard negatives from FAISS (retrieved but not gold)
        hard_negs = []
        for rid in I[i].tolist():
            tup = rowid2pt.get(int(rid))
            if tup is None:
                continue
            pid, text = tup
            if pid not in pos_pids:
                hard_negs.append(text)
        for neg in random.sample(hard_negs, min(3, len(hard_negs))):
            queries_list.append(query)
            passages_list.append(neg)
            labels_list.append(0.0)

    if (b_start // batch_size) % 5 == 0:
        print(f"Processed {b_start+len(batch)}/{len(train_data)} queries, {len(queries_list)} samples so far")

conn.close()
del retriever
torch.cuda.empty_cache()

print(f"Total training samples: {len(queries_list)}")

train_dataset = Dataset.from_dict({
    "query": queries_list,
    "passage": passages_list,
    "label": labels_list,
})

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
loss = BinaryCrossEntropyLoss(model)

args = CrossEncoderTrainingArguments(
    output_dir="./models/reranker_v2",
    num_train_epochs=5,
    per_device_train_batch_size=64,
    warmup_steps=200,
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=2e-5,
    seed=SEED,
)

trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)

trainer.train()

# Save loss history
loss_history = [e for e in trainer.state.log_history if "loss" in e]
with open("./reranker_v2_loss.json", "w") as f:
    json.dump(loss_history, f, indent=2)
print("Loss saved to reranker_v2_loss.json")

model.save_pretrained("./models/reranker_v2/final")
print("Reranker v2 saved to ./models/reranker_v2/final")
