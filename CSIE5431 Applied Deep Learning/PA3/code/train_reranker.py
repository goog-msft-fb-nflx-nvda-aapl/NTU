import json, random, os, torch
from datasets import Dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

corpus = {}
with open("./data/corpus.txt") as f:
    for line in f:
        obj = json.loads(line)
        corpus[obj["id"]] = obj["text"]

with open("./data/qrels.txt") as f:
    qrels = json.load(f)

queries, passages, labels = [], [], []
with open("./data/train.txt") as f:
    for line in f:
        obj = json.loads(line)
        qid = obj["qid"]
        query = obj["rewrite"]
        evidences = obj["evidences"]
        label_list = obj["retrieval_labels"]

        pos_pids = [pid for pid, lab in qrels.get(qid, {}).items() if lab != 0]
        if not pos_pids: continue
        pos_text = corpus.get(pos_pids[0], "")
        if not pos_text: continue

        queries.append(query); passages.append(pos_text); labels.append(1.0)

        neg_texts = [e for e, l in zip(evidences, label_list) if l == 0]
        for neg in random.sample(neg_texts, min(3, len(neg_texts))):
            queries.append(query); passages.append(neg); labels.append(0.0)

print(f"Training samples: {len(queries)}")

train_dataset = Dataset.from_dict({"query": queries, "passage": passages, "label": labels})

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
loss = BinaryCrossEntropyLoss(model)

args = CrossEncoderTrainingArguments(
    output_dir="./models/reranker",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    warmup_steps=100,
    save_strategy="epoch",
    logging_steps=50,
)

trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)

trainer.train()
model.save_pretrained("./models/reranker")
print("Reranker saved to ./models/reranker")