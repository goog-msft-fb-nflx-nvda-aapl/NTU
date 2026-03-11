import json, torch, random, os
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from datasets import Dataset

SEED = 42
random.seed(SEED)

# Load corpus
corpus = {}
with open("./data/corpus.txt") as f:
    for line in f:
        obj = json.loads(line)
        corpus[obj["id"]] = obj["text"]

# Load qrels
with open("./data/qrels.txt") as f:
    qrels = json.load(f)

# Build training examples
anchors, positives, negatives = [], [], []
with open("./data/train.txt") as f:
    for line in f:
        obj = json.loads(line)
        qid = obj["qid"]
        query = obj["rewrite"]
        evidences = obj["evidences"]
        labels = obj["retrieval_labels"]

        pos_pids = [pid for pid, lab in qrels.get(qid, {}).items() if lab != 0]
        if not pos_pids:
            continue
        pos_text = corpus.get(pos_pids[0], "")
        if not pos_text:
            continue

        neg_texts = [e for e, l in zip(evidences, labels) if l == 0]
        if not neg_texts:
            continue
        neg_texts = random.sample(neg_texts, min(7, len(neg_texts)))

        for neg in neg_texts:
            anchors.append(query)
            positives.append(pos_text)
            negatives.append(neg)

print(f"Training triplets: {len(anchors)}")

train_dataset = Dataset.from_dict({
    "anchor": anchors,
    "positive": positives,
    "negative": negatives,
})

# Continue from existing checkpoint
model = SentenceTransformer("./models/retriever")

loss_fn = losses.MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir="./models/retriever_v2",
    num_train_epochs=5,
    per_device_train_batch_size=64,
    warmup_steps=100,
    logging_steps=50,
    save_strategy="epoch",
    fp16=False,
    bf16=True,
    seed=SEED,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss_fn,
)

trainer.train()

# Save loss history
loss_history = [e for e in trainer.state.log_history if "loss" in e]
with open("./retriever_v2_loss.json", "w") as f:
    json.dump(loss_history, f, indent=2)
print("Loss saved to retriever_v2_loss.json")

model.save_pretrained("./models/retriever_v2/final")
print("Retriever v2 saved to ./models/retriever_v2/final")
