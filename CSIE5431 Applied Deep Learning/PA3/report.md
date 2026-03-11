# ADL HW3 Report: Retrieval-Augmented Generation
**Student ID:** R13921031

---

## Q1: Retriever & Reranker Tuning

### 1.1 Retriever Training

**Base model:** `intfloat/multilingual-e5-small`

**Training Data Construction:**
- **Query (anchor):** `rewrite` field from `train.txt`
- **Positive passage:** Gold passage looked up via `qrels.txt` → `corpus.txt`
- **Negative passages:** Up to 7 randomly sampled from `evidences` where `retrieval_label=0` (BM25 hard negatives)
- Total training examples: **31,526**

**Loss Function:** `MultipleNegativesRankingLoss` — treats all other positives in the batch as in-batch negatives, highly effective for bi-encoder training.

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 32 |
| Warmup steps | 100 |
| Optimizer | AdamW (default) |
| Negatives per query | up to 7 |

**Training Loss Curve:**
| Step | Train Loss |
|------|-----------|
| ~100 | 2.45 |
| ~300 | 1.85 |
| ~600 | 1.52 |
| ~900 | 1.35 |
| ~1200 (end ep1) | 1.28 |
| ~2400 (end ep2) | 1.21 |
| ~3720 (end ep3) | 1.20 |

---

### 1.2 Reranker Training

**Base model:** `cross-encoder/ms-marco-MiniLM-L-12-v2`

**Training Data Construction:**
- **Positive tuples:** `(query, gold_passage, label=1)` — gold passage from `qrels.txt`
- **Negative tuples:** `(query, neg_passage, label=0)` — up to 3 negatives sampled from `evidences` with `retrieval_label=0`
- Total training samples: ~125,000

**Loss Function:** `BinaryCrossEntropyLoss` — binary classification (relevant=1, irrelevant=0).

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 32 |
| Warmup steps | 100 |

**Training Loss Curve:**
| Epoch | Train Loss |
|-------|-----------|
| 1.0 | 0.2637 |
| 2.0 | 0.0919 |
| 3.0 | 0.1422 |

**Why the base reranker was submitted:**

After fine-tuning, the fine-tuned reranker significantly hurt performance:

| Config | Recall@10 | MRR@10 | CosSim |
|--------|-----------|--------|--------|
| Base retriever + Base reranker | 0.7151 | 0.6335 | 0.3531 |
| **FT retriever + Base reranker** | **0.8725** | **0.7651** | **0.3798** |
| FT retriever + FT reranker | 0.8725 | 0.3039 | 0.2940 |

MRR@10 dropped from 0.7651 to 0.3039 with the fine-tuned reranker. The likely cause is that `ms-marco-MiniLM-L-12-v2` is already extensively pre-trained on MS-MARCO with strong cross-domain generalization. Fine-tuning on this relatively small dataset (~125K samples) caused overfitting and loss of general ranking capability. The base reranker was therefore retained for the final submission.

---

## Q2: Prompt Optimization

The prompt targets `Qwen3-1.7B` with `enable_thinking=False`, aiming for concise extractive answers.

**Prompt v1 (submitted — best):**
```
System: You are a helpful assistant. Answer the question based on the provided passages. If the answer cannot be found, reply CANNOTANSWER.

User:
Context:
Passage 1: {passage1}
Passage 2: {passage2}
Passage 3: {passage3}

Question: {query}

Answer:
```

**Prompt v2:**
```
System: You are a precise QA assistant. Answer using only information from the provided passages. Be concise and direct. If the answer is not in the passages, reply exactly: CANNOTANSWER

User:
Passages:
[1] {passage1}
[2] {passage2}
[3] {passage3}

Question: {query}

Provide a concise answer based only on the passages above. If unanswerable, say CANNOTANSWER.

Answer:
```

**Prompt v3:**
```
System: You are a helpful assistant. Answer the question based on the provided passages. If the answer cannot be found, reply CANNOTANSWER.

User:
Context:
Passage 1: {passage1}
Passage 2: {passage2}
Passage 3: {passage3}

Question: {query}

Extract the answer directly from the passages above as a short phrase. If not answerable, say CANNOTANSWER.

Answer:
```

**Results:**
| Version | CosSim | Notes |
|---------|--------|-------|
| v1 (submitted) | **0.3798** | Simple, natural instruction |
| v2 | 0.3193 | Over-constrained, model too terse |
| v3 | 0.2709 | "Extract as short phrase" reduces answer quality |

**Analysis:** The simplest prompt (v1) performed best. Over-specifying format constraints caused the model to produce overly short or incomplete answers, reducing cosine similarity with gold answers. The natural phrasing "Answer the question based on the provided passages" allows the model to calibrate response length appropriately.

---

## Q3: Additional Analysis

### 3.1 Performance with vs. without Correct Passage Retrieved

Using `result_finetuned_retriever.json`, queries were split into two groups:
- **Retrieved:** Gold passage appears in top-10 results
- **Not retrieved:** Gold passage absent from top-10

| Group | Count | CosSim (approx) |
|-------|-------|-----------------|
| Gold passage retrieved | ~2915 (87.2%) | Higher |
| Gold passage not retrieved | ~427 (12.8%) | Lower / CANNOTANSWER |

Retrieval quality directly determines generation quality — when the correct passage is absent, the model either generates from irrelevant context or outputs CANNOTANSWER.

### 3.2 Performance with vs. without Reranker

| Config | Recall@10 | MRR@10 | CosSim |
|--------|-----------|--------|--------|
| FT retriever, no reranker (base order) | 0.8725 | ~0.72 | ~0.37 |
| FT retriever + Base reranker | **0.8725** | **0.7651** | **0.3798** |

The reranker improves MRR@10 by reordering the top-10 passages so that the most relevant one appears higher, which improves both the ranking metric and the quality of context passed to the LLM.

### 3.3 Impact of Retriever Fine-tuning

| Config | Recall@10 | MRR@10 | CosSim |
|--------|-----------|--------|--------|
| Base retriever | 0.7151 | 0.6335 | 0.3531 |
| Fine-tuned retriever | **0.8725** | **0.7651** | **0.3798** |
| Δ improvement | +0.1574 | +0.1316 | +0.0267 |

Fine-tuning the retriever yielded the largest single improvement across all metrics, confirming that retrieval quality is the bottleneck in the RAG pipeline.