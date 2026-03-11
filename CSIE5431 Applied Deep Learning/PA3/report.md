# ADL HW3 Report: Retrieval-Augmented Generation
**Student ID:** R13921031

---

## Task Overview

This homework implements a **Retrieval-Augmented Generation (RAG)** system for open-domain question answering. The pipeline consists of three stages:

1. **Retriever** — given a query, retrieve the top-K relevant passages from a large corpus using a bi-encoder (dense retrieval)
2. **Reranker** — reorder the top-K passages using a cross-encoder for more precise relevance scoring
3. **Generator** — given the query and top-M reranked passages as context, generate a natural language answer using an LLM

The goal is to fine-tune the retriever and reranker, and design prompts to maximize generation quality.

---

## Dataset

| File | Description |
|------|-------------|
| `corpus.txt` | 100,000 passages to be retrieved. Each passage has `text`, `title`, `aid`, `bid`, `id` fields. |
| `train.txt` | 31,526 training examples with query, evidences (BM25 negatives), retrieval labels, and gold answers |
| `test_open.txt` | 3,342 public test examples |
| `qrels.txt` | Ground-truth mapping from query IDs to positive passage IDs |

**Data format example:**
```json
{
  "qid": "...",
  "rewrite": "Where is Malayali located?",
  "evidences": ["passage1...", "passage2..."],
  "retrieval_labels": [0, 0, 0, 0, 1],
  "answer": {"text": "Kerala", "answer_start": 0}
}
```

Each query has exactly one positive passage in `qrels.txt`. The `evidences` field contains BM25-sampled passages with binary relevance labels.

---

## Model Components

| Component | Model | Role |
|-----------|-------|------|
| Retriever | `intfloat/multilingual-e5-small` | Bi-encoder dense retrieval |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-12-v2` | Cross-encoder reranking |
| Generator | `Qwen/Qwen3-1.7B` (bf16, frozen) | Answer generation |

---

## Evaluation Metrics

- **Recall@10:** Whether the gold passage appears in the top-10 retrieved results. Binary per query, averaged across all queries.
- **MRR@10 (after rerank):** Mean Reciprocal Rank — measures the rank position of the first relevant passage in the reranked top-10. Higher rank = higher score.
- **Bi-Encoder CosSim:** Cosine similarity between the generated answer embedding and the gold answer embedding, using `sentence-transformers/all-MiniLM-L6-v2`. Measures semantic similarity of generated answers.

---

## Q1: Retriever & Reranker Tuning

### 1.1 Retriever Training

**Base model:** `intfloat/multilingual-e5-small`

**Training Data Construction:**
- **Query (anchor):** `rewrite` field from `train.txt`
- **Positive passage:** Gold passage looked up via `qrels.txt` → `corpus.txt`
- **Negative passages:** Up to 7 randomly sampled from `evidences` where `retrieval_label=0` (BM25 hard negatives provided in the dataset)
- Total training triplets: **126,104**

**Loss Function:** `MultipleNegativesRankingLoss` — for each (query, positive) pair, all other positives in the batch are treated as additional in-batch negatives. This is highly effective for bi-encoder training as it scales the number of negatives with batch size.

**Hyperparameters (v1):**
| Parameter | Value |
|-----------|-------|
| Base model | intfloat/multilingual-e5-small |
| Epochs | 3 |
| Batch size | 32 |
| Warmup steps | 100 |
| Optimizer | AdamW (default) |
| Negatives per query | up to 7 (BM25) |

**Training Loss Curve (v1):**
| Step | Epoch | Loss |
|------|-------|------|
| 50 | 0.025 | 0.197 |
| 100 | 0.051 | 0.174 |
| 200 | 0.102 | 0.182 |
| 300 | 0.152 | 0.146 |
| 500 | 0.254 | 0.145 |
| 1000 | 0.507 | 0.134 |
| 2000 | 1.015 | 0.121 |
| 3000 | 1.522 | 0.109 |
| 6000 | 3.045 | 0.094 |
| 9855 | 5.0 | 0.072 |

**Retriever v2** was also trained continuing from v1 checkpoint for 5 more epochs with batch size 64 and bf16. However, v1 performed slightly better on the test set, suggesting mild overfitting with additional epochs.

---

### 1.2 Reranker Training

**Base model:** `cross-encoder/ms-marco-MiniLM-L-12-v2`

#### Reranker v1

**Training Data Construction:**
- **Positive:** `(query, gold_passage, label=1)`
- **Negative:** `(query, neg_passage, label=0)` — up to 3 from BM25 evidences
- Total samples: ~125,000

**Loss:** `BinaryCrossEntropyLoss`

**Training Loss Curve (v1):**
| Epoch | Loss |
|-------|------|
| 1.0 | 0.2637 |
| 2.0 | 0.0919 |
| 3.0 | 0.1422 |

#### Reranker v2 (Hard Negatives)

To improve reranker quality, v2 uses **FAISS-mined hard negatives** — passages retrieved by the fine-tuned retriever that are not the gold passage. These are more challenging negatives than BM25 samples.

**Training Data Construction:**
- **Positive:** `(query, gold_passage, label=1)`
- **BM25 negatives:** up to 2 per query
- **Hard negatives from FAISS top-20:** up to 3 per query (retrieved but not gold)
- Total samples: ~190,000

**Hyperparameters (v2):**
| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch size | 64 |
| Warmup steps | 200 |
| Learning rate | 2e-5 |

**Training Loss Curve (v2):**
| Step | Epoch | Loss |
|------|-------|------|
| 50 | 0.017 | 0.657 |
| 100 | 0.034 | 0.431 |
| 200 | 0.068 | 0.270 |
| 300 | 0.101 | 0.234 |
| 500 | 0.169 | 0.191 |
| 1000 | 0.338 | 0.152 |
| 2000 | 0.677 | 0.121 |
| 5000 | 1.691 | 0.089 |
| 10000 | 3.382 | 0.071 |

**Why base reranker was submitted:**

Despite two rounds of reranker training, fine-tuned versions consistently underperformed the base model:

| Config | Recall@10 | MRR@10 | CosSim |
|--------|-----------|--------|--------|
| FT v1 retriever + Base reranker | **0.8725** | **0.7651** | **0.3798** |
| FT v2 retriever + FT v2 reranker | 0.8417 | 0.7224 | 0.3759 |

The base `ms-marco-MiniLM-L-12-v2` is extensively pre-trained on MS-MARCO, giving it strong cross-domain generalization. Fine-tuning on this relatively small dataset appears to reduce this generalization despite using hard negatives and more epochs. The base reranker was therefore retained for the final submission.

---

## Q2: Prompt Optimization

The prompt targets `Qwen3-1.7B` with `enable_thinking=False`. The goal is to produce concise, semantically accurate answers that maximize cosine similarity with gold answers.

**Prompt v1 (submitted — best):**
```
System:
You are a helpful assistant. Answer the question based on the provided passages. If the answer cannot be found, reply CANNOTANSWER.

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
System:
You are a precise QA assistant. Answer using only information from the provided passages. Be concise and direct. If the answer is not in the passages, reply exactly: CANNOTANSWER

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
System:
You are a helpful assistant. Answer the question based on the provided passages. If the answer cannot be found, reply CANNOTANSWER.

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
| v2 | 0.3193 | Over-constrained, model produces overly terse answers |
| v3 | 0.2709 | "Short phrase" instruction degrades answer completeness |

**Analysis:** The simplest prompt (v1) performed best. Over-specifying output constraints caused the model to produce incomplete answers, reducing cosine similarity with gold answers. The natural phrasing "Answer the question based on the provided passages" allows the model to calibrate response length and style appropriately. Instructing the model to extract short phrases (v3) was particularly harmful as many gold answers require multi-word or sentence-level responses.

---

## Q3: Additional Analysis

### 3.1 Full Model Comparison

| Config | Recall@10 | MRR@10 | CosSim |
|--------|-----------|--------|--------|
| Base retriever + Base reranker | 0.7151 | 0.6335 | 0.3531 |
| FT v1 retriever + Base reranker ✅ | **0.8725** | **0.7651** | **0.3798** |
| FT v2 retriever + Base reranker | 0.8665 | 0.7601 | 0.3791 |
| FT v2 retriever + FT v2 reranker | 0.8417 | 0.7224 | 0.3759 |

**Key finding:** Retriever fine-tuning provides the largest single improvement. The reranker contributes to MRR (passage ordering) but the base model generalizes better than fine-tuned versions on this dataset size.

### 3.2 Impact of Retriever Fine-tuning

| Metric | Base | FT v1 | Δ |
|--------|------|-------|---|
| Recall@10 | 0.7151 | 0.8725 | **+0.157** |
| MRR@10 | 0.6335 | 0.7651 | **+0.132** |
| CosSim | 0.3531 | 0.3798 | **+0.027** |

Retriever fine-tuning yielded the largest improvement across all metrics, confirming that **retrieval quality is the primary bottleneck** in the RAG pipeline. When the correct passage is not retrieved, the generator has no basis for a correct answer.

### 3.3 Effect of Training Epochs on Retriever

| Config | Recall@10 | MRR@10 | CosSim |
|--------|-----------|--------|--------|
| FT v1 (3 epochs) + Base reranker | **0.8725** | **0.7651** | **0.3798** |
| FT v2 (5 epochs, continued) + Base reranker | 0.8665 | 0.7601 | 0.3791 |

Continuing training beyond 3 epochs slightly hurt performance, suggesting the model begins to overfit to the training distribution. 3 epochs was optimal for this dataset size.

### 3.4 Performance: Gold Passage Retrieved vs. Not Retrieved

Based on the test set (Recall@10 = 0.8725), approximately 87.3% of queries had the gold passage in top-10:

| Group | Count (approx) | Impact on CosSim |
|-------|---------------|-----------------|
| Gold passage retrieved | ~2917 (87.3%) | Higher — model has correct context |
| Gold passage not retrieved | ~425 (12.7%) | Lower — model answers from irrelevant context or outputs CANNOTANSWER |

This directly explains the gap between Recall@10 and CosSim: the ~12.7% of queries where retrieval fails contributes near-zero cosine similarity, dragging down the overall average. Improving retrieval coverage is the most impactful direction for further improvement.
