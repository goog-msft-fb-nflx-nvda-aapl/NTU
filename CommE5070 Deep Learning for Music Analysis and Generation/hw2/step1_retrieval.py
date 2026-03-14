"""
Step 1: CLAP-based Audio Retrieval (Improved)
For each target music, find top-3 most similar songs from reference_music_list.

Key improvement over v1:
  - Average embeddings over multiple 10s windows (instead of only first 10s)
    This gives a more representative embedding for 60s tracks.
"""

import os
import glob
import json
import torch
import numpy as np
import laion_clap
import librosa
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.expanduser("~/CommE5070/home/fundwotsai/Deep_MIR_hw2")
TARGET_DIR = os.path.join(BASE_DIR, "target_music_list_60s")
REF_DIR    = os.path.join(BASE_DIR, "referecne_music_list_60s")
OUT_DIR    = os.path.join(BASE_DIR, "retrieval_results")
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_RATE  = 48000   # CLAP expects 48 kHz
WINDOW_SEC   = 10      # CLAP's native window size
HOP_SEC      = 10      # non-overlapping windows (set < WINDOW_SEC for overlap)
TOP_K        = 3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load CLAP model ───────────────────────────────────────────────────────────
print("Loading CLAP model...")
model = laion_clap.CLAP_Module(enable_fusion=False, device=DEVICE)
model.load_ckpt()   # downloads music_audioset_epoch_15_esc_90.14.pt
model.eval()
print(f"CLAP loaded on {DEVICE}\n")

# ── Helper: get multi-window averaged embedding ───────────────────────────────
def embed_file(path: str) -> np.ndarray:
    """
    Load audio, slice into non-overlapping 10s windows, embed each,
    then return the L2-normalised mean embedding (512-d).
    Falls back to a zero vector on error.
    """
    window_len = SAMPLE_RATE * WINDOW_SEC
    hop_len    = SAMPLE_RATE * HOP_SEC

    try:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  [WARN] Cannot load {Path(path).name}: {e}")
        return np.zeros(512, dtype=np.float32)

    # Build windows
    chunks = []
    start = 0
    while start + window_len <= len(audio):
        chunk = audio[start : start + window_len].astype(np.float32)
        chunks.append(chunk)
        start += hop_len

    # If audio shorter than one window, pad it
    if len(chunks) == 0:
        chunk = audio.astype(np.float32)
        chunk = np.pad(chunk, (0, window_len - len(chunk)))
        chunks = [chunk]

    # Embed all windows in one batch
    try:
        embs = model.get_audio_embedding_from_data(x=chunks, use_tensor=False)  # (W, 512)
    except Exception as e:
        print(f"  [WARN] Embedding failed for {Path(path).name}: {e}")
        return np.zeros(512, dtype=np.float32)

    # Average over windows, then L2-normalise
    mean_emb = embs.mean(axis=0)
    norm = np.linalg.norm(mean_emb) + 1e-8
    return (mean_emb / norm).astype(np.float32)

# ── Embed all files ───────────────────────────────────────────────────────────
def embed_dir(paths: list, label: str) -> np.ndarray:
    embs = []
    for i, p in enumerate(paths):
        name = Path(p).name
        print(f"  [{i+1}/{len(paths)}] {name}")
        embs.append(embed_file(p))
    return np.array(embs)   # (N, 512)

# ── Cosine similarity (embeddings already L2-normalised) ─────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Extra safety normalisation
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T   # (N_target, N_ref)

# ── Gather file lists ─────────────────────────────────────────────────────────
target_files = sorted(glob.glob(os.path.join(TARGET_DIR, "*")))
ref_files    = sorted(glob.glob(os.path.join(REF_DIR,    "*")))

print(f"Targets   : {len(target_files)}")
print(f"References: {len(ref_files)}\n")

# ── Embed ─────────────────────────────────────────────────────────────────────
print("Embedding target files...")
target_embs = embed_dir(target_files, "target")

print("\nEmbedding reference files...")
ref_embs = embed_dir(ref_files, "reference")

# ── Retrieve top-K ────────────────────────────────────────────────────────────
sim_matrix = cosine_sim(target_embs, ref_embs)   # (N_target, N_ref)

print("\n" + "="*70)
print("RETRIEVAL RESULTS")
print("="*70)

results = {}
for i, tgt in enumerate(target_files):
    tgt_name = Path(tgt).name
    scores   = sim_matrix[i]
    top_idx  = np.argsort(scores)[::-1][:TOP_K]

    print(f"\nTarget: {tgt_name}")
    top_refs = []
    for rank, idx in enumerate(top_idx):
        ref_name = Path(ref_files[idx]).name
        sim_val  = float(scores[idx])
        print(f"  #{rank+1}  sim={sim_val:.4f}  {ref_name}")
        top_refs.append({"rank": rank + 1, "file": ref_name, "similarity": sim_val})

    results[tgt_name] = top_refs

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "retrieval_results_v2.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved to {out_path}")