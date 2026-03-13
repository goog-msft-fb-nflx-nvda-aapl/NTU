# Deep MIR HW2 Report — Controllable Text-to-Music Generation

---

## 1. Problem Description

This homework addresses the task of **controllable text-to-music generation**. Given a set of 9 target music clips (60 seconds each), the goal is to:

1. **Retrieve** the most similar songs from a reference music list using audio embeddings.
2. **Caption** each target track using an Audio Language Model (ALM).
3. **Generate** music conditioned on the extracted text and melody using a pretrained controllable generation model.
4. **Evaluate** the results using CLAP similarity, Melody Accuracy, and Meta Audiobox Aesthetics scores.

The target music list contains:
- 3 drum/rhythm solo tracks (country, jazz, rock)
- 2 Chinese bamboo flute solos
- 2 Chinese flute + piano duets
- 2 piano solos (classical Western)

No model training or fine-tuning is performed. The entire pipeline uses pretrained models zero-shot.

---

## 2. Implementation Details

### 2.1 Pipeline Overview

```
target_music_list_60s
        │
        ├─► [Step 1] CLAP Retrieval ──────────────► retrieval_results.json
        │
        ├─► [Step 2] Qwen2-Audio Captioning ──────► captions.json
        │
        ├─► [Step 3] MusicGen-Melody Generation ──► outputs/generated/*.wav
        │               (text + melody condition)
        │
        └─► [Step 4] Evaluation ──────────────────► eval_results.json
                        CLAP / Melody Acc / Audiobox
```

### 2.2 Step 1 — Retrieval

- **Model:** LAION-CLAP (`music_speech_audioset_epoch_15_esc_89.98.pt`)
- **Method:** Multi-window averaging — each audio file is encoded in overlapping windows; embeddings are averaged to produce a single track-level embedding. Cosine similarity is used to rank reference tracks.
- **Output:** Top-3 most similar reference tracks per target.

| Target | Rank 1 | Similarity |
|---|---|---|
| 10_country_114_beat_4-4.wav | 81_dance-breakbeat_170_beat_4-4.wav | 0.830 |
| 4_jazz_120_beat_3-4.wav | 82_neworleans-funk_84_beat_4-4.wav | 0.912 |
| 6_rock_102_beat_3-4.wav | 82_neworleans-funk_84_beat_4-4.wav | 0.976 |
| Hedwig's theme x dizi | 安童哥買菜.MP3 | 0.787 |
| Mussorgsky: Pictures at an Exhibition | MIDI-Unprocessed_XP_21... | 0.698 |
| Spirited Away OST (Piano Cover) | MIDI-Unprocessed_XP_21... | 0.719 |
| IRIS OUT / 米津玄師 (Piano Solo) | 山樂III 繆儀琳、小巨人團演奏.mp3 | 0.553 |
| 菊花台 (Flute + Piano) | MIDI-Unprocessed_XP_21... | 0.611 |
| 竹笛｜这世界那么多人 | MIDI-Unprocessed_XP_21... | 0.628 |

### 2.3 Step 2 — Audio Captioning

- **Model:** `Qwen/Qwen2-Audio-7B-Instruct` (HuggingFace, fp16, `device_map="auto"`)
- **Prompt:** Detailed music description including instruments, tempo, mood, genre, melody, rhythm, and notable features.
- **Output:** One caption per track, saved to `outputs/captions/captions.json`.

Example captions:

| Track | Caption (truncated) |
|---|---|
| 4_jazz_120_beat_3-4.wav | *"Jazz-inspired instrumental with slow tempo ~89 BPM, 4/4 time, chord progression through E major, F# minor..."* |
| Hedwig's theme x dizi | *"Instrumental flute in E minor at 93.8 BPM, waltz-like rhythm, calming and peaceful mood..."* |
| 竹笛｜这世界那么多人 | *"Flute playing main melody, slow 89.6 BPM, C major, emotional and sad, classical genre..."* |

### 2.4 Step 3 — Controllable Text-to-Music Generation

- **Model:** `facebook/musicgen-melody` (AudioCraft)
- **Text condition:** Caption from Step 2 (Medium level)
- **Melody condition:** Chromagram extracted from target audio via `generate_with_chroma()` — MIR-based, rule-compliant
- **Duration:** 30 seconds
- **CFG scale:** 3.0
- **Sampling:** Greedy (top_k=250, top_p=0.0, temperature=1.0)

The melody conditioning uses MusicGen's built-in chroma-based melody prefix, which extracts a chromagram from the reference waveform and conditions the transformer decoder on it. This satisfies the "Medium" condition requirement (melody extracted via MIR tool).

### 2.5 Step 4 — Evaluation

- **CLAP:** `laion_clap` with `music_speech_audioset_epoch_15_esc_89.98.pt`; audio loaded via librosa at 48 kHz, 10s window
- **Melody Accuracy:** Provided `Melody_acc.py` — one-hot chromagram argmax matching between trimmed target and generated
- **Meta Audiobox Aesthetics:** `audiobox_aesthetics` pip package (`facebook/audiobox-aesthetics`); scores CE, CU, PC, PQ
- Target audio trimmed to 30 seconds before evaluation to match generated length

---

## 3. Evaluation Results

### 3.1 Generation Metrics

| Track | tgt↔txt | txt↔gen | tgt↔gen | MelAcc | T-CE | T-CU | T-PC | T-PQ | G-CE | G-CU | G-PC | G-PQ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10_country_114_beat_4-4.wav | 0.107 | 0.356 | 0.394 | 0.1354 | 5.96 | 7.54 | 2.57 | 7.23 | 6.53 | 7.27 | 5.67 | 6.87 |
| 4_jazz_120_beat_3-4.wav | 0.322 | 0.297 | 0.625 | 0.2740 | 6.27 | 7.83 | 3.43 | 7.57 | 7.29 | 7.83 | 5.93 | 6.93 |
| 6_rock_102_beat_3-4.wav | 0.296 | 0.452 | 0.649 | 0.1041 | 6.79 | 8.03 | 3.12 | 7.75 | 7.01 | 7.88 | 5.81 | 7.38 |
| Hedwig's theme x dizi | 0.438 | 0.432 | 0.586 | 0.6587 | 7.02 | 7.33 | 3.02 | 7.38 | 7.08 | 7.13 | 4.74 | 6.91 |
| Mussorgsky: Pictures at an Exhibition | 0.307 | 0.183 | 0.448 | 0.5341 | 7.41 | 7.97 | 2.51 | 7.86 | 7.53 | 7.81 | 3.44 | 7.37 |
| Spirited Away OST (Piano Cover) | 0.435 | 0.276 | 0.548 | 0.5468 | 7.80 | 8.13 | 3.09 | 8.00 | 7.46 | 7.84 | 4.64 | 7.68 |
| IRIS OUT / 米津玄師 (Piano Solo) | 0.371 | 0.354 | 0.570 | 0.5151 | 7.47 | 7.87 | 3.84 | 8.00 | 7.46 | 7.82 | 4.18 | 7.33 |
| 菊花台 (Flute + Piano) | 0.253 | 0.368 | 0.518 | 0.4721 | 7.72 | 8.06 | 4.37 | 8.01 | 7.25 | 7.85 | 4.73 | 7.52 |
| 竹笛｜这世界那么多人 | 0.329 | 0.498 | 0.577 | 0.5789 | 7.66 | 8.03 | 3.76 | 7.78 | 7.68 | 8.00 | 4.53 | 7.67 |

Column legend: `tgt↔txt` = CLAP(target, text), `txt↔gen` = CLAP(text, generated), `tgt↔gen` = CLAP(target, generated), `T-*` = Audiobox scores for target, `G-*` = Audiobox scores for generated.

### 3.2 Retrieval Metrics

Retrieval used CLAP cosine similarity. Top-1 similarities ranged from 0.553 (IRIS OUT piano solo — hardest, no close match in reference set) to 0.976 (rock drum track — strong match with funk/rock references). Semantically the retrieval is sensible: Chinese flute tracks retrieve Chinese traditional music; piano solos retrieve MIDI piano recordings; drum tracks retrieve drum-heavy genres.

---

## 4. Observations and Reflection

### What worked well

- **Retrieval** performed semantically well for tracks with clear genre/instrumentation matches (rock → funk, jazz → New Orleans funk, flute → Chinese traditional).
- **Melody accuracy** was high for melodic tracks (Hedwig's theme 0.659, Spirited Away 0.547, 竹笛 0.579), confirming that MusicGen's chroma conditioning captures melodic contour effectively.
- **Audiobox PQ and CU** scores for generated audio are close to targets, indicating reasonable production quality from MusicGen.

### What did not work well

- **Melody accuracy was poor for rhythm-only tracks** (country 0.135, rock 0.104). These are drum solos with no stable melodic pitch class, so the chromagram-based melody condition provides no meaningful signal and accuracy degrades to near-random.
- **CLAP tgt↔txt scores are low** for several tracks (country 0.107, 菊花台 0.253). Qwen2-Audio captions sometimes hallucinated incorrect tempo/genre details — e.g., labeling a 4/4 country beat as "rock/electronic pop." This propagates error into generation.
- **Generated PC (Production Complexity) scores consistently drop** compared to targets (e.g., Mussorgsky: 2.51 → 3.44). MusicGen produces cleaner, simpler audio than real studio or live recordings, which lowers perceived complexity.
- **Retrieval for Japanese/Chinese pop piano tracks** (IRIS OUT: 0.553) was weak — no close stylistic matches existed in the reference set.

### TODO — Potential Improvements

- **Better captioning:** Use Audio Flamingo 3 or LP-MusicCaps for more accurate music-specific captions (correct tempo, key, genre). Qwen2-Audio is a general-purpose model not specialized for music description.
- **Better generation model:** Replace MusicGen-Melody with **MuseControlLite**, which supports rhythm, dynamics, and melody conditioning simultaneously in a latent diffusion framework, potentially improving both melody accuracy and production quality.
- **Rhythm conditioning for drum tracks:** Use beat/downbeat extraction (e.g., madmom, librosa beat tracker) as an explicit condition via MuseControlLite's rhythm control — this would directly address the low melody accuracy on drum-only tracks.
- **Per-track CFG tuning (Strong condition):** Adjust `cfg_coef` per track rather than using a fixed value of 3.0. Higher CFG (~5–7) for text-rich melodic tracks; lower (~1.5–2) for rhythm-only tracks where text is less reliable.
- **Longer generation:** MuseControlLite generates 47s vs MusicGen's 30s, covering more of the 60s target and improving metric alignment.
- **Retrieval improvement:** Use MuQ or Music2Latent embeddings instead of CLAP for retrieval, as they are trained specifically for music similarity rather than audio-text alignment.

---

## 5. How to Reproduce

### 5.1 Environment Setup

```bash
conda create -n deep_mir_hw2 python=3.10
conda activate deep_mir_hw2

# Core stack (must be pinned for AudioCraft compatibility)
pip install torch==2.1.0+cu121 torchaudio==2.1.0+cu121 torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

pip install numpy==1.26.4
pip install audiocraft==1.3.0
pip install transformers==4.45.2 accelerate
pip install laion-clap==1.1.7
pip install audiobox_aesthetics
pip install librosa soundfile scipy av==13.1.0
```

Download CLAP checkpoint:
```bash
cd /path/to/Deep_MIR_hw2
wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt
```

### 5.2 Run Pipeline

```bash
# Step 1: Retrieval
python step1_retrieval.py

# Step 2: Captioning (Qwen2-Audio)
python step2_captioning.py

# Step 3: Generation (MusicGen-Melody)
# Pin transformers back for AudioCraft before running
pip install transformers==4.35.2
python step3_generation.py

# Step 4: Evaluation
# Upgrade transformers again for CLAP/Audiobox
pip install transformers==4.45.2
python step4_eval.py
```

All outputs are saved under `Deep_MIR_hw2/outputs/`:
- `captions/captions.json` — ALM captions
- `generated/*.wav` — generated music (30s each)
- `eval/eval_results.json` — full evaluation results
- `eval/eval_results_table.txt` — formatted table
- `retrieval_results/retrieval_results.json` — top-3 retrieval per track