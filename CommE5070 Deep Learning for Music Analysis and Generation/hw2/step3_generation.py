"""
Step 3: Controllable Text-to-Music Generation with MusicGen-Melody
===================================================================
- Text condition: captions from step2
- Melody condition: chromagram extracted from target music (MIR tool, rule-compliant)
- Output: outputs/generated/<trackname>.wav

Usage:
    conda activate deep_mir_hw2
    python step3_generation.py
"""

import os
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path("/home/jtan/CommE5070/home/fundwotsai/Deep_MIR_hw2")
TARGET_DIR    = BASE_DIR / "target_music_list_60s"
CAPTIONS_FILE = BASE_DIR / "outputs" / "captions" / "captions.json"
OUTPUT_DIR    = BASE_DIR / "outputs" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Generation Config ─────────────────────────────────────────────────────────
MODEL_NAME     = "facebook/musicgen-melody"
DURATION       = 30        # seconds — MusicGen-Melody max is ~30s reliably
CFG_COEF       = 3.0       # classifier-free guidance scale (Strong condition)
TOP_K          = 250
TOP_P          = 0.0
TEMPERATURE    = 1.0
SAMPLE_RATE    = 32000     # MusicGen output sample rate

# ── Load captions ─────────────────────────────────────────────────────────────
with open(CAPTIONS_FILE, encoding="utf-8") as f:
    captions = json.load(f)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} ...")
model = MusicGen.get_pretrained(MODEL_NAME, device="cuda")
model.set_generation_params(
    duration=DURATION,
    cfg_coef=CFG_COEF,
    top_k=TOP_K,
    top_p=TOP_P,
    temperature=TEMPERATURE,
)
print("Model loaded.\n")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_audio_for_melody(path: str, target_sr: int = 32000):
    """Load audio and resample to model SR for melody conditioning."""
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    # MusicGen expects mono or stereo [C, T], batch dim added below
    if wav.shape[0] > 2:
        wav = wav[:2]
    return wav  # [C, T]


def trim_to_duration(wav: torch.Tensor, sr: int, duration: float) -> torch.Tensor:
    """Trim waveform to `duration` seconds."""
    max_samples = int(sr * duration)
    return wav[:, :max_samples]


# ── Generate ──────────────────────────────────────────────────────────────────
audio_files = sorted(TARGET_DIR.glob("*"))
audio_files = [f for f in audio_files if f.suffix in {".wav", ".mp3", ".flac"}]

for audio_path in audio_files:
    name = audio_path.name
    stem = audio_path.stem
    out_path = OUTPUT_DIR / f"{stem}.wav"

    if out_path.exists():
        print(f"[SKIP] {name}")
        continue

    caption = captions.get(name, "")
    if not caption or caption.startswith("ERROR"):
        print(f"[SKIP - no caption] {name}")
        continue

    print(f"Generating: {name}")
    print(f"  Prompt: {caption[:100]}...")

    try:
        melody_wav = load_audio_for_melody(str(audio_path), target_sr=SAMPLE_RATE)
        melody_wav = trim_to_duration(melody_wav, SAMPLE_RATE, DURATION)
        # Add batch dim: [1, C, T]
        melody_wav = melody_wav.unsqueeze(0).to("cuda")

        with torch.no_grad():
            wav = model.generate_with_chroma(
                descriptions=[caption],
                melody_wavs=melody_wav,
                melody_sample_rate=SAMPLE_RATE,
                progress=True,
            )

        # wav: [B, C, T] — save first item
        audio_write(
            str(OUTPUT_DIR / stem),
            wav[0].cpu(),
            model.sample_rate,
            strategy="loudness",
            loudness_compressor=True,
        )
        print(f"  Saved → {out_path}\n")

    except Exception as e:
        print(f"  [ERROR] {name}: {e}\n")

print("Done. All generated audio saved to:", OUTPUT_DIR)