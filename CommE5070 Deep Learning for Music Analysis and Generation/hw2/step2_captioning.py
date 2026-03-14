"""
Step 2: Audio Captioning with Qwen-Audio-Chat
=============================================
Generates text descriptions for each track in target_music_list_60s.
Output: outputs/captions/captions.json

Usage:
    conda activate deep_mir_hw2
    python step2_captioning.py
"""

import os
import json
import glob
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/home/jtan/CommE5070/home/fundwotsai/Deep_MIR_hw2")
TARGET_DIR  = BASE_DIR / "target_music_list_60s"
OUTPUT_DIR  = BASE_DIR / "outputs" / "captions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "captions.json"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"   # best quality; swap to
                                              # "Qwen/Qwen2-Audio-7B" for
                                              # non-chat if VRAM is tight

PROMPT = (
    "Please describe this music in detail. "
    "Include: instruments, tempo, mood, genre, melody characteristics, "
    "rhythm pattern, and any notable musical features. "
    "Be specific and thorough — your description will be used as a "
    "text prompt for music generation."
)

TARGET_SR = 16_000   # Qwen-Audio expects 16 kHz mono

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_audio_16k(path: str) -> np.ndarray:
    """Load any audio file, resample to 16 kHz mono, return float32 array."""
    import librosa
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio.astype(np.float32)


def find_audio_files(directory: Path) -> list[Path]:
    exts = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]
    files = []
    for ext in exts:
        files.extend(directory.glob(ext))
    return sorted(files)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load existing captions if resuming
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            captions = json.load(f)
        print(f"Resuming — {len(captions)} captions already saved.")
    else:
        captions = {}

    # Load model
    print(f"Loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},          # spreads across available GPUs automatically
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.\n")

    audio_files = find_audio_files(TARGET_DIR)
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {TARGET_DIR}")

    print(f"Found {len(audio_files)} target tracks.\n")

    for audio_path in audio_files:
        name = audio_path.name

        if name in captions:
            print(f"  [SKIP] {name}")
            continue

        print(f"  Captioning: {name}")
        try:
            audio_array = load_audio_16k(str(audio_path))

            # Build conversation input
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": str(audio_path)},
                        {"type": "text",  "text": PROMPT},
                    ],
                }
            ]

            # Qwen2-Audio-Instruct uses apply_chat_template
            text = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )

            inputs = processor(
                text=text,
                audios=[audio_array],
                sampling_rate=TARGET_SR,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,          # greedy for reproducibility
                )

            # Decode only the newly generated tokens
            input_len = inputs["input_ids"].shape[1]
            new_ids = generated_ids[:, input_len:]
            caption = processor.batch_decode(
                new_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0].strip()

            captions[name] = caption
            print(f"    → {caption[:120]}{'...' if len(caption) > 120 else ''}\n")

        except Exception as e:
            print(f"    [ERROR] {name}: {e}\n")
            captions[name] = f"ERROR: {e}"

        # Save after every track so you don't lose progress
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(captions, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Captions saved to {OUTPUT_FILE}")
    print("\nAll captions:")
    print("─" * 60)
    for name, cap in captions.items():
        print(f"\n[{name}]\n{cap}\n")


if __name__ == "__main__":
    main()