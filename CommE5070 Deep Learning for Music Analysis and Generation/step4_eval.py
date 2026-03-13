"""
Step 4: Evaluation
==================
Computes for each track:
  - CLAP cosine similarity: (target vs text), (text vs generated), (generated vs target)
  - Melody Accuracy
  - Meta Audiobox Aesthetics: CE, CU, PC, PQ

Output: outputs/eval/eval_results.json  +  eval_results_table.txt

Usage:
    pip install audiobox_aesthetics laion-clap
    python step4_eval.py
"""

import json
import torch
import torchaudio
import numpy as np
import librosa
import scipy.signal as signal
from pathlib import Path
from torchaudio import transforms as T

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path("/home/jtan/CommE5070/home/fundwotsai/Deep_MIR_hw2")
TARGET_DIR    = BASE_DIR / "target_music_list_60s"
GEN_DIR       = BASE_DIR / "outputs" / "generated"
CAPTIONS_FILE = BASE_DIR / "outputs" / "captions" / "captions.json"
EVAL_DIR      = BASE_DIR / "outputs" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON   = EVAL_DIR / "eval_results.json"
OUTPUT_TXT    = EVAL_DIR / "eval_results_table.txt"

# ── Load captions ─────────────────────────────────────────────────────────────
with open(CAPTIONS_FILE, encoding="utf-8") as f:
    captions = json.load(f)

# ── MusicGen output SR ────────────────────────────────────────────────────────
GEN_SR = 32000
GEN_DURATION = 30  # seconds — must match step3

# ─────────────────────────────────────────────────────────────────────────────
# 1. MELODY ACCURACY  (from provided Melody_acc.py)
# ─────────────────────────────────────────────────────────────────────────────

def extract_melody_one_hot(audio_path, sr=44100, cutoff=261.2,
                            win_length=2048, hop_length=256):
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    nyquist = 0.5 * sr
    b, a = signal.butter(N=2, Wn=cutoff / nyquist, btype='high', analog=False)
    y_hp = signal.filtfilt(b, a, y)
    chroma = librosa.feature.chroma_stft(
        y=y_hp, sr=sr, n_fft=win_length,
        win_length=win_length, hop_length=hop_length
    )
    pitch_class_idx = np.argmax(chroma, axis=0)
    one_hot = np.zeros_like(chroma)
    one_hot[pitch_class_idx, np.arange(chroma.shape[1])] = 1.0
    return one_hot


def melody_accuracy(target_path, gen_path):
    gt  = extract_melody_one_hot(target_path)
    gen = extract_melody_one_hot(gen_path)
    min_len = min(gt.shape[1], gen.shape[1])
    matches = ((gen[:, :min_len] == gt[:, :min_len]) &
               (gen[:, :min_len] == 1)).sum()
    return float(matches / min_len)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLAP SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

def load_clap():
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(ckpt="/home/jtan/CommE5070/home/fundwotsai/Deep_MIR_hw2/music_speech_audioset_epoch_15_esc_89.98.pt", model_id=3)
    model.eval()
    return model


def clap_audio_embed(model, path):
    import librosa
    audio, _ = librosa.load(str(path), sr=48000, mono=True, duration=10.0)
    audio = torch.tensor(audio).unsqueeze(0).float()  # [1, samples]
    embed = model.get_audio_embedding_from_data(x=audio, use_tensor=True)
    return embed  # [1, D]


def clap_text_embed(model, text):
    embed = model.get_text_embedding([text], use_tensor=True)
    return embed  # [1, D]


def cosine_sim(a, b):
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return float((a * b).sum().item())


# ─────────────────────────────────────────────────────────────────────────────
# 3. META AUDIOBOX AESTHETICS
# ─────────────────────────────────────────────────────────────────────────────

def load_audiobox():
    from audiobox_aesthetics.infer import initialize_predictor
    predictor = initialize_predictor()   # auto-downloads checkpoint
    return predictor


def audiobox_score(predictor, path):
    result = predictor.forward([{"path": str(path)}])
    # returns list of dicts with CE, CU, PC, PQ
    return result[0]


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRIM TARGET to GEN_DURATION
# ─────────────────────────────────────────────────────────────────────────────

def trim_audio(src_path: Path, duration: float, out_dir: Path) -> Path:
    """Trim target audio to `duration` seconds, save to out_dir, return path."""
    import soundfile as sf
    out_path = out_dir / ("trimmed_" + src_path.stem + ".wav")
    if out_path.exists():
        return out_path
    audio, sr = librosa.load(str(src_path), sr=None, mono=False, duration=duration)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]  # [1, T]
    sf.write(str(out_path), audio.T, sr)
    return out_path


TRIM_DIR = EVAL_DIR / "trimmed_targets"
TRIM_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading CLAP ...")
    clap = load_clap()
    print("Loading Audiobox Aesthetics ...")
    audiobox = load_audiobox()
    print("Models loaded.\n")

    audio_files = sorted(TARGET_DIR.glob("*"))
    audio_files = [f for f in audio_files if f.suffix in {".wav", ".mp3", ".flac"}]

    results = {}

    for target_path in audio_files:
        name  = target_path.name
        stem  = target_path.stem
        caption = captions.get(name, "")

        # Find matching generated file
        gen_path = GEN_DIR / f"{stem}.wav"
        if not gen_path.exists():
            print(f"[SKIP - no generated file] {name}")
            continue

        print(f"Evaluating: {name}")

        # Trim target to same length as generated
        trimmed_target = trim_audio(target_path, GEN_DURATION, TRIM_DIR)

        # ── CLAP ──────────────────────────────────────────────────────────────
        print("  CLAP embeddings ...")
        e_target = clap_audio_embed(clap, trimmed_target)
        e_gen    = clap_audio_embed(clap, gen_path)
        e_text   = clap_text_embed(clap, caption)

        clap_target_text = cosine_sim(e_target, e_text)
        clap_text_gen    = cosine_sim(e_text, e_gen)
        clap_target_gen  = cosine_sim(e_target, e_gen)

        # ── Melody Accuracy ───────────────────────────────────────────────────
        print("  Melody accuracy ...")
        mel_acc = melody_accuracy(trimmed_target, gen_path)

        # ── Audiobox — target ─────────────────────────────────────────────────
        print("  Audiobox (target) ...")
        aes_target = audiobox_score(audiobox, trimmed_target)

        # ── Audiobox — generated ──────────────────────────────────────────────
        print("  Audiobox (generated) ...")
        aes_gen = audiobox_score(audiobox, gen_path)

        results[name] = {
            "clap": {
                "target_vs_text":      round(clap_target_text, 4),
                "text_vs_generated":   round(clap_text_gen,    4),
                "generated_vs_target": round(clap_target_gen,  4),
            },
            "melody_accuracy": round(mel_acc, 4),
            "audiobox_target": {k: round(v, 4) for k, v in aes_target.items()},
            "audiobox_generated": {k: round(v, 4) for k, v in aes_gen.items()},
        }

        print(f"  CLAP  tgt↔txt={clap_target_text:.3f}  txt↔gen={clap_text_gen:.3f}  tgt↔gen={clap_target_gen:.3f}")
        print(f"  Melody Acc = {mel_acc:.4f}")
        print(f"  Audiobox target    : {aes_target}")
        print(f"  Audiobox generated : {aes_gen}\n")

        # Save incrementally
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # ── Pretty table ──────────────────────────────────────────────────────────
    lines = []
    header = (f"{'Track':<55} {'tgt↔txt':>8} {'txt↔gen':>8} {'tgt↔gen':>8} "
              f"{'MelAcc':>7} "
              f"{'T-CE':>6} {'T-CU':>6} {'T-PC':>6} {'T-PQ':>6} "
              f"{'G-CE':>6} {'G-CU':>6} {'G-PC':>6} {'G-PQ':>6}")
    lines.append(header)
    lines.append("─" * len(header))
    for name, r in results.items():
        c  = r["clap"]
        at = r["audiobox_target"]
        ag = r["audiobox_generated"]
        lines.append(
            f"{name[:54]:<55} "
            f"{c['target_vs_text']:>8.3f} {c['text_vs_generated']:>8.3f} {c['generated_vs_target']:>8.3f} "
            f"{r['melody_accuracy']:>7.4f} "
            f"{at.get('CE',0):>6.2f} {at.get('CU',0):>6.2f} {at.get('PC',0):>6.2f} {at.get('PQ',0):>6.2f} "
            f"{ag.get('CE',0):>6.2f} {ag.get('CU',0):>6.2f} {ag.get('PC',0):>6.2f} {ag.get('PQ',0):>6.2f}"
        )
    table = "\n".join(lines)
    print("\n" + table)
    with open(OUTPUT_TXT, "w") as f:
        f.write(table + "\n")

    print(f"\nSaved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()