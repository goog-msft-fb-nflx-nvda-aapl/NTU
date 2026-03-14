# """Inference script - outputs top3 predictions as JSON"""
# import os
# import sys
# import json
# import random
# import argparse
# import numpy as np
# import torch
# import torch.nn.functional as F
# import torchaudio
# import torchaudio.transforms as T
# from pathlib import Path

# from dataset import ARTISTS
# from task2_dl import SingerCNN, SR, N_MELS, HOP_LENGTH, N_FFT, SEGMENT_SAMPLES

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# NUM_SEGMENTS = 5  # average over multiple segments for better accuracy


# def load_model(checkpoint_path="checkpoints/task2_best.pt"):
#     model = SingerCNN().to(DEVICE)
#     model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
#     model.eval()
#     return model


# def predict_file(model, path, mel_transform, amp_to_db):
#     waveform, sr = torchaudio.load(path)
#     if sr != SR:
#         waveform = torchaudio.functional.resample(waveform, sr, SR)
#     if waveform.shape[0] > 1:
#         waveform = waveform.mean(dim=0, keepdim=True)

#     total = waveform.shape[1]
#     probs_list = []

#     for _ in range(NUM_SEGMENTS):
#         if total <= SEGMENT_SAMPLES:
#             seg = F.pad(waveform, (0, SEGMENT_SAMPLES - total))
#         else:
#             start = random.randint(0, total - SEGMENT_SAMPLES)
#             seg = waveform[:, start:start + SEGMENT_SAMPLES]

#         mel = mel_transform(seg)
#         mel = amp_to_db(mel)
#         mel = (mel - mel.mean()) / (mel.std() + 1e-6)
#         mel = mel.unsqueeze(0).to(DEVICE)

#         with torch.no_grad():
#             logits = model(mel)
#             probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
#         probs_list.append(probs)

#     avg_probs = np.mean(probs_list, axis=0)
#     top3 = avg_probs.argsort()[::-1][:3].tolist()
#     return [ARTISTS[i] for i in top3]


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--test_path', type=str, required=True,
#                         help='Path to test directory with mp3 files')
#     parser.add_argument('--checkpoint', type=str, default='checkpoints/task2_best.pt')
#     parser.add_argument('--output', type=str, default='student_ID.json')
#     args = parser.parse_args()

#     random.seed(42)
#     model = load_model(args.checkpoint)
#     mel_transform = T.MelSpectrogram(
#         sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
#         n_mels=N_MELS, f_min=20, f_max=8000
#     )
#     amp_to_db = T.AmplitudeToDB(top_db=80)

#     test_files = sorted(Path(args.test_path).glob("*.mp3"))
#     print(f"Found {len(test_files)} test files")

#     results = {}
#     for i, path in enumerate(test_files):
#         file_id = path.stem  # e.g. "001"
#         top3 = predict_file(model, str(path), mel_transform, amp_to_db)
#         results[file_id] = top3
#         if (i + 1) % 20 == 0:
#             print(f"  {i+1}/{len(test_files)}")

#     with open(args.output, 'w') as f:
#         json.dump(results, f, indent=2)
#     print(f"\nSaved predictions to {args.output}")


# if __name__ == "__main__":
#     main()
"""Inference script - CRNN + TTA, outputs top3 predictions as JSON"""
import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from pathlib import Path

from dataset import ARTISTS
from task2_dl_v2 import CRNN, SR, N_MELS, HOP_LENGTH, N_FFT, SEGMENT_SAMPLES

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_SEGMENTS = 10


def load_model(checkpoint_path="checkpoints/task2_best.pt"):
    model = CRNN().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


def predict_file(model, path, mel_transform, amp_to_db):
    waveform, sr = torchaudio.load(path)
    if sr != SR:
        waveform = torchaudio.functional.resample(waveform, sr, SR)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    total = waveform.shape[1]
    probs_list = []

    for _ in range(NUM_SEGMENTS):
        if total <= SEGMENT_SAMPLES:
            seg = F.pad(waveform, (0, SEGMENT_SAMPLES - total))
        else:
            start = random.randint(0, total - SEGMENT_SAMPLES)
            seg = waveform[:, start:start + SEGMENT_SAMPLES]

        mel = mel_transform(seg)
        mel = amp_to_db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        mel = mel.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(mel)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        probs_list.append(probs)

    avg_probs = np.mean(probs_list, axis=0)
    top3_idx = avg_probs.argsort()[::-1][:3].tolist()
    return [ARTISTS[i] for i in top3_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/task2_best.pt')
    parser.add_argument('--output', type=str, default='student_ID.json')
    args = parser.parse_args()

    random.seed(42)
    model = load_model(args.checkpoint)
    mel_transform = T.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, f_min=20, f_max=8000
    )
    amp_to_db = T.AmplitudeToDB(top_db=80)

    test_files = sorted(Path(args.test_path).glob("*.mp3"))
    print(f"Found {len(test_files)} test files")

    results = {}
    for i, path in enumerate(test_files):
        file_id = path.stem
        top3 = predict_file(model, str(path), mel_transform, amp_to_db)
        results[file_id] = top3
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(test_files)}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved predictions to {args.output}")


if __name__ == "__main__":
    main()