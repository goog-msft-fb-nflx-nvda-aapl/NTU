"""Task 1: Traditional ML - MFCC features + SVM classifier"""
import os
import numpy as np
import librosa
import pickle
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
import warnings
warnings.filterwarnings('ignore')

from dataset import ARTISTS, ARTIST2IDX, get_split

TRAIN_VAL_DIR = os.path.expanduser("~/CommE5070/hw1/hw1/artist20/train_val")
SEGMENT_DURATION = 5  # seconds
SR = 16000
N_MFCC = 40
HOP_LENGTH = 512
N_MELS = 128


def extract_features(path, sr=SR, segment_duration=SEGMENT_DURATION):
    """Extract MFCC + delta + delta2 statistics from audio segments."""
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

    seg_len = sr * segment_duration
    # Take up to 3 segments (beginning, middle, end)
    segments = []
    total = len(y)
    if total < seg_len:
        y = np.pad(y, (0, seg_len - total))
        total = seg_len

    for start in [0, max(0, total//2 - seg_len//2), max(0, total - seg_len)]:
        seg = y[start:start + seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        segments.append(seg)

    all_feats = []
    for seg in segments:
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        chroma = librosa.feature.chroma_stft(y=seg, sr=sr, hop_length=HOP_LENGTH)
        spectral_contrast = librosa.feature.spectral_contrast(y=seg, sr=sr, hop_length=HOP_LENGTH)
        zcr = librosa.feature.zero_crossing_rate(seg, hop_length=HOP_LENGTH)

        feat = np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            delta.mean(axis=1), delta.std(axis=1),
            delta2.mean(axis=1), delta2.std(axis=1),
            chroma.mean(axis=1), chroma.std(axis=1),
            spectral_contrast.mean(axis=1), spectral_contrast.std(axis=1),
            zcr.mean(axis=1), zcr.std(axis=1),
        ])
        all_feats.append(feat)

    return np.mean(all_feats, axis=0)


def load_features(files, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cache from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    X, y = [], []
    for i, (path, label) in enumerate(files):
        if i % 50 == 0:
            print(f"  {i}/{len(files)}")
        feat = extract_features(path)
        if feat is not None:
            X.append(feat)
            y.append(label)

    X, y = np.array(X), np.array(y)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump((X, y), f)
    return X, y


def evaluate(model, scaler, X, y, name="Val"):
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)

    acc1 = (y_pred == y).mean()
    acc3 = top_k_accuracy_score(y, proba, k=3)
    cm = confusion_matrix(y, y_pred)

    print(f"\n{name} Top-1 Accuracy: {acc1:.4f}")
    print(f"{name} Top-3 Accuracy: {acc3:.4f}")
    return acc1, acc3, cm


def main():
    print("Loading train/val splits...")
    train_files, val_files = get_split(TRAIN_VAL_DIR)
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    os.makedirs("cache", exist_ok=True)
    print("Extracting train features...")
    X_train, y_train = load_features(train_files, cache_path="cache/train_features.pkl")
    print("Extracting val features...")
    X_val, y_val = load_features(val_files, cache_path="cache/val_features.pkl")

    print(f"\nTrain shape: {X_train.shape}, Val shape: {X_val.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("\nTraining SVM...")
    model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    evaluate(model, scaler, X_train, y_train, "Train")
    acc1, acc3, cm = evaluate(model, scaler, X_val, y_val, "Val")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/task1_svm.pkl", 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    print("\nModel saved to checkpoints/task1_svm.pkl")

    return acc1, acc3


if __name__ == "__main__":
    main()
