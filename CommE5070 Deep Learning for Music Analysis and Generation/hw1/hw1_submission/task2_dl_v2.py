"""
Task 2: Deep Learning - CRNN on Mel-spectrogram
Reference: Nasrullah & Tan, "Musical artist classification with convolutional 
recurrent neural networks," IJCNN 2019.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
import warnings
warnings.filterwarnings('ignore')

from dataset import ARTISTS, ARTIST2IDX, get_split

TRAIN_VAL_DIR = os.path.expanduser("~/CommE5070/hw1/hw1/artist20/train_val")
SR = 16000
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
SEGMENT_SECONDS = 10
SEGMENT_SAMPLES = SR * SEGMENT_SECONDS
NUM_CLASSES = 20
BATCH_SIZE = 32
EPOCHS = int(os.environ.get("EPOCHS", 80))
LR = 3e-4
SEGS_PER_TRACK = 3   # sample N segments per track per epoch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class AudioDataset(Dataset):
    def __init__(self, files, augment=False, segs_per_track=1):
        self.files = files
        self.augment = augment
        self.segs_per_track = segs_per_track
        # Expand: each track appears segs_per_track times
        self.samples = [(path, label) for path, label in files
                        for _ in range(segs_per_track)]

        self.mel = T.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=20, f_max=8000
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80)
        if augment:
            self.time_mask = T.TimeMasking(time_mask_param=40)
            self.freq_mask = T.FrequencyMasking(freq_mask_param=20)

    def load_segment(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != SR:
            waveform = torchaudio.functional.resample(waveform, sr, SR)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        total = waveform.shape[1]
        if total <= SEGMENT_SAMPLES:
            waveform = F.pad(waveform, (0, SEGMENT_SAMPLES - total))
        else:
            start = random.randint(0, total - SEGMENT_SAMPLES)
            waveform = waveform[:, start:start + SEGMENT_SAMPLES]
        return waveform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform = self.load_segment(path)
        mel = self.mel(waveform)
        mel = self.amplitude_to_db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        if self.augment:
            mel = self.time_mask(mel)
            mel = self.freq_mask(mel)
        return mel, label


# ---- CRNN Architecture (IJCNN 2019 inspired) ----
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=(2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(),
            nn.MaxPool2d(pool),
            nn.Dropout2d(0.1),
        )
    def forward(self, x):
        return self.block(x)


class CRNN(nn.Module):
    """
    CNN encodes mel-spectrogram patches, GRU models temporal sequence.
    Ref: Nasrullah & Tan, IJCNN 2019.
    """
    def __init__(self, num_classes=NUM_CLASSES, gru_hidden=256, gru_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(1,  64,  pool=(2, 2)),
            ConvBlock(64, 128, pool=(2, 2)),
            ConvBlock(128, 256, pool=(2, 2)),
            ConvBlock(256, 256, pool=(4, 1)),  # compress freq, keep time
        )
        # After CNN: freq=2, time=T', channels=256 -> reshape for GRU
        self.gru = nn.GRU(input_size=1024, hidden_size=gru_hidden,
                          num_layers=gru_layers, batch_first=True,
                          dropout=0.3, bidirectional=True)
        self.attn = nn.Linear(gru_hidden * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, 256),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.cnn(x)  # (B, 256, 2, T')
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)  # (B, T', 512)
        x, _ = self.gru(x)  # (B, T', 512)
        # Attention pooling
        attn_w = torch.softmax(self.attn(x), dim=1)  # (B, T', 1)
        x = (x * attn_w).sum(dim=1)  # (B, 512)
        return self.classifier(x)

    def get_embedding(self, x):
        x = self.cnn(x)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)
        x, _ = self.gru(x)
        attn_w = torch.softmax(self.attn(x), dim=1)
        return (x * attn_w).sum(dim=1)  # 512-dim


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for mel, labels in loader:
            mel, labels = mel.to(DEVICE), labels.to(DEVICE)
            logits = model(mel)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    acc1 = (all_preds == all_labels).mean()
    acc3 = top_k_accuracy_score(all_labels, all_probs, k=3)
    cm = confusion_matrix(all_labels, all_preds)
    return acc1, acc3, cm


def main():
    print(f"Device: {DEVICE}")
    random.seed(42); torch.manual_seed(42)

    train_files, val_files = get_split(TRAIN_VAL_DIR)
    print(f"Train tracks: {len(train_files)}, Val tracks: {len(val_files)}")

    train_ds = AudioDataset(train_files, augment=True, segs_per_track=SEGS_PER_TRACK)
    val_ds   = AudioDataset(val_files,   augment=False, segs_per_track=1)
    print(f"Train samples/epoch: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = CRNN().to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs("checkpoints", exist_ok=True)
    best_acc = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for mel, labels in train_loader:
            mel, labels = mel.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(mel)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            train_acc = correct / total
            acc1, acc3, _ = evaluate(model, val_loader)
            print(f"Ep {epoch:3d} | Loss {total_loss/total:.4f} | "
                  f"TrainAcc {train_acc:.4f} | ValTop1 {acc1:.4f} | ValTop3 {acc3:.4f}")
            if acc1 > best_acc:
                best_acc = acc1
                torch.save(model.state_dict(), "checkpoints/task2_best.pt")
                print(f"  -> Best saved ({best_acc:.4f})")

    print(f"\nBest Val Top-1: {best_acc:.4f}")
    model.load_state_dict(torch.load("checkpoints/task2_best.pt"))
    acc1, acc3, cm = evaluate(model, val_loader)
    print(f"Final Val Top-1: {acc1:.4f}, Top-3: {acc3:.4f}")
    np.save("checkpoints/confusion_matrix.npy", cm)


if __name__ == "__main__":
    main()
