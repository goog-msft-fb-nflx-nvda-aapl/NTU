"""
Task 2 Model C: Fully Generalized Non-Local Network for Singer Identification
Reference: "Positions, channels, and layers fully generalized non-local network 
for singer identification," AAAI 2021.

Non-local blocks capture long-range dependencies across time-frequency in mel-spectrograms.
Run on GPU 2: CUDA_VISIBLE_DEVICES=2 python task2_nonlocal.py
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
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
import warnings
warnings.filterwarnings('ignore')

from dataset import ARTISTS, ARTIST2IDX, get_split

TRAIN_VAL_DIR = os.path.expanduser("~/CommE5070/hw1/hw1/artist20/train_val")
SR = 16000
N_MELS = 128
HOP_LENGTH = 256
N_FFT = 2048
SEGMENT_SECONDS = 10
SEGMENT_SAMPLES = SR * SEGMENT_SECONDS
NUM_CLASSES = 20
BATCH_SIZE = 32
EPOCHS = 120
LR = 3e-4
SEGS_PER_TRACK = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class AudioDataset(Dataset):
    def __init__(self, files, augment=False, segs_per_track=1):
        self.files = files
        self.augment = augment
        self.samples = [(p, l) for p, l in files for _ in range(segs_per_track)]
        self.mel = T.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=20, f_max=8000
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80)
        if augment:
            self.time_mask = T.TimeMasking(time_mask_param=50)
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


# ---- Fully Generalized Non-Local Block (AAAI 2021) ----
class NonLocalBlock(nn.Module):
    """
    Generalized non-local block with position, channel, and layer-wise attention.
    Captures long-range time-frequency dependencies in the mel-spectrogram.
    """
    def __init__(self, in_channels, reduction=2):
        super().__init__()
        inter_ch = max(in_channels // reduction, 32)

        # Theta, phi, g projections
        self.theta = nn.Conv2d(in_channels, inter_ch, 1, bias=False)
        self.phi   = nn.Conv2d(in_channels, inter_ch, 1, bias=False)
        self.g     = nn.Conv2d(in_channels, inter_ch, 1, bias=False)
        self.out   = nn.Conv2d(inter_ch, in_channels, 1, bias=False)
        self.bn    = nn.BatchNorm2d(in_channels)

        # Channel attention (squeeze-excitation style)
        self.ch_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        theta = self.theta(x).view(B, -1, N).permute(0, 2, 1)  # (B, N, C')
        phi   = self.phi(x).view(B, -1, N)                      # (B, C', N)
        g     = self.g(x).view(B, -1, N).permute(0, 2, 1)       # (B, N, C')

        # Non-local attention map
        attn = torch.softmax(torch.bmm(theta, phi) / (theta.shape[-1] ** 0.5), dim=-1)  # (B, N, N)
        out = torch.bmm(attn, g).permute(0, 2, 1).view(B, -1, H, W)
        out = self.bn(self.out(out))

        # Channel attention
        ch_w = self.ch_attn(x).view(B, C, 1, 1)

        return x + out * ch_w


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_nonlocal=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ELU(),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch or stride != 1 else nn.Identity()
        self.nl = NonLocalBlock(out_ch) if use_nonlocal else nn.Identity()
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        out = self.conv(x)
        out = out + self.shortcut(x)
        out = self.nl(out)
        return self.drop(out)


class NonLocalSingerNet(nn.Module):
    """
    CNN backbone with non-local blocks inserted at multiple layers.
    AAAI 2021: non-local attention at positions, channels, and layers.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ELU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        # Non-local at layer 2, 3, 4
        self.layer1 = ConvBlock(32,  64,  stride=1, use_nonlocal=False)
        self.layer2 = ConvBlock(64,  128, stride=2, use_nonlocal=True)
        self.layer3 = ConvBlock(128, 256, stride=2, use_nonlocal=True)
        self.layer4 = ConvBlock(256, 512, stride=2, use_nonlocal=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.classifier(x)

    def get_embedding(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.pool(x).flatten(1)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for mel, labels in loader:
            mel, labels = mel.to(DEVICE), labels.to(DEVICE)
            logits = model(mel)
            probs = F.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    acc1 = (all_preds == all_labels).mean()
    acc3 = top_k_accuracy_score(all_labels, all_probs, k=3)
    return acc1, acc3, confusion_matrix(all_labels, all_preds)


def main():
    print(f"Device: {DEVICE}")
    random.seed(42); torch.manual_seed(42)

    train_files, val_files = get_split(TRAIN_VAL_DIR)
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    train_ds = AudioDataset(train_files, augment=True,  segs_per_track=SEGS_PER_TRACK)
    val_ds   = AudioDataset(val_files,   augment=False, segs_per_track=1)
    print(f"Train samples/epoch: {len(train_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = NonLocalSingerNet().to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )
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
            acc1, acc3, _ = evaluate(model, val_loader)
            print(f"Ep {epoch:3d} | Loss {total_loss/total:.4f} | "
                  f"TrainAcc {correct/total:.4f} | ValTop1 {acc1:.4f} | ValTop3 {acc3:.4f}")
            if acc1 > best_acc:
                best_acc = acc1
                torch.save(model.state_dict(), "checkpoints/nonlocal_best.pt")
                print(f"  -> Best saved ({best_acc:.4f})")

    print(f"\nBest Val Top-1: {best_acc:.4f}")
    model.load_state_dict(torch.load("checkpoints/nonlocal_best.pt"))
    acc1, acc3, cm = evaluate(model, val_loader)
    print(f"Final Val Top-1: {acc1:.4f}, Top-3: {acc3:.4f}")
    np.save("checkpoints/nonlocal_confusion_matrix.npy", cm)


if __name__ == "__main__":
    main()
