"""
Task 2 Model B: CNN with Squeeze-and-Excitation blocks on Mel-spectrogram
Inspired by: music classification and singer identification literature.
Run on GPU 1: CUDA_VISIBLE_DEVICES=1 python task2_se_cnn.py
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
BATCH_SIZE = 64
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


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class ConvSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SEBlock(out_ch)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch or stride != 1 else nn.Identity()
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        return F.elu(out + self.shortcut(x))


class SingerSENet(nn.Module):
    """SE-ResNet on mel-spectrogram for singer identification."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ELU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(ConvSEBlock(32,  64), ConvSEBlock(64, 64))
        self.layer2 = nn.Sequential(ConvSEBlock(64,  128, stride=2), ConvSEBlock(128, 128))
        self.layer3 = nn.Sequential(ConvSEBlock(128, 256, stride=2), ConvSEBlock(256, 256))
        self.layer4 = nn.Sequential(ConvSEBlock(256, 512, stride=2), ConvSEBlock(512, 512))
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

    model = SingerSENet().to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS,
        steps_per_epoch=len(train_loader), pct_start=0.1
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
            scheduler.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        if epoch % 5 == 0 or epoch == 1:
            acc1, acc3, _ = evaluate(model, val_loader)
            print(f"Ep {epoch:3d} | Loss {total_loss/total:.4f} | "
                  f"TrainAcc {correct/total:.4f} | ValTop1 {acc1:.4f} | ValTop3 {acc3:.4f}")
            if acc1 > best_acc:
                best_acc = acc1
                torch.save(model.state_dict(), "checkpoints/senet_best.pt")
                print(f"  -> Best saved ({best_acc:.4f})")

    print(f"\nBest Val Top-1: {best_acc:.4f}")
    model.load_state_dict(torch.load("checkpoints/senet_best.pt"))
    acc1, acc3, cm = evaluate(model, val_loader)
    print(f"Final Val Top-1: {acc1:.4f}, Top-3: {acc3:.4f}")
    np.save("checkpoints/senet_confusion_matrix.npy", cm)


if __name__ == "__main__":
    main()
