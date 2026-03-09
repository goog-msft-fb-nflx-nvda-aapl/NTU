import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SegDataset, label_to_color
from unet import UNet

def compute_miou(preds, labels, n_classes=7):
    ious = []
    for c in range(1, n_classes):  # skip class 0 (Unknown)
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        if tp + fp + fn == 0:
            continue
        ious.append(tp / (tp + fp + fn))
    return float(np.mean(ious)) if ious else 0.0

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        out = model(imgs).argmax(1).cpu().numpy()
        all_preds.append(out)
        all_labels.append(labels.numpy())
    return compute_miou(np.concatenate(all_preds), np.concatenate(all_labels))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  type=str, required=True)
    parser.add_argument('--val_dir',   type=str, required=True)
    parser.add_argument('--output_dir',type=str, default='unet_output')
    parser.add_argument('--epochs',    type=int, default=50)
    parser.add_argument('--batch_size',type=int, default=8)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--drop_skip', type=int, default=None,
                        help='Which skip connection to drop (1-4). None = standard UNet.')
    parser.add_argument('--gpu',       type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    train_ds = SegDataset(args.data_dir, augment=True)
    val_ds   = SegDataset(args.val_dir,  augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = UNet(n_classes=7, drop_skip=args.drop_skip).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_miou = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        miou = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f} | Val mIoU: {miou:.4f}")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_unet.pth'))

    print(f"\nBest mIoU: {best_miou:.4f}")

if __name__ == '__main__':
    main()