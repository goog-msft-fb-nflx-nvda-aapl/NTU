import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ── Dataset ──────────────────────────────────────────────────────────────────
class OfficeHomeDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row['filename'])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(row['label'])

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(setting, dino_ckpt_path=None, freeze_backbone=False):
    model = models.resnet50(weights=None)
    
    if setting == 'A':
        pass  # random init, train everything
    elif setting in ('B', 'D'):
        # Load TA-provided DINO weights
        state = torch.load(dino_ckpt_path, map_location='cpu')
        # DINO saves as {'student': ..., 'teacher': ...} or direct state_dict
        if 'teacher' in state:
            sd = {k.replace('module.', '').replace('backbone.', ''): v
                  for k, v in state['teacher'].items()}
        elif 'student' in state:
            sd = {k.replace('module.', '').replace('backbone.', ''): v
                  for k, v in state['student'].items()}
        else:
            sd = state
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    elif setting in ('C', 'E'):
        # Load our SSL pre-trained weights
        state = torch.load(dino_ckpt_path, map_location='cpu')
        if 'teacher' in state:
            sd = {k.replace('module.', '').replace('backbone.', ''): v
                  for k, v in state['teacher'].items()}
        elif 'student' in state:
            sd = {k.replace('module.', '').replace('backbone.', ''): v
                  for k, v in state['student'].items()}
        else:
            sd = state
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # Replace FC with 65-class head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 65)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    return model

# ── Train / Eval ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, required=True, choices=['A','B','C','D','E'])
    parser.add_argument('--data_dir', type=str, default='/home/jtan/CommE5052/hw1/data_2025/p1_data/office')
    parser.add_argument('--dino_ckpt', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='/home/jtan/CommE5052/hw1/finetune_output')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}')

    freeze_backbone = args.setting in ('D', 'E')

    # Transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = OfficeHomeDataset(
        os.path.join(args.data_dir, 'train.csv'),
        os.path.join(args.data_dir, 'train'), train_tf)
    val_ds = OfficeHomeDataset(
        os.path.join(args.data_dir, 'val.csv'),
        os.path.join(args.data_dir, 'val'), val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = build_model(args.setting, args.dino_ckpt, freeze_backbone).to(device)

    # Only optimize trainable params
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, f'best_{args.setting}.pth'))

    print(f"\nBest Val Acc (Setting {args.setting}): {best_acc:.4f}")

if __name__ == '__main__':
    main()
