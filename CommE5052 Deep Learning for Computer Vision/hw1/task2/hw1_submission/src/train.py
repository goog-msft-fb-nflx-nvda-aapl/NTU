import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

# ─── Dataset ────────────────────────────────────────────────────────────────

def mask_to_label(mask_np):
    mask = (mask_np >= 128).astype(int)
    code = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    label = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)
    label[code == 3] = 0  # Urban
    label[code == 6] = 1  # Agriculture
    label[code == 5] = 2  # Rangeland
    label[code == 2] = 3  # Forest
    label[code == 1] = 4  # Water
    label[code == 7] = 5  # Barren
    label[code == 0] = 6  # Unknown
    return label

class SegDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.data_dir = data_dir
        self.augment = augment
        ids = set()
        for f in os.listdir(data_dir):
            if f.endswith('_sat.jpg'):
                ids.add(f.replace('_sat.jpg', ''))
        self.ids = sorted(ids)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img = Image.open(os.path.join(self.data_dir, name + '_sat.jpg')).convert('RGB')
        mask = Image.open(os.path.join(self.data_dir, name + '_mask.png')).convert('RGB')

        if self.augment and torch.rand(1) > 0.5:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)
        if self.augment and torch.rand(1) > 0.5:
            img = transforms.functional.vflip(img)
            mask = transforms.functional.vflip(mask)

        img_t = self.img_transform(img)
        label = mask_to_label(np.array(mask))
        return img_t, torch.from_numpy(label)

# ─── U-Net ───────────────────────────────────────────────────────────────────

def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, n_classes=7, drop_skip=None):
        super().__init__()
        self.drop_skip = drop_skip  # which skip to drop (0-3), None = standard UNet

        self.enc1 = double_conv(3, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = double_conv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = double_conv(128, 64)

        self.out_conv = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        skips = [e4, e3, e2, e1]

        d = self.up4(b)
        s = torch.zeros_like(skips[0]) if self.drop_skip == 0 else skips[0]
        d = self.dec4(torch.cat([d, s], dim=1))

        d = self.up3(d)
        s = torch.zeros_like(skips[1]) if self.drop_skip == 1 else skips[1]
        d = self.dec3(torch.cat([d, s], dim=1))

        d = self.up2(d)
        s = torch.zeros_like(skips[2]) if self.drop_skip == 2 else skips[2]
        d = self.dec2(torch.cat([d, s], dim=1))

        d = self.up1(d)
        s = torch.zeros_like(skips[3]) if self.drop_skip == 3 else skips[3]
        d = self.dec1(torch.cat([d, s], dim=1))

        return self.out_conv(d)

# ─── Train ───────────────────────────────────────────────────────────────────

def mean_iou(pred_labels, true_labels, n_classes=6):
    ious = []
    for c in range(n_classes):
        tp = ((pred_labels == c) & (true_labels == c)).sum()
        fp = ((pred_labels == c) & (true_labels != c)).sum()
        fn = ((pred_labels != c) & (true_labels == c)).sum()
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else 0.0)
    return np.mean(ious)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_set = SegDataset(args.train_dir, augment=True)
    val_set   = SegDataset(args.val_dir,   augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = UNet(n_classes=7).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=6)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} train'):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # ── val ──
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f'Epoch {epoch}/{args.epochs} val  '):
                imgs = imgs.to(device)
                out = model(imgs)
                pred = out.argmax(dim=1).cpu().numpy()
                all_pred.append(pred)
                all_true.append(labels.numpy())
        all_pred = np.concatenate(all_pred, axis=0)
        all_true = np.concatenate(all_true, axis=0)
        miou = mean_iou(all_pred, all_true)
        print(f'Epoch {epoch:3d} | loss={total_loss/len(train_loader):.4f} | val mIoU={miou:.4f}')

        # save every epoch for visualization (early/mid/final)
        if epoch == 1 or epoch == args.epochs // 2 or epoch == args.epochs:
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f'unet_epoch{epoch:03d}.pth'))

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'unet_best.pth'))
            print(f'  -> Best model saved (mIoU={best_miou:.4f})')

    print(f'\nTraining done. Best val mIoU: {best_miou:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='../data_2025/p2_data/train')
    parser.add_argument('--val_dir',   default='../data_2025/p2_data/validation')
    parser.add_argument('--ckpt_dir',  default='../checkpoints')
    parser.add_argument('--epochs',    type=int, default=50)
    parser.add_argument('--batch_size',type=int, default=8)
    parser.add_argument('--lr',        type=float, default=1e-3)
    args = parser.parse_args()
    train(args)