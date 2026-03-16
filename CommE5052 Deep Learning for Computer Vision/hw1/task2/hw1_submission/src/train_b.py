import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm

# ─── Dataset ────────────────────────────────────────────────────────────────

def mask_to_label(mask_np):
    mask = (mask_np >= 128).astype(int)
    code = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    label = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)
    label[code == 3] = 0
    label[code == 6] = 1
    label[code == 5] = 2
    label[code == 2] = 3
    label[code == 1] = 4
    label[code == 7] = 5
    label[code == 0] = 6
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
        img  = Image.open(os.path.join(self.data_dir, name + '_sat.jpg')).convert('RGB')
        mask = Image.open(os.path.join(self.data_dir, name + '_mask.png')).convert('RGB')

        if self.augment:
            if torch.rand(1) > 0.5:
                img  = transforms.functional.hflip(img)
                mask = transforms.functional.hflip(mask)
            if torch.rand(1) > 0.5:
                img  = transforms.functional.vflip(img)
                mask = transforms.functional.vflip(mask)
            if torch.rand(1) > 0.5:
                angle = float(torch.randint(-30, 30, (1,)))
                img  = transforms.functional.rotate(img,  angle)
                mask = transforms.functional.rotate(mask, angle)

        img_t = self.img_transform(img)
        label = mask_to_label(np.array(mask))
        return img_t, torch.from_numpy(label)

# ─── Model: ResNet50 encoder + UNet-style decoder ───────────────────────────

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class DecodeBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch // 2 + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResUNet(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        backbone = models.resnet50(pretrained=True)

        # encoder stages
        self.stage0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2, 64
        self.pool   = backbone.maxpool                                              # /4
        self.stage1 = backbone.layer1   # /4,  256
        self.stage2 = backbone.layer2   # /8,  512
        self.stage3 = backbone.layer3   # /16, 1024
        self.stage4 = backbone.layer4   # /32, 2048

        # bridge
        self.bridge = nn.Sequential(ConvBnRelu(2048, 1024), ConvBnRelu(1024, 1024))

        # decoder
        self.dec4 = DecodeBlock(1024, 1024, 512)
        self.dec3 = DecodeBlock(512,  512,  256)
        self.dec2 = DecodeBlock(256,  256,  128)
        self.dec1 = DecodeBlock(128,  64,   64)
        self.dec0 = DecodeBlock(64,   0,    32)   # no skip at /1 level

        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        s0 = self.stage0(x)          # /2
        p  = self.pool(s0)
        s1 = self.stage1(p)          # /4
        s2 = self.stage2(s1)         # /8
        s3 = self.stage3(s2)         # /16
        s4 = self.stage4(s3)         # /32

        b  = self.bridge(s4)

        d  = self.dec4(b,  s3)
        d  = self.dec3(d,  s2)
        d  = self.dec2(d,  s1)
        d  = self.dec1(d,  s0)
        d  = self.dec0(d,  None)

        return self.head(d)

# ─── Losses ─────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=6):
        super().__init__()
        self.ignore = ignore_index

    def forward(self, logits, targets):
        n_cls = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        mask  = (targets != self.ignore)
        loss  = 0.0
        count = 0
        for c in range(n_cls):
            if c == self.ignore:
                continue
            p = probs[:, c][mask]
            t = (targets[mask] == c).float()
            inter = (p * t).sum()
            loss += 1 - (2 * inter + 1) / (p.sum() + t.sum() + 1)
            count += 1
        return loss / count

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(ignore_index=6)
        self.dice = DiceLoss(ignore_index=6)

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)

# ─── mIoU ───────────────────────────────────────────────────────────────────

def mean_iou(pred, true, n_classes=6):
    ious = []
    for c in range(n_classes):
        tp = ((pred == c) & (true == c)).sum()
        fp = ((pred == c) & (true != c)).sum()
        fn = ((pred != c) & (true == c)).sum()
        d  = tp + fp + fn
        ious.append(tp / d if d > 0 else 0.0)
    return float(np.mean(ious))

# ─── Train ───────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_set    = SegDataset(args.train_dir, augment=True)
    val_set      = SegDataset(args.val_dir,   augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model     = ResUNet(n_classes=7).to(device)
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f'E{epoch:03d} train'):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f'E{epoch:03d} val  '):
                out  = model(imgs.to(device))
                pred = out.argmax(dim=1).cpu().numpy()
                all_pred.append(pred)
                all_true.append(labels.numpy())
        miou = mean_iou(np.concatenate(all_pred), np.concatenate(all_true))
        print(f'Epoch {epoch:3d} | loss={total_loss/len(train_loader):.4f} | val mIoU={miou:.4f}')

        if epoch == 1 or epoch == args.epochs // 2 or epoch == args.epochs:
            torch.save(model.state_dict(),
                       os.path.join(args.ckpt_dir, f'resunet_epoch{epoch:03d}.pth'))

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(),
                       os.path.join(args.ckpt_dir, 'resunet_best.pth'))
            print(f'  -> Best saved (mIoU={best_miou:.4f})')

    print(f'\nDone. Best val mIoU: {best_miou:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir',  default='../data_2025/p2_data/train')
    parser.add_argument('--val_dir',    default='../data_2025/p2_data/validation')
    parser.add_argument('--ckpt_dir',   default='../checkpoints')
    parser.add_argument('--epochs',     type=int,   default=60)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--lr',         type=float, default=5e-4)
    args = parser.parse_args()
    train(args)