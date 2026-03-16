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
        self.augment  = augment
        ids = set()
        for f in os.listdir(data_dir):
            if f.endswith('_sat.jpg'):
                ids.add(f.replace('_sat.jpg', ''))
        self.ids = sorted(ids)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img  = Image.open(os.path.join(self.data_dir, name+'_sat.jpg')).convert('RGB')
        mask = Image.open(os.path.join(self.data_dir, name+'_mask.png')).convert('RGB')

        if self.augment:
            if torch.rand(1) > 0.5:
                img  = transforms.functional.hflip(img)
                mask = transforms.functional.hflip(mask)
            if torch.rand(1) > 0.5:
                img  = transforms.functional.vflip(img)
                mask = transforms.functional.vflip(mask)
            if torch.rand(1) > 0.5:
                angle = float(torch.randint(-30, 30, (1,)))
                img   = transforms.functional.rotate(img,  angle)
                mask  = transforms.functional.rotate(mask, angle)
            # color jitter on image only
            img = transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)(img)

        return self.norm(img), torch.from_numpy(mask_to_label(np.array(mask)))

# ─── ASPP ───────────────────────────────────────────────────────────────────

class ASPPConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

class ASPPPool(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x):
        size = x.shape[-2:]
        for m in self: x = m(x)
        return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
            ASPPConv(in_ch, out_ch, 6),
            ASPPConv(in_ch, out_ch, 12),
            ASPPConv(in_ch, out_ch, 18),
            ASPPPool(in_ch, out_ch),
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout(0.5))

    def forward(self, x):
        return self.proj(torch.cat([c(x) for c in self.convs], dim=1))

# ─── DeepLabV3+ ResNet101 ────────────────────────────────────────────────────

class DeepLabV3Plus(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        backbone = models.resnet101(pretrained=True)

        # stage 1-3 → low-level features at /4
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # /4,  256
        self.layer2 = backbone.layer2   # /8,  512
        # dilated stages
        backbone.layer3[0].conv2.stride      = (1,1)
        backbone.layer3[0].downsample[0].stride = (1,1)
        for blk in backbone.layer3: 
            for m in blk.modules():
                if isinstance(m, nn.Conv2d) and m.kernel_size==(3,3):
                    m.dilation=(2,2); m.padding=(2,2)
        self.layer3 = backbone.layer3   # /8, 1024

        backbone.layer4[0].conv2.stride      = (1,1)
        backbone.layer4[0].downsample[0].stride = (1,1)
        for blk in backbone.layer4:
            for m in blk.modules():
                if isinstance(m, nn.Conv2d) and m.kernel_size==(3,3):
                    m.dilation=(4,4); m.padding=(4,4)
        self.layer4 = backbone.layer4   # /8, 2048

        self.aspp = ASPP(2048, 256)

        # low-level conv
        self.low_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True))

        # decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256+48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, n_classes, 1),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x    = self.layer0(x)
        low  = self.layer1(x)          # /4, 256
        x    = self.layer2(low)
        x    = self.layer3(x)
        x    = self.layer4(x)

        x    = self.aspp(x)
        x    = nn.functional.interpolate(x, size=low.shape[-2:],
                                         mode='bilinear', align_corners=False)
        low  = self.low_conv(low)
        x    = self.decoder(torch.cat([x, low], dim=1))
        return nn.functional.interpolate(x, size=size,
                                         mode='bilinear', align_corners=False)

# ─── Losses ─────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=6):
        super().__init__()
        self.gamma  = gamma
        self.ignore = ignore_index

    def forward(self, logits, targets):
        ce   = nn.functional.cross_entropy(logits, targets, ignore_index=self.ignore, reduction='none')
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss[targets != self.ignore].mean()

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=6):
        super().__init__()
        self.ignore = ignore_index

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        mask  = (targets != self.ignore)
        loss, count = 0.0, 0
        for c in range(logits.shape[1]):
            if c == self.ignore: continue
            p = probs[:, c][mask]
            t = (targets[mask] == c).float()
            loss += 1 - (2*(p*t).sum()+1) / (p.sum()+t.sum()+1)
            count += 1
        return loss / count

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss(gamma=2.0)
        self.dice  = DiceLoss()
    def forward(self, logits, targets):
        return self.focal(logits, targets) + self.dice(logits, targets)

# ─── mIoU ───────────────────────────────────────────────────────────────────

def mean_iou(pred, true, n=6):
    ious = []
    for c in range(n):
        tp = ((pred==c)&(true==c)).sum()
        fp = ((pred==c)&(true!=c)).sum()
        fn = ((pred!=c)&(true==c)).sum()
        d  = tp+fp+fn
        ious.append(tp/d if d>0 else 0.0)
    return float(np.mean(ious))

# ─── TTA inference ──────────────────────────────────────────────────────────

def tta_predict(model, img_t, device):
    """Average predictions over 4 flips."""
    preds = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            x = img_t.clone()
            if hflip: x = torch.flip(x, [-1])
            if vflip: x = torch.flip(x, [-2])
            logit = model(x.unsqueeze(0).to(device))[0]
            if hflip: logit = torch.flip(logit, [-1])
            if vflip: logit = torch.flip(logit, [-2])
            preds.append(torch.softmax(logit, dim=0))
    return torch.stack(preds).mean(0).argmax(0).cpu().numpy()

# ─── Train ───────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_set    = SegDataset(args.train_dir, augment=True)
    val_set      = SegDataset(args.val_dir,   augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = DeepLabV3Plus(n_classes=7).to(device)

    # differential LR: backbone lower, decoder higher
    backbone_params = list(model.layer0.parameters()) + list(model.layer1.parameters()) + \
                      list(model.layer2.parameters()) + list(model.layer3.parameters()) + \
                      list(model.layer4.parameters())
    decoder_params  = list(model.aspp.parameters()) + list(model.low_conv.parameters()) + \
                      list(model.decoder.parameters())
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': decoder_params,  'lr': args.lr},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = CombinedLoss()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f'E{epoch:03d} train'):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # ── val (with TTA) ──
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f'E{epoch:03d} val  '):
                pred = tta_predict(model, imgs[0], device)
                all_pred.append(pred)
                all_true.append(labels[0].numpy())
        miou = mean_iou(np.stack(all_pred), np.stack(all_true))
        print(f'Epoch {epoch:3d} | loss={total_loss/len(train_loader):.4f} | val mIoU={miou:.4f}')

        if epoch == 1 or epoch == args.epochs // 2 or epoch == args.epochs:
            torch.save(model.state_dict(),
                       os.path.join(args.ckpt_dir, f'deeplab101_epoch{epoch:03d}.pth'))

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(),
                       os.path.join(args.ckpt_dir, 'deeplab101_best.pth'))
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