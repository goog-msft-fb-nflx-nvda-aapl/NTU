import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SegDataset, label_to_color
from deeplabv3plus import DeepLabV3Plus
from PIL import Image

SAVE_IDS = ['0013', '0065', '0104']

def compute_miou(preds, labels, n_classes=7):
    ious = []
    for c in range(1, n_classes):
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
        out = model(imgs.to(device)).argmax(1).cpu().numpy()
        all_preds.append(out)
        all_labels.append(labels.numpy())
    return compute_miou(np.concatenate(all_preds), np.concatenate(all_labels))

@torch.no_grad()
def save_vis(model, val_dir, out_dir, epoch, device):
    from dataset import SegDataset, label_to_color
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    for fid in SAVE_IDS:
        img_path = os.path.join(val_dir, f'{fid}_sat.jpg')
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert('RGB')
        inp = tf(img).unsqueeze(0).to(device)
        pred = model(inp).argmax(1).squeeze(0).cpu().numpy()
        color_mask = label_to_color(pred)
        color_mask.save(os.path.join(out_dir, f'{fid}_epoch{epoch:03d}.png'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str, required=True)
    parser.add_argument('--val_dir',    type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='deeplab_output')
    parser.add_argument('--epochs',     type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--gpu',        type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    vis_dir = os.path.join(args.output_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    train_ds    = SegDataset(args.data_dir, augment=True)
    val_ds      = SegDataset(args.val_dir,  augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DeepLabV3Plus(n_classes=7).to(device)
    
    
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")
    optimizer = torch.optim.AdamW([
        {'params': [p for n,p in model.named_parameters() if 'layer' in n], 'lr': args.lr * 0.1},
        {'params': [p for n,p in model.named_parameters() if 'layer' not in n], 'lr': args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    mid_epoch = args.epochs // 2
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

        # Save vis at epoch 1, mid, last
        if epoch in (1, mid_epoch, args.epochs):
            save_vis(model, args.val_dir, vis_dir, epoch, device)
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'deeplab_epoch{epoch:03d}.pth'))

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_deeplab.pth'))

    print(f"\nBest mIoU: {best_miou:.4f}")

if __name__ == '__main__':
    main()