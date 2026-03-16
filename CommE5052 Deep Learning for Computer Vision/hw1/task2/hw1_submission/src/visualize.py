import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── DeepLabV3+ (same arch as train_c) ──────────────────────────────────────

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
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
            ASPPConv(in_ch, out_ch, 6), ASPPConv(in_ch, out_ch, 12), ASPPConv(in_ch, out_ch, 18),
            ASPPPool(in_ch, out_ch),
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch*5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout(0.5))
    def forward(self, x):
        return self.proj(torch.cat([c(x) for c in self.convs], dim=1))

class DeepLabV3Plus(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        bb = models.resnet101(pretrained=False)
        self.layer0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1; self.layer2 = bb.layer2
        bb.layer3[0].conv2.stride = (1,1); bb.layer3[0].downsample[0].stride = (1,1)
        for blk in bb.layer3:
            for m in blk.modules():
                if isinstance(m, nn.Conv2d) and m.kernel_size==(3,3):
                    m.dilation=(2,2); m.padding=(2,2)
        self.layer3 = bb.layer3
        bb.layer4[0].conv2.stride = (1,1); bb.layer4[0].downsample[0].stride = (1,1)
        for blk in bb.layer4:
            for m in blk.modules():
                if isinstance(m, nn.Conv2d) and m.kernel_size==(3,3):
                    m.dilation=(4,4); m.padding=(4,4)
        self.layer4 = bb.layer4
        self.aspp = ASPP(2048, 256)
        self.low_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.Conv2d(256+48, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Conv2d(256,   256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv2d(256, n_classes, 1),
        )
    def forward(self, x):
        size = x.shape[-2:]
        x = self.layer0(x); low = self.layer1(x)
        x = self.layer2(low); x = self.layer3(x); x = self.layer4(x)
        x = self.aspp(x)
        x = nn.functional.interpolate(x, size=low.shape[-2:], mode='bilinear', align_corners=False)
        x = self.decoder(torch.cat([x, self.low_conv(low)], dim=1))
        return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)

# ─── Utils ───────────────────────────────────────────────────────────────────

NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

CLASS_COLORS = np.array([
    [0,   255, 255],  # 0 Urban       - Cyan
    [255, 255,   0],  # 1 Agriculture - Yellow
    [255,   0, 255],  # 2 Rangeland   - Purple
    [0,   255,   0],  # 3 Forest      - Green
    [0,     0, 255],  # 4 Water       - Blue
    [255, 255, 255],  # 5 Barren      - White
    [0,     0,   0],  # 6 Unknown     - Black
], dtype=np.uint8)

CLASS_NAMES = ['Urban', 'Agriculture', 'Rangeland', 'Forest', 'Water', 'Barren', 'Unknown']

def predict(model, img_path, device):
    img   = Image.open(img_path).convert('RGB')
    img_t = NORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img_t).argmax(dim=1)[0].cpu().numpy()
    rgb = CLASS_COLORS[pred]
    return np.array(img), rgb

def mask_to_rgb(mask_path):
    mask_np = np.array(Image.open(mask_path).convert('RGB'))
    m    = (mask_np >= 128).astype(int)
    code = 4*m[:,:,0] + 2*m[:,:,1] + m[:,:,2]
    label = np.zeros(code.shape, dtype=np.int64)
    label[code==3]=0; label[code==6]=1; label[code==5]=2
    label[code==2]=3; label[code==1]=4; label[code==7]=5; label[code==0]=6
    return CLASS_COLORS[label]

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir',    default='../data_2025/p2_data/validation')
    parser.add_argument('--ckpt_early', default='../checkpoints/deeplab101_epoch001.pth')
    parser.add_argument('--ckpt_mid',   default='../checkpoints/deeplab101_epoch030.pth')
    parser.add_argument('--ckpt_final', default='../checkpoints/deeplab101_best.pth')
    parser.add_argument('--out_dir',    default='../viz_output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    stages = [
        ('Early  (Epoch 1)',  args.ckpt_early),
        ('Middle (Epoch 30)', args.ckpt_mid),
        ('Final  (Best)',     args.ckpt_final),
    ]

    target_ids = ['0018', '0065', '0109']

    for img_id in target_ids:
        img_path  = os.path.join(args.val_dir, f'{img_id}_sat.jpg')
        mask_path = os.path.join(args.val_dir, f'{img_id}_mask.png')

        # cols: Input | GT | Early | Mid | Final
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(f'Image {img_id}', fontsize=14, fontweight='bold')

        img_np = np.array(Image.open(img_path).convert('RGB'))
        axes[0].imshow(img_np);              axes[0].set_title('Input');          axes[0].axis('off')
        axes[1].imshow(mask_to_rgb(mask_path)); axes[1].set_title('Ground Truth'); axes[1].axis('off')

        for col, (stage_name, ckpt_path) in enumerate(stages, start=2):
            model = DeepLabV3Plus(n_classes=7).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            _, pred_rgb = predict(model, img_path, device)
            axes[col].imshow(pred_rgb)
            axes[col].set_title(stage_name)
            axes[col].axis('off')
            del model

        # legend
        patches = [mpatches.Patch(color=np.array(CLASS_COLORS[i])/255., label=CLASS_NAMES[i])
                   for i in range(7)]
        fig.legend(handles=patches, loc='lower center', ncol=7, fontsize=9,
                   bbox_to_anchor=(0.5, -0.05))

        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f'viz_{img_id}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved {out_path}')

    print('Visualization done.')

if __name__ == '__main__':
    main()