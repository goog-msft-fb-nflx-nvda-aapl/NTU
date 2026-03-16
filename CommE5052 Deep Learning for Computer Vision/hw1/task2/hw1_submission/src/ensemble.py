import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm

# ─── Label utils ─────────────────────────────────────────────────────────────

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

def label_to_mask(label):
    """Convert class label map back to RGB mask PNG."""
    color_map = {
        0: (0,   255, 255),  # Urban      - Cyan
        1: (255, 255,   0),  # Agriculture- Yellow
        2: (255,   0, 255),  # Rangeland  - Purple
        3: (0,   255,   0),  # Forest     - Green
        4: (0,     0, 255),  # Water      - Blue
        5: (255, 255, 255),  # Barren     - White
        6: (0,     0,   0),  # Unknown    - Black
    }
    h, w  = label.shape
    rgb   = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in color_map.items():
        rgb[label == cls] = color
    return rgb

# ─── Models (same as train_c / train_d) ─────────────────────────────────────

# -- DeepLabV3+ (train_c) --

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
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        bb.layer3[0].conv2.stride = (1,1); bb.layer3[0].downsample[0].stride = (1,1)
        for blk in bb.layer3:
            for m in blk.modules():
                if isinstance(m, nn.Conv2d) and m.kernel_size==(3,3): m.dilation=(2,2); m.padding=(2,2)
        self.layer3 = bb.layer3
        bb.layer4[0].conv2.stride = (1,1); bb.layer4[0].downsample[0].stride = (1,1)
        for blk in bb.layer4:
            for m in blk.modules():
                if isinstance(m, nn.Conv2d) and m.kernel_size==(3,3): m.dilation=(4,4); m.padding=(4,4)
        self.layer4 = bb.layer4
        self.aspp = ASPP(2048, 256)
        self.low_conv = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.Conv2d(256+48, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.1),
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

# -- PSPNet (train_d) --

class PPM(nn.Module):
    def __init__(self, in_ch, out_ch, pool_sizes=(1,2,3,6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(s), nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)) for s in pool_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch + out_ch*len(pool_sizes), out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout2d(0.1))
    def forward(self, x):
        size = x.shape[-2:]
        out = [x] + [nn.functional.interpolate(s(x), size=size, mode='bilinear', align_corners=False) for s in self.stages]
        return self.bottleneck(torch.cat(out, dim=1))

class PSPNet(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        bb = models.resnet101(pretrained=False)
        self.layer0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1; self.layer2 = bb.layer2
        bb.layer3[0].conv2.stride = (1,1); bb.layer3[0].downsample[0].stride = (1,1)
        for blk in bb.layer3:
            for m in blk.modules():
                if isinstance(m, nn.Conv2d) and m.kernel_size==(3,3): m.dilation=(2,2); m.padding=(2,2)
        self.layer3 = bb.layer3
        bb.layer4[0].conv2.stride = (1,1); bb.layer4[0].downsample[0].stride = (1,1)
        for blk in bb.layer4:
            for m in blk.modules():
                if isinstance(m, nn.Conv2d) and m.kernel_size==(3,3): m.dilation=(4,4); m.padding=(4,4)
        self.layer4 = bb.layer4
        self.ppm = PPM(2048, 512)
        self.aux_head = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1), nn.Conv2d(256, n_classes, 1))
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1), nn.Conv2d(256, n_classes, 1))
    def forward(self, x):
        size = x.shape[-2:]
        x = self.layer0(x); x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x); x = self.ppm(x)
        return nn.functional.interpolate(self.head(x), size=size, mode='bilinear', align_corners=False)

# ─── TTA + Ensemble ──────────────────────────────────────────────────────────

NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def tta_probs(model, img_t, device):
    preds = []
    for hf in [False, True]:
        for vf in [False, True]:
            x = img_t.clone()
            if hf: x = torch.flip(x, [-1])
            if vf: x = torch.flip(x, [-2])
            logit = model(x.unsqueeze(0).to(device))[0]
            if hf: logit = torch.flip(logit, [-1])
            if vf: logit = torch.flip(logit, [-2])
            preds.append(torch.softmax(logit, dim=0).cpu())
    return torch.stack(preds).mean(0)   # (C, H, W)

def mean_iou(pred, true, n=6):
    ious = []
    for c in range(n):
        tp = ((pred==c)&(true==c)).sum()
        fp = ((pred==c)&(true!=c)).sum()
        fn = ((pred!=c)&(true==c)).sum()
        d  = tp+fp+fn
        ious.append(tp/d if d>0 else 0.0)
    return float(np.mean(ious))

# ─── Main ─────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # load models
    model_c = DeepLabV3Plus(n_classes=7).to(device)
    model_c.load_state_dict(torch.load(args.ckpt_c, map_location=device))
    model_c.eval()
    print(f'Loaded DeepLabV3+: {args.ckpt_c}')

    model_d = PSPNet(n_classes=7).to(device)
    model_d.load_state_dict(torch.load(args.ckpt_d, map_location=device))
    model_d.eval()
    print(f'Loaded PSPNet:     {args.ckpt_d}')

    # gather images
    img_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('_sat.jpg')])

    os.makedirs(args.output_dir, exist_ok=True)

    all_pred, all_true = [], []
    has_gt = os.path.exists(os.path.join(args.input_dir, img_files[0].replace('_sat.jpg','_mask.png')))

    with torch.no_grad():
        for fname in tqdm(img_files):
            img = Image.open(os.path.join(args.input_dir, fname)).convert('RGB')
            img_t = NORM(img)

            # ensemble: average TTA probs from both models
            probs = (tta_probs(model_c, img_t, device) +
                     tta_probs(model_d, img_t, device)) / 2.0
            pred  = probs.argmax(0).numpy()

            # save mask
            out_name = fname.replace('_sat.jpg', '_mask.png')
            Image.fromarray(label_to_mask(pred)).save(os.path.join(args.output_dir, out_name))

            if has_gt:
                gt_path = os.path.join(args.input_dir, fname.replace('_sat.jpg','_mask.png'))
                gt = mask_to_label(np.array(Image.open(gt_path).convert('RGB')))
                all_pred.append(pred); all_true.append(gt)

    if has_gt:
        miou = mean_iou(np.stack(all_pred), np.stack(all_true))
        print(f'\nEnsemble val mIoU: {miou:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',  required=True,  help='directory with *_sat.jpg')
    parser.add_argument('--output_dir', required=True,  help='directory to write *_mask.png')
    parser.add_argument('--ckpt_c',     default='../checkpoints/deeplab101_best.pth')
    parser.add_argument('--ckpt_d',     default='../checkpoints_psp/pspnet_best.pth')
    args = parser.parse_args()
    run(args)