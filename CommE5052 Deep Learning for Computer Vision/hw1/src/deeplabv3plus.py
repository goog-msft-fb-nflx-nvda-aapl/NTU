import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.b0 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU())
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=6,  dilation=6,  bias=False), nn.BatchNorm2d(out_ch), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU())
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU())
        self.proj = nn.Sequential(nn.Conv2d(out_ch*5, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(), nn.Dropout(0.5))

    def forward(self, x):
        h, w = x.shape[2:]
        gap = F.interpolate(self.gap(x), size=(h,w), mode='bilinear', align_corners=False)
        return self.proj(torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x), gap], dim=1))

class DeepLabV3Plus(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Use dilated conv in layer3/4
        backbone.layer3[0].conv2.stride = (1,1)
        backbone.layer3[0].downsample[0].stride = (1,1)
        for m in backbone.layer3[1:]:
            m.conv2.dilation = (2,2); m.conv2.padding = (2,2)
        backbone.layer4[0].conv2.stride = (1,1)
        backbone.layer4[0].downsample[0].stride = (1,1)
        for m in backbone.layer4:
            m.conv2.dilation = (4,4); m.conv2.padding = (4,4)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # low-level: 256ch, stride 4
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4  # high-level: 2048ch

        self.aspp = ASPP(2048, 256)
        self.low_proj = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Conv2d(256+48, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, n_classes, 1)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        x = self.layer0(x)
        low = self.layer1(x)   # stride 4
        x   = self.layer2(low)
        x   = self.layer3(x)
        x   = self.layer4(x)
        x   = self.aspp(x)
        x   = F.interpolate(x, size=low.shape[2:], mode='bilinear', align_corners=False)
        x   = torch.cat([x, self.low_proj(low)], dim=1)
        x   = self.decoder(x)
        return F.interpolate(x, size=(h,w), mode='bilinear', align_corners=False)