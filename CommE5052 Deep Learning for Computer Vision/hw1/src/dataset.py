import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# 7 classes: map RGB -> class index
COLOR_MAP = {
    (0,   0,   0):   0,  # Unknown
    (0,   0,   255): 1,  # Water
    (0,   255, 0):   2,  # Forest
    (0,   255, 255): 3,  # Agriculture
    (255, 0,   255): 4,  # Urban
    (255, 255, 0):   5,  # Rangeland
    (255, 255, 255): 6,  # Barren
}

INDEX_TO_COLOR = {v: k for k, v in COLOR_MAP.items()}

def mask_to_label(mask_pil):
    mask = np.array(mask_pil.convert('RGB'))
    label = np.zeros(mask.shape[:2], dtype=np.int64)
    for rgb, idx in COLOR_MAP.items():
        match = np.all(mask == np.array(rgb), axis=-1)
        label[match] = idx
    return label

def label_to_color(label_np):
    h, w = label_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, rgb in INDEX_TO_COLOR.items():
        color[label_np == idx] = rgb
    return Image.fromarray(color)

class SegDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.data_dir = data_dir
        self.augment = augment
        ids = sorted(set(f.split('_')[0] for f in os.listdir(data_dir) if f.endswith('_sat.jpg')))
        self.ids = ids
        self.img_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.aug_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]
        img  = Image.open(os.path.join(self.data_dir, f'{fid}_sat.jpg')).convert('RGB')
        mask = Image.open(os.path.join(self.data_dir, f'{fid}_mask.png')).convert('RGB')
        if self.augment:
            # Joint transform via same random seed
            import random, torchvision.transforms.functional as F
            if random.random() > 0.5:
                img  = F.hflip(img)
                mask = F.hflip(mask)
            if random.random() > 0.5:
                img  = F.vflip(img)
                mask = F.vflip(mask)
            img = transforms.ColorJitter(0.2,0.2,0.2,0.1)(img)
        label = mask_to_label(mask)
        return self.img_tf(img), label