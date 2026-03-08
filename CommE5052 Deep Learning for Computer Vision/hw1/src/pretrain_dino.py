import sys
sys.path.insert(0, '/home/jtan/CommE5052/hw1/dino')

import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import main_dino
import utils

class FlatImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = sorted(glob.glob(os.path.join(folder, '*.jpg')))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label

if __name__ == '__main__':
    # Monkey-patch ImageFolder with our flat dataset
    import torchvision.datasets as tvd
    tvd.ImageFolder = lambda root, transform: FlatImageDataset(root, transform)

    sys.argv = [
        'main_dino.py',
        '--arch', 'resnet50',
        '--data_path', '/home/jtan/CommE5052/hw1/data_2025/p1_data/mini/train',
        '--output_dir', '/home/jtan/CommE5052/hw1/dino_output',
        '--epochs', '200',
        '--batch_size_per_gpu', '128',
        '--optimizer', 'adamw',
        '--lr', '0.0005',
        '--min_lr', '1e-6',
        '--weight_decay', '0.04',
        '--weight_decay_end', '0.4',
        '--global_crops_scale', '0.4', '1.',
        '--local_crops_scale', '0.05', '0.4',
        '--local_crops_number', '8',
        '--warmup_epochs', '10',
        '--num_workers', '8',
        '--saveckp_freq', '20',
    ]
    main_dino.train_dino(main_dino.get_args_parser().parse_args(sys.argv[1:]))
