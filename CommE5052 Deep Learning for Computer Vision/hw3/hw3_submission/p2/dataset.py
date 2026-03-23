import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict


class CaptionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, tokenizer, transform=None, max_length=64):
        self.image_dir = image_dir
        self.transform = transform
        self.max_length = max_length
        self.tokenizer = tokenizer

        with open(annotation_file, 'r') as f:
            data = json.load(f)

        # Group captions by image_id
        captions_by_id = defaultdict(list)
        for ann in data['annotations']:
            captions_by_id[ann['image_id']].append(ann['caption'])

        # Build flat list: one entry per (image, caption) pair
        self.samples = []
        for image_id, captions in captions_by_id.items():
            filename = f"{image_id:012d}.jpg"
            for cap in captions:
                self.samples.append((filename, cap))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, caption = self.samples[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Tokenize: <|im_start|> caption <|im_end|>
        im_start = self.tokenizer.encode('<|im_start|>')[0]
        im_end = self.tokenizer.encode('<|im_end|>')[0]
        pad_id = 151643

        cap_ids = self.tokenizer.encode(caption)
        token_ids = [im_start] + cap_ids + [im_end]

        # Pad or truncate
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [pad_id] * (self.max_length - len(token_ids))

        return image, torch.tensor(token_ids, dtype=torch.long)


class ValDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, filename
