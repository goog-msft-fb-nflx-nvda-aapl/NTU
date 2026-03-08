import os
import glob
from PIL import Image
from torch.utils.data import Dataset

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