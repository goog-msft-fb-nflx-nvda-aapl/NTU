import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

class OfficeHomeTestDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row['filename'])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, row['id'], row['filename']

def main():
    csv_path  = sys.argv[1]   # e.g. hw1_hiddendata/p1_data/office/test.csv
    img_dir   = sys.argv[2]   # e.g. hw1_hiddendata/p1_data/office/test/
    out_path  = sys.argv[3]   # e.g. output_p1/test_pred.csv

    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'ckpt', 'best_C.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 65)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds = OfficeHomeTestDataset(csv_path, img_dir, tf)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    ids, filenames, preds = [], [], []
    with torch.no_grad():
        for imgs, batch_ids, batch_fnames in loader:
            out = model(imgs.to(device))
            pred = out.argmax(1).cpu().tolist()
            ids.extend(batch_ids.tolist())
            filenames.extend(batch_fnames)
            preds.extend(pred)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    df_out = pd.DataFrame({'id': ids, 'filename': filenames, 'label': preds})
    df_out = df_out.sort_values('id').reset_index(drop=True)
    df_out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

if __name__ == '__main__':
    main()