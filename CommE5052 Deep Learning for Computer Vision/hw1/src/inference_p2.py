import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from deeplabv3plus import DeepLabV3Plus
from dataset import label_to_color

def main():
    test_dir = sys.argv[1]
    out_dir  = sys.argv[2]
    ckpt     = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ckpt_p2', 'best_deeplab.pth')

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DeepLabV3Plus(n_classes=7).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    tf = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    for fname in sorted(os.listdir(test_dir)):
        if not fname.endswith('_sat.jpg'):
            continue
        fid = fname.replace('_sat.jpg', '')
        img = Image.open(os.path.join(test_dir, fname)).convert('RGB')
        inp = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp).argmax(1).squeeze(0).cpu().numpy()
        label_to_color(pred).save(os.path.join(out_dir, f'{fid}_mask.png'))

    print(f"Saved predictions to {out_dir}")

if __name__ == '__main__':
    main()