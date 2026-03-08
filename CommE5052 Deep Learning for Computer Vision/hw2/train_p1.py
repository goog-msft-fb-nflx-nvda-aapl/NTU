import os
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from UNet import UNet

# ─── Reproducibility ────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ─── Dataset ────────────────────────────────────────────────────────────────
class DigitDataset(Dataset):
    """Loads MNIST-M (dataset_id=0) and SVHN (dataset_id=1) jointly."""
    def __init__(self, mnistm_root, svhn_root, transform=None):
        self.transform = transform
        self.samples = []  # (img_path, digit_label, dataset_id)

        for csv_path, img_dir, ds_id in [
            (os.path.join(mnistm_root, 'train.csv'), os.path.join(mnistm_root, 'data'), 0),
            (os.path.join(svhn_root,   'train.csv'), os.path.join(svhn_root,   'data'), 1),
        ]:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.samples.append((
                    os.path.join(img_dir, row['image_name']),
                    int(row['label']),
                    ds_id,
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, ds_id = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, ds_id


# ─── Conditional UNet wrapper ────────────────────────────────────────────────
class ConditionalUNet(nn.Module):
    """
    Wraps the provided UNet and injects class + dataset embeddings
    into the time embedding via addition (same dimension trick).
    """
    def __init__(self, num_classes=10, num_datasets=2, channel=128):
        super().__init__()
        self.unet = UNet(in_channel=3, channel=channel)
        time_dim = channel * 4

        # Learnable embeddings for digit (0-9) + null token (10)
        self.class_emb = nn.Embedding(num_classes + 1, time_dim)
        # Learnable embeddings for dataset (0=mnistm, 1=svhn) + null token (2)
        self.dataset_emb = nn.Embedding(num_datasets + 1, time_dim)

        nn.init.normal_(self.class_emb.weight, std=0.02)
        nn.init.normal_(self.dataset_emb.weight, std=0.02)

    def forward(self, x, t, class_label, dataset_id):
        """
        x          : (B, 3, H, W)
        t          : (B,) int timesteps
        class_label: (B,) int in [0,9] or 10 for null
        dataset_id : (B,) int in [0,1] or 2 for null
        """
        # Get UNet's internal time embedding
        time_emb = self.unet.time(t)                     # (B, time_dim)
        class_emb   = self.class_emb(class_label)        # (B, time_dim)
        dataset_emb = self.dataset_emb(dataset_id)       # (B, time_dim)

        # Inject conditioning by adding to time_emb, then override unet.time
        cond_emb = time_emb + class_emb + dataset_emb

        # Monkey-patch: temporarily replace time so UNet's forward uses our cond_emb
        # We call UNet internals directly replicating its forward() with cond_emb
        return self._unet_forward(x, cond_emb)

    def _unet_forward(self, x, time_embed):
        """Replicate UNet.forward but with precomputed time_embed."""
        u = self.unet
        feats = []

        x = u.down1(x);         feats.append(x)
        x = u.down2(x, time_embed);  feats.append(x)
        x = u.down3(x, time_embed);  feats.append(x)
        x = u.down4(x);         feats.append(x)
        x = u.down5(x, time_embed);  feats.append(x)
        x = u.down6(x, time_embed);  feats.append(x)
        x = u.down7(x);         feats.append(x)
        x = u.down8(x, time_embed);  feats.append(x)
        x = u.down9(x, time_embed);  feats.append(x)
        x = u.down10(x);        feats.append(x)
        x = u.down11(x, time_embed); feats.append(x)
        x = u.down12(x, time_embed); feats.append(x)
        x = u.down13(x);        feats.append(x)
        x = u.down14(x, time_embed); feats.append(x)
        x = u.down15(x, time_embed); feats.append(x)
        x = u.down16(x);        feats.append(x)
        x = u.down17(x, time_embed); feats.append(x)
        x = u.down18(x, time_embed); feats.append(x)

        x = u.mid1(x, time_embed)
        x = u.mid2(x, time_embed)

        x = u.up1(torch.cat((x, feats.pop()), 1), time_embed)
        x = u.up2(torch.cat((x, feats.pop()), 1), time_embed)
        x = u.up3(torch.cat((x, feats.pop()), 1), time_embed)
        x = u.up4(x)
        x = u.up5(torch.cat((x, feats.pop()), 1), time_embed)
        x = u.up6(torch.cat((x, feats.pop()), 1), time_embed)
        x = u.up7(torch.cat((x, feats.pop()), 1), time_embed)
        x = u.up8(x)
        x = u.up9(torch.cat((x, feats.pop()), 1), time_embed)
        x = u.up10(torch.cat((x, feats.pop()), 1), time_embed)
        x = u.up11(torch.cat((x, feats.pop()), 1), time_embed)
        x = u.up12(x)
        x = u.up13(torch.cat((x, feats.pop()), 1), time_embed)
        x = u.up14(torch.cat((x, feats.pop()), 1), time_embed)
        x = u.up15(torch.cat((x, feats.pop()), 1), time_embed)
        x = u.up16(x)
        x = u.up17(torch.cat((x, feats.pop()), 1), time_embed)
        x = u.up18(torch.cat((x, feats.pop()), 1), time_embed)
        x = u.up19(torch.cat((x, feats.pop()), 1), time_embed)
        x = u.up20(x)
        x = u.up21(torch.cat((x, feats.pop()), 1), time_embed)
        x = u.up22(torch.cat((x, feats.pop()), 1), time_embed)
        x = u.up23(torch.cat((x, feats.pop()), 1), time_embed)

        from UNet import spatial_unfold
        return spatial_unfold(u.out(x), 1)


# ─── DDPM Noise Schedule ─────────────────────────────────────────────────────
def make_ddpm_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    betas      = torch.linspace(beta_start, beta_end, T)          # (T,)
    alphas     = 1.0 - betas                                       # (T,)
    alpha_bars = torch.cumprod(alphas, dim=0)                      # (T,)
    return betas, alphas, alpha_bars


# ─── Training ────────────────────────────────────────────────────────────────
def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Noise schedule
    T = args.T
    betas, alphas, alpha_bars = make_ddpm_schedule(T, args.beta_start, args.beta_end)
    alpha_bars = alpha_bars.to(device)

    # Dataset & Dataloader
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = DigitDataset(
        mnistm_root=os.path.join(args.data_root, 'mnistm'),
        svhn_root=os.path.join(args.data_root, 'svhn'),
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
    print(f"Dataset size: {len(dataset)}")

    # Model
    model = ConditionalUNet(num_classes=10, num_datasets=2, channel=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for imgs, labels, ds_ids in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs   = imgs.to(device)       # (B, 3, 28, 28)
            labels = labels.to(device)     # (B,)
            ds_ids = ds_ids.to(device)     # (B,)
            B = imgs.size(0)

            # Sample random timesteps
            t = torch.randint(1, T + 1, (B,), device=device)  # (B,)

            # Sample noise
            noise = torch.randn_like(imgs)

            # Forward diffusion: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps
            ab = alpha_bars[t - 1].view(B, 1, 1, 1)
            x_t = torch.sqrt(ab) * imgs + torch.sqrt(1 - ab) * noise

            # Classifier-Free Guidance: randomly drop conditions with prob p_uncond
            p = args.p_uncond
            # Drop class label
            class_mask   = (torch.rand(B, device=device) < p)
            class_input  = labels.clone()
            class_input[class_mask] = 10   # null token

            # Drop dataset id
            ds_mask  = (torch.rand(B, device=device) < p)
            ds_input = ds_ids.clone()
            ds_input[ds_mask] = 2          # null token

            # Predict noise
            pred_noise = model(x_t, t, class_input, ds_input)

            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.ckpt_dir, f'model_epoch{epoch:04d}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',   type=str, default='hw2_data/digits')
    parser.add_argument('--ckpt_dir',    type=str, default='checkpoints_p1')
    parser.add_argument('--epochs',      type=int, default=100)
    parser.add_argument('--batch_size',  type=int, default=128)
    parser.add_argument('--lr',          type=float, default=2e-4)
    parser.add_argument('--T',           type=int, default=1000)
    parser.add_argument('--beta_start',  type=float, default=1e-4)
    parser.add_argument('--beta_end',    type=float, default=0.02)
    parser.add_argument('--p_uncond',    type=float, default=0.1,
                        help='probability of dropping condition (CFG training)')
    parser.add_argument('--save_every',  type=int, default=10)
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()
    train(args)
