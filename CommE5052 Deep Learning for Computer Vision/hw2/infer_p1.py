import os
import argparse
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms.functional as TF

from UNet import UNet
from train_p1 import ConditionalUNet, make_ddpm_schedule

# ─── Reproducibility ────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ─── DDPM Sampling with Classifier-Free Guidance ────────────────────────────
@torch.no_grad()
def sample(model, betas, alphas, alpha_bars, device,
           class_label, dataset_id, n_samples=50,
           T=1000, guidance_scale=3.0, img_size=32,
           record_steps=None):
    """
    Generate n_samples images for a given (class_label, dataset_id).
    record_steps: list of timesteps to save intermediate images (for report)
    Returns: list of PIL images, and optionally dict of intermediate PIL images
    """
    model.eval()
    B = n_samples

    class_t  = torch.full((B,), class_label,  dtype=torch.long, device=device)
    ds_t     = torch.full((B,), dataset_id,    dtype=torch.long, device=device)
    null_cls = torch.full((B,), 10,            dtype=torch.long, device=device)
    null_ds  = torch.full((B,), 2,             dtype=torch.long, device=device)

    x = torch.randn(B, 3, img_size, img_size, device=device)

    intermediates = {}  # t -> PIL image of sample[0]

    for t_val in tqdm(range(T, 0, -1), desc=f"Sampling digit {class_label}", leave=False):
        t_tensor = torch.full((B,), t_val, dtype=torch.long, device=device)

        # CFG: conditional and unconditional predictions
        eps_cond   = model(x, t_tensor, class_t,  ds_t)
        eps_uncond = model(x, t_tensor, null_cls,  null_ds)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        alpha_t    = alphas[t_val - 1].to(device)
        alpha_bar_t = alpha_bars[t_val - 1].to(device)
        beta_t     = betas[t_val - 1].to(device)

        # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps) + sigma_t * z
        coef = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        mean = (1 / torch.sqrt(alpha_t)) * (x - coef * eps)

        if t_val > 1:
            z = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
        else:
            z = torch.zeros_like(x)
            sigma_t = 0.0

        x = mean + sigma_t * z

        # Record intermediate for report visualization
        if record_steps is not None and t_val in record_steps:
            img0 = x[0].clamp(-1, 1)
            img0 = (img0 + 1) / 2  # [0,1]
            img0 = TF.to_pil_image(img0.cpu())
            img0 = img0.resize((28, 28), Image.BILINEAR)
            intermediates[t_val] = img0

    # Convert to PIL images at 28x28
    x = x.clamp(-1, 1)
    x = (x + 1) / 2  # [0,1]
    pil_images = []
    for i in range(B):
        img = TF.to_pil_image(x[i].cpu())
        img = img.resize((28, 28), Image.BILINEAR)
        pil_images.append(img)

    # Record final step
    if record_steps is not None:
        intermediates[0] = pil_images[0]

    return pil_images, intermediates


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = ConditionalUNet(num_classes=10, num_datasets=2, channel=128).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Noise schedule
    T = args.T
    betas, alphas, alpha_bars = make_ddpm_schedule(T, args.beta_start, args.beta_end)

    # Output directories
    mnistm_dir = os.path.join(args.output_dir, 'mnistm')
    svhn_dir   = os.path.join(args.output_dir, 'svhn')
    os.makedirs(mnistm_dir, exist_ok=True)
    os.makedirs(svhn_dir,   exist_ok=True)

    # Even digits (0,2,4,6,8) -> MNIST-M (dataset_id=0)
    # Odd  digits (1,3,5,7,9) -> SVHN    (dataset_id=1)
    digit_config = {
        0: (0, mnistm_dir),
        2: (0, mnistm_dir),
        4: (0, mnistm_dir),
        6: (0, mnistm_dir),
        8: (0, mnistm_dir),
        1: (1, svhn_dir),
        3: (1, svhn_dir),
        5: (1, svhn_dir),
        7: (1, svhn_dir),
        9: (1, svhn_dir),
    }

    # Steps to record for report visualization (reverse process)
    record_steps = [1000, 800, 600, 400, 200, 1]

    all_intermediates = {}  # digit -> intermediates dict

    for digit in range(10):
        ds_id, out_dir = digit_config[digit]
        imgs, intermediates = sample(
            model, betas, alphas, alpha_bars, device,
            class_label=digit,
            dataset_id=ds_id,
            n_samples=50,
            T=T,
            guidance_scale=args.guidance_scale,
            img_size=32,
            record_steps=record_steps,
        )
        all_intermediates[digit] = intermediates

        for i, img in enumerate(imgs):
            fname = f"{digit}_{i+1:03d}.png"
            img.save(os.path.join(out_dir, fname))

        print(f"Digit {digit}: saved 50 images to {out_dir}")

    # Save reverse process visualization images
    vis_dir = os.path.join(args.output_dir, 'reverse_process')
    os.makedirs(vis_dir, exist_ok=True)
    for digit in [0, 1]:
        for t_step in record_steps:
            if t_step in all_intermediates[digit]:
                img = all_intermediates[digit][t_step]
                fname = f"digit{digit}_t{t_step:04d}.png"
                img.save(os.path.join(vis_dir, fname))
    print(f"Reverse process images saved to {vis_dir}")
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',     type=str, required=True)
    parser.add_argument('--output_dir',     type=str, required=True)
    parser.add_argument('--T',              type=int,   default=1000)
    parser.add_argument('--beta_start',     type=float, default=1e-4)
    parser.add_argument('--beta_end',       type=float, default=0.02)
    parser.add_argument('--guidance_scale', type=float, default=3.0)
    parser.add_argument('--seed',           type=int,   default=42)
    args = parser.parse_args()
    main(args)