import os
import argparse
import torch
import torchvision.utils as vutils
from tqdm import tqdm

from UNet import UNet
from utils import beta_scheduler


def get_ddim_schedule(T=1000, n_steps=50):
    """Uniform timestep scheduler: 981, 961, ..., 1 (50 steps)."""
    step_size = T // n_steps
    timesteps = list(range(T - step_size, -1, -step_size))
    timesteps = [t + 1 for t in timesteps]
    return timesteps


def ddim_sample(model, x_T, betas, timesteps, eta=0.0, device='cuda'):
    alphas     = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0).to(device)

    x = x_T.to(device)
    model.eval()

    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps, leave=False)):
            ab_t    = alpha_bars[t - 1]
            ab_prev = alpha_bars[timesteps[i+1]-1] if i+1 < len(timesteps) else torch.tensor(1.0, device=device)

            t_tensor = torch.full((x.shape[0],), t, dtype=torch.long, device=device)
            eps      = model(x, t_tensor)

            x0_pred = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()

            sigma  = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
            dir_xt = (1 - ab_prev - sigma ** 2).sqrt() * eps
            z      = torch.randn_like(x) if (eta > 0 and t > 1) else torch.zeros_like(x)
            x      = ab_prev.sqrt() * x0_pred + dir_xt + sigma * z

    return x


def slerp(x0, x1, alpha):
    x0_flat = x0.view(1, -1)
    x1_flat = x1.view(1, -1)
    dot   = (x0_flat * x1_flat).sum() / (x0_flat.norm() * x1_flat.norm())
    dot   = dot.clamp(-1, 1)
    theta = torch.acos(dot)
    if theta.abs() < 1e-6:
        return (1 - alpha) * x0 + alpha * x1
    return (torch.sin((1 - alpha) * theta) / torch.sin(theta)) * x0 + \
           (torch.sin(alpha * theta)       / torch.sin(theta)) * x1


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet().to(device)
    ckpt  = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded UNet from {args.model_path}")

    T         = 1000
    betas     = beta_scheduler(T).float().to(device)
    timesteps = get_ddim_schedule(T, n_steps=50)
    print(f"Timesteps: {timesteps[:5]} ... {timesteps[-5:]}")

    os.makedirs(args.output_dir, exist_ok=True)

    noise_files = sorted([f for f in os.listdir(args.noise_dir) if f.endswith('.pt')])
    for fname in noise_files:
        idx    = os.path.splitext(fname)[0]
        x_T    = torch.load(os.path.join(args.noise_dir, fname), map_location=device)
        result = ddim_sample(model, x_T, betas, timesteps, eta=0.0, device=device)
        vutils.save_image(result, os.path.join(args.output_dir, f"{idx}.png"), normalize=True)
        print(f"Saved {idx}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_dir',  type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()
    main(args)