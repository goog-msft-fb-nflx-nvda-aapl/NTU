"""
Metropolis Sampling - ImageCopy
Reproduces a target image by sampling pixels proportional to luminance (f = brightness).
"""

import numpy as np
from PIL import Image
import argparse
import os

def luminance(rgb):
    """Perceived luminance of an RGB pixel (0-1 range)."""
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

def build_luminance_map(img_array):
    """Compute per-pixel luminance for the target image."""
    img_float = img_array.astype(np.float64) / 255.0
    lum = (0.2126 * img_float[:,:,0] +
           0.7152 * img_float[:,:,1] +
           0.0722 * img_float[:,:,2])
    return lum  # shape: (H, W)

def mutate_small(x, y, H, W, sigma=0.02):
    """Small Gaussian mutation around current position (local exploration)."""
    nx = x + np.random.normal(0, sigma * W)
    ny = y + np.random.normal(0, sigma * H)
    # Reflect boundary
    nx = nx % W
    ny = ny % H
    return nx, ny

def mutate_large(H, W):
    """Large (uniform) mutation for ergodicity — escape local modes."""
    return np.random.uniform(0, W), np.random.uniform(0, H)

def metropolis_image_copy(target_img, spp, large_step_prob=0.1, seed=42):
    """
    Metropolis ImageCopy algorithm.

    State space: continuous (x, y) coordinates over the image.
    Target distribution: f(x,y) = luminance(target[y,x])
    Mutation: mix of small Gaussian and large uniform steps.
    Output: reconstructed image via sample accumulation.
    """
    np.random.seed(seed)
    H, W = target_img.shape[:2]
    lum_map = build_luminance_map(target_img)

    total_samples = H * W * spp

    # Accumulation buffers
    accum_color = np.zeros((H, W, 3), dtype=np.float64)
    accum_count = np.zeros((H, W), dtype=np.float64)

    # --- Initialization: find a good starting sample via uniform sampling ---
    # (Hybrid initialization to avoid start-up bias)
    while True:
        x0 = np.random.uniform(0, W)
        y0 = np.random.uniform(0, H)
        ix0, iy0 = int(x0) % W, int(y0) % H
        if lum_map[iy0, ix0] > 0:
            break
    cur_x, cur_y = x0, y0
    cur_f = lum_map[int(cur_y) % H, int(cur_x) % W]

    accepted = 0

    for i in range(total_samples):
        # Propose mutation
        if np.random.random() < large_step_prob:
            prop_x, prop_y = mutate_large(H, W)
        else:
            prop_x, prop_y = mutate_small(cur_x, cur_y, H, W)

        prop_ix = int(prop_x) % W
        prop_iy = int(prop_y) % H
        prop_f = lum_map[prop_iy, prop_ix]

        # Acceptance probability (T is symmetric for both mutations here)
        if cur_f > 0:
            alpha = min(1.0, prop_f / cur_f)
        else:
            alpha = 1.0

        # Accept or reject
        if np.random.random() < alpha:
            cur_x, cur_y = prop_x, prop_y
            cur_f = prop_f
            accepted += 1

        # Record current sample
        ix = int(cur_x) % W
        iy = int(cur_y) % H
        accum_color[iy, ix] += target_img[iy, ix].astype(np.float64)
        accum_count[iy, ix] += 1

    acceptance_rate = accepted / total_samples
    print(f"  spp={spp}: acceptance rate = {acceptance_rate:.3f}")

    # Normalize: divide by count where sampled, leave zeros where unsampled
    result = np.zeros((H, W, 3), dtype=np.float64)
    mask = accum_count > 0
    for c in range(3):
        result[:,:,c][mask] = accum_color[:,:,c][mask] / accum_count[mask]

    # Scale so that total energy (mean brightness) matches target
    target_mean = target_img.astype(np.float64).mean()
    result_mean = result[mask].mean() if mask.any() else 1.0
    if result_mean > 0:
        result *= (target_mean / result_mean)

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def main():
    parser = argparse.ArgumentParser(description="Metropolis ImageCopy")
    parser.add_argument("--input",  default="input.png", help="Target image path")
    parser.add_argument("--outdir", default="results",   help="Output directory")
    parser.add_argument("--spps",   nargs="+", type=int,
                        default=[1, 4, 16, 64, 256],
                        help="List of samples-per-pixel to test")
    parser.add_argument("--large_step_prob", type=float, default=0.1,
                        help="Probability of large (uniform) mutation")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load and resize target to 256x256
    img = Image.open(args.input).convert("RGB").resize((256, 256), Image.LANCZOS)
    img.save(os.path.join(args.outdir, "target_256.png"))
    target = np.array(img)
    print(f"Target image: {target.shape}")

    for spp in args.spps:
        print(f"Running Metropolis with spp={spp} ...")
        result = metropolis_image_copy(target, spp,
                                       large_step_prob=args.large_step_prob,
                                       seed=args.seed)
        out_path = os.path.join(args.outdir, f"metropolis_spp{spp:04d}.png")
        Image.fromarray(result).save(out_path)
        print(f"  Saved: {out_path}")

    print("Done.")

if __name__ == "__main__":
    main()
