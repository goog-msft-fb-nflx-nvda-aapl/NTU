import os
import numpy as np
from PIL import Image

# ── Exposure times for memorial0061..memorial0076 (seconds) ──────────────────
EXPOSURE_TIMES = np.array([
    32, 16, 8, 4, 2, 1,
    0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,
    0.0078125, 0.00390625, 0.001953125, 0.0009765625
], dtype=np.float32)

def load_images(image_dir):
    """
    Load all PNG images from image_dir, sorted by filename.
    Returns:
        images : list of np.ndarray, shape (H, W, 3), dtype uint8
        times  : np.ndarray of exposure times, shape (P,)
    """
    filenames = sorted(f for f in os.listdir(image_dir) if f.endswith('.png'))
    assert len(filenames) == len(EXPOSURE_TIMES), \
        f"Expected {len(EXPOSURE_TIMES)} images, found {len(filenames)}"

    images = []
    for fname in filenames:
        path = os.path.join(image_dir, fname)
        img  = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
        images.append(img)
        print(f"  Loaded {fname}  shape={img.shape}")

    return images, EXPOSURE_TIMES

def hat_weight(z, z_min=0, z_max=255):
    """
    Hat (tent) weighting function.
    Gives full weight to mid-range values, tapers to 0 at extremes.
    """
    midpoint = (z_min + z_max) / 2.0
    return np.where(z <= midpoint, z - z_min + 1, z_max - z + 1).astype(np.float32)


def solve_crf(images, log_times, lam=50, n_samples=50, seed=42):
    """
    Solve for the camera response function g(z) = log(f^-1(z))
    using Debevec & Malik's linear system (Eq. 6, SIGGRAPH 1997).

    Args:
        images    : list of P images, each (H, W) uint8  -- single channel
        log_times : array of shape (P,) -- log exposure times
        lam       : smoothness regularization weight
        n_samples : number of pixel locations to sample
        seed      : random seed for reproducibility

    Returns:
        g    : np.ndarray shape (256,) -- the response curve
        lE   : np.ndarray shape (N,)   -- log irradiance at sampled pixels
    """
    P = len(images)          # number of exposures
    N = n_samples            # number of sampled pixels
    n = 256                  # number of intensity levels

    H, W = images[0].shape

    # ── Sample N pixels, avoiding pure-blue border regions ───────────────────
    # Build a validity mask: exclude pixels where the image is pure blue
    # (R<10, G<10, B>245 in the original RGB — but here we work per-channel,
    #  so we simply avoid near-zero/near-255 pixels across ALL exposures)
    rng = np.random.default_rng(seed)
    
    # Pick candidate pixels and keep only those that are neither saturated
    # nor black in a majority of exposures (so they carry useful information)
    candidates = rng.choice(H * W, size=N * 20, replace=False)
    flat_imgs  = np.stack([img.ravel() for img in images], axis=1)  # (H*W, P)
    
    good = []
    for idx in candidates:
        vals = flat_imgs[idx]          # pixel value across all P exposures
        # Reject if more than half the exposures are saturated or black
        if np.sum((vals <= 2) | (vals >= 253)) < P // 2:
            good.append(idx)
        if len(good) == N:
            break

    assert len(good) == N, f"Could not find {N} good sample pixels"
    Z = flat_imgs[good]   # shape (N, P) -- integer pixel values 0..255

    # ── Build the linear system  A·x = b ─────────────────────────────────────
    # Unknowns x = [g(0), g(1), ..., g(255), lE_1, ..., lE_N]
    #            total = 256 + N
    # Equations:
    #   (1) data equations: N*P rows
    #   (2) smoothness equations: 254 rows  (g''(z) for z=1..254)
    #   (3) fix g(128) = 0: 1 row

    n_data   = N * P
    n_smooth = n - 2          # 254
    n_rows   = n_data + n_smooth + 1
    n_cols   = n + N          # 256 unknowns for g + N unknowns for lE

    A = np.zeros((n_rows, n_cols), dtype=np.float32)
    b = np.zeros(n_rows,           dtype=np.float32)

    row = 0

    # ── (1) Data equations ────────────────────────────────────────────────────
    for i in range(N):
        for j in range(P):
            z   = int(Z[i, j])
            w   = hat_weight(z)
            A[row, z]         =  w          # g(z)
            A[row, n + i]     = -w          # -lE_i
            b[row]            =  w * log_times[j]
            row += 1

    # ── (2) Fix the middle value so g is not underdetermined ─────────────────
    A[row, 128] = 1.0
    b[row]      = 0.0
    row += 1

    # ── (3) Smoothness equations ──────────────────────────────────────────────
    for z in range(1, n - 1):
        w = hat_weight(z)
        A[row, z - 1] =  lam * w
        A[row, z    ] = -2 * lam * w
        A[row, z + 1] =  lam * w
        b[row]        =  0.0
        row += 1

    # ── Solve via least squares (SVD) ─────────────────────────────────────────
    print(f"  Solving {A.shape[0]} x {A.shape[1]} system ...")
    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

    g  = x[:n]        # response curve, shape (256,)
    lE = x[n:]        # log irradiances at sample pixels, shape (N,)

    return g, lE

def build_hdr(images, g_channels, log_times):
    """
    Assemble the HDR radiance map from all exposures.

    Args:
        images     : list of P images, each (H, W, 3) uint8
        g_channels : list of 3 arrays, each shape (256,) -- one per R/G/B
        log_times  : array shape (P,) -- log exposure times

    Returns:
        hdr : np.ndarray shape (H, W, 3) float32 -- log radiance map
    """
    P = len(images)
    H, W = images[0].shape[:2]
    hdr  = np.zeros((H, W, 3), dtype=np.float32)

    for c in range(3):
        g = g_channels[c]
        num = np.zeros((H, W), dtype=np.float32)  # weighted numerator
        den = np.zeros((H, W), dtype=np.float32)  # sum of weights

        for j in range(P):
            Z = images[j][:, :, c].astype(np.int32)   # (H, W)
            w = hat_weight(Z).astype(np.float32)       # (H, W)

            num += w * (g[Z] - log_times[j])
            den += w

        # Avoid division by zero — pixels where all exposures are saturated/black
        valid      = den > 0
        hdr[:, :, c][valid]  = num[valid] / den[valid]
        hdr[:, :, c][~valid] = np.min(hdr[:, :, c][valid])  # fallback

        print(f"  Channel {'RGB'[c]}: log-radiance range "
              f"[{hdr[:,:,c].min():.2f}, {hdr[:,:,c].max():.2f}]")

    return hdr  # still in log space
def float_to_rgbe(hdr):
    """
    Convert linear float HDR image to Radiance RGBE (4 bytes per pixel).
    Gregory Ward's RGBE encoding from the Radiance format spec.

    Args:
        hdr  : np.ndarray (H, W, 3) float32, linear radiance

    Returns:
        rgbe : np.ndarray (H, W, 4) uint8
    """
    H, W = hdr.shape[:2]
    rgbe  = np.zeros((H, W, 4), dtype=np.uint8)

    # Max of R,G,B determines the shared exponent
    max_val = np.max(hdr, axis=2)          # (H, W)
    valid   = max_val > 1e-32

    # Compute exponent e such that max_val = mantissa * 2^(e-128)
    # mantissa is in [0.5, 1.0)
    exp  = np.zeros((H, W), dtype=np.float32)
    mant = np.zeros((H, W), dtype=np.float32)

    exp[valid]  = np.floor(np.log2(max_val[valid])) + 1
    mant[valid] = max_val[valid] / np.exp2(exp[valid])  # normalised to [0.5,1)

    scale = np.zeros((H, W), dtype=np.float32)
    scale[valid] = mant[valid] * 256.0 / max_val[valid]

    rgbe[:, :, 0] = np.clip(hdr[:, :, 0] * scale, 0, 255).astype(np.uint8)
    rgbe[:, :, 1] = np.clip(hdr[:, :, 1] * scale, 0, 255).astype(np.uint8)
    rgbe[:, :, 2] = np.clip(hdr[:, :, 2] * scale, 0, 255).astype(np.uint8)
    rgbe[:, :, 3] = np.clip(exp + 128, 0, 255).astype(np.uint8)  # biased exponent

    return rgbe


def save_hdr(filename, hdr):
    """
    Save a linear float HDR image as a Radiance .hdr file (RGBE format).
    Writes a minimal valid header + uncompressed scanlines.

    Args:
        filename : output path ending in .hdr
        hdr      : np.ndarray (H, W, 3) float32, linear radiance
    """
    H, W = hdr.shape[:2]
    rgbe = float_to_rgbe(hdr)

    with open(filename, 'wb') as f:
        # ── Radiance HDR header ───────────────────────────────────────────────
        f.write(b"#?RADIANCE\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n")
        f.write(b"\n")
        # Image size line: -Y height +X width  (top-to-bottom, left-to-right)
        f.write(f"-Y {H} +X {W}\n".encode('ascii'))

        # ── Scanlines (uncompressed) ──────────────────────────────────────────
        # Each scanline: 4 bytes per pixel, written as flat RGBE
        for row in range(H):
            f.write(rgbe[row].tobytes())

    print(f"  Saved {filename}  ({H}x{W}, uncompressed RGBE)")

def rgb_to_luminance(hdr):
    """CIE luminance from linear RGB."""
    return 0.2126 * hdr[:,:,0] + 0.7152 * hdr[:,:,1] + 0.0722 * hdr[:,:,2]


def tonemap_reinhard(hdr, a=0.18, delta=1e-6):
    """
    Reinhard et al. global tone mapping operator.
    'Photographic Tone Reproduction for Digital Images', SIGGRAPH 2002.

    Args:
        hdr   : (H, W, 3) float32 linear radiance
        a     : key value (exposure control), typically 0.18
        delta : small offset to avoid log(0)

    Returns:
        ldr   : (H, W, 3) float32 in [0, 1]
    """
    H, W = hdr.shape[:2]

    # ── Step 1: log-average luminance ────────────────────────────────────────
    L_w      = rgb_to_luminance(hdr)                          # (H, W)
    L_w_bar  = np.exp(np.mean(np.log(delta + L_w)))           # scalar
    print(f"  Log-average luminance L_w_bar = {L_w_bar:.4f}")

    # ── Step 2: scale luminance ───────────────────────────────────────────────
    L_scaled = (a / L_w_bar) * L_w                            # (H, W)

    # ── Step 3: display compression ──────────────────────────────────────────
    L_d = L_scaled / (1.0 + L_scaled)                         # (H, W), in [0,1)

    # ── Apply same scale to each colour channel (preserves hue) ──────────────
    # Avoid division by zero where L_w is zero
    scale        = np.zeros_like(L_w)
    nonzero      = L_w > 1e-10
    scale[nonzero] = L_d[nonzero] / L_w[nonzero]

    ldr          = hdr * scale[:, :, np.newaxis]              # (H, W, 3)
    ldr          = np.clip(ldr, 0.0, 1.0)

    return ldr


def save_ldr(filename, ldr, gamma=2.2):
    """
    Save tone-mapped [0,1] float image as 8-bit PNG with gamma correction.

    Args:
        filename : output path ending in .png
        ldr      : (H, W, 3) float32 in [0, 1]
        gamma    : display gamma (2.2 for standard monitors)
    """
    # Apply gamma correction
    ldr_gamma = np.clip(ldr, 0.0, 1.0) ** (1.0 / gamma)
    img_8bit  = (ldr_gamma * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img_8bit, 'RGB').save(filename)
    print(f"  Saved {filename}")

def rgb_to_gray(img):
    """Luminance-weighted grayscale from uint8 RGB."""
    return (0.2126 * img[:,:,0] +
            0.7152 * img[:,:,1] +
            0.0722 * img[:,:,2]).astype(np.float32)


def get_mtb(gray, exclude_range=4):
    """
    Compute the Median Threshold Bitmap and exclusion mask.

    Args:
        gray          : (H, W) float32 grayscale image
        exclude_range : pixels within this range of median are excluded

    Returns:
        mtb      : (H, W) bool  -- True where pixel > median
        excl     : (H, W) bool  -- True where pixel is NOT near median
    """
    median = np.median(gray)
    mtb    = gray > median
    excl   = np.abs(gray - median) > exclude_range
    return mtb, excl

def shift_image(img, dx, dy):
    """Shift a 2D array by (dx, dy) pixels, filling vacated areas with 0."""
    if dx == 0 and dy == 0:
        return img.copy()
    H, W   = img.shape[0], img.shape[1]
    result = np.zeros_like(img)
    dr = slice(max(0,  dy), H + min(0,  dy))
    dc = slice(max(0,  dx), W + min(0,  dx))
    sr = slice(max(0, -dy), H - max(0,  dy))
    sc = slice(max(0, -dx), W - max(0,  dx))
    result[dr, dc] = img[sr, sc]
    return result

def mtb_align(ref, src, max_levels=6):
    """
    Align src to ref using Ward's MTB algorithm.

    Args:
        ref        : (H, W, 3) uint8 reference image
        src        : (H, W, 3) uint8 image to align
        max_levels : pyramid depth

    Returns:
        aligned : (H, W, 3) uint8 -- src shifted to align with ref
        (dx, dy): total shift applied
    """
    ref_gray = rgb_to_gray(ref)
    src_gray = rgb_to_gray(src)

    total_dx, total_dy = 0, 0

    for level in range(max_levels - 1, -1, -1):
        # Downsample by 2^level
        scale     = 2 ** level
        ref_small = ref_gray[::scale, ::scale]
        src_small = src_gray[::scale, ::scale]

        # Current accumulated shift at this scale
        cur_dx = total_dx // scale
        cur_dy = total_dy // scale

        # Apply current shift to src at this level
        src_shifted = shift_image(src_small, cur_dx, cur_dy)

        ref_mtb, ref_excl = get_mtb(ref_small)
        src_mtb, src_excl = get_mtb(src_shifted)

        # Initialise with the (0,0) error — only update on strict improvement
        best_err = np.sum((ref_mtb ^ src_mtb) & (ref_excl & src_excl))
        best_dx, best_dy = 0, 0
        for ddx in [-1, 0, 1]:
            for ddy in [-1, 0, 1]:
                if ddx == 0 and ddy == 0:
                    continue                    # already computed above
                s_mtb  = shift_image(src_mtb,  ddx, ddy)
                s_excl = shift_image(src_excl, ddx, ddy)
                err    = np.sum((ref_mtb ^ s_mtb) & (ref_excl & s_excl))
                if err < best_err:
                    best_err = err
                    best_dx, best_dy = ddx, ddy

        total_dx += best_dx * scale
        total_dy += best_dy * scale

    # Apply final total shift to the full-resolution src
    aligned = np.stack([
        shift_image(src[:, :, c], total_dx, total_dy)
        for c in range(3)
    ], axis=2).astype(np.uint8)

    return aligned, (total_dx, total_dy)


def align_images(images):
    """
    Align all images to the middle exposure as reference.

    Args:
        images : list of P arrays (H, W, 3) uint8

    Returns:
        aligned_images : list of P arrays (H, W, 3) uint8
    """
    ref_idx = len(images) // 2
    ref     = images[ref_idx]
    aligned = []

    print(f"  Reference image index: {ref_idx}")
    for i, img in enumerate(images):
        if i == ref_idx:
            aligned.append(img)
            print(f"  Image {i:2d}: reference (shift 0, 0)")
        else:
            al, (dx, dy) = mtb_align(ref, img, max_levels=6)
            aligned.append(al)
            print(f"  Image {i:2d}: shift ({dx:+d}, {dy:+d})")

    return aligned

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # ── Step 1: Load images ───────────────────────────────────────────────────
    images, times = load_images('.')
    log_times = np.log(times)

    # ── Step 2: Align images (Ward's MTB) ────────────────────────────────────
    print("\nAligning images (Ward's MTB)...")
    images_aligned = align_images(images)

    # ── Step 3: Solve CRF for each channel ───────────────────────────────────
    print("\nSolving CRF for each channel...")
    g_channels = []
    for c, name in enumerate(['Red', 'Green', 'Blue']):
        print(f"\n  Channel: {name}")
        channel_imgs = [img[:, :, c] for img in images_aligned]
        g, lE = solve_crf(channel_imgs, log_times, lam=50, n_samples=200)
        g_channels.append(g)
        print(f"  g range: [{g.min():.3f}, {g.max():.3f}]")

    # ── Step 4: Plot and save response curves ─────────────────────────────────
    os.makedirs('output', exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(256)
    for g, name, color in zip(g_channels, ['Red', 'Green', 'Blue'], ['r', 'g', 'b']):
        ax.plot(x, g, color=color, label=name)
    ax.set_xlabel('Pixel value Z')
    ax.set_ylabel('log exposure  g(Z)')
    ax.set_title("Recovered Camera Response Curves (Debevec)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('output/crf.png', dpi=150)
    print("\nSaved output/crf.png")

    # ── Step 5: Build HDR radiance map ────────────────────────────────────────
    print("\nBuilding HDR radiance map...")
    log_hdr = build_hdr(images_aligned, g_channels, log_times)
    hdr     = np.exp(log_hdr)
    print(f"\nHDR shape: {hdr.shape}, dtype: {hdr.dtype}")
    print(f"Radiance range: [{hdr.min():.4f}, {hdr.max():.4f}]")

    # ── Step 6: Save .hdr file ────────────────────────────────────────────────
    print("\nSaving HDR file...")
    save_hdr('output/result.hdr', hdr)

    # ── Step 7: Tone map and save ─────────────────────────────────────────────
    print("\nTone mapping (Reinhard global)...")
    ldr = tonemap_reinhard(hdr, a=0.18)
    save_ldr('output/tonemapped_reinhard.png', ldr, gamma=2.2)

    print("\nAll done!")