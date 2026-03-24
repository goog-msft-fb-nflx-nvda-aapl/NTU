import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

DUST3R_DIR = '/home/jtan/CommE5052/dlcv-fall-2025-hw4-goog-msft-fb-nflx-nvda-aapl-main/dust3r'
sys.path.insert(0, DUST3R_DIR)

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

device = 'cuda'
model_path = os.path.join(DUST3R_DIR, 'checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)

data_root = '/home/jtan/CommE5052/data/hw_4_1_data/public/images'
interp_dir = '/home/jtan/CommE5052/data/hw_4_1_data/public/interpolated_images/results_wide_baseline/dynamicrafter_512_wide_baseline_seed12306/3781/dynamicrafter'
img1 = os.path.join(data_root, 'Street/img/image2_000666.png')
img2 = os.path.join(data_root, 'Street/img_south/image_south_1_0245.png')

images_pair = load_images([img1, img2], size=480)
pairs = make_pairs(images_pair, scene_graph='complete', symmetrize=True)
output = inference(pairs, model, device, batch_size=1)
scene_pair = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)
pts3d_pair = scene_pair.get_pts3d()
pts = pts3d_pair[0].detach().cpu().numpy().reshape(-1, 3)
colors_raw = images_pair[0]['img'].squeeze().permute(1,2,0).cpu().numpy()
colors_raw = (colors_raw - colors_raw.min()) / (colors_raw.max() - colors_raw.min())
colors = colors_raw.reshape(-1, 3)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
views = [('Side View', (20, 45)), ('Top View', (90, 0)), ('Front View', (0, 90))]
for ax, (title, (elev, azim)) in zip(axes, views):
    ax3d = fig.add_subplot(1, 3, axes.tolist().index(ax)+1, projection='3d')
    ax3d.scatter(pts[::5,0], pts[::5,1], pts[::5,2], c=colors[::5], s=0.3)
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_title(f'idx 3781 Pair - {title}')
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')

plt.suptitle('DUSt3R Point Cloud idx 3781 (Pair) - Co-planar Structure', fontsize=13)
plt.tight_layout()
plt.savefig('/home/jtan/CommE5052/results/3781_pointcloud_pair_multiview.png', dpi=150)
plt.close()
print('Saved multi-view point cloud.')
