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
all_pngs = sorted(glob.glob(os.path.join(interp_dir, '*.png')),
                  key=lambda p: int(os.path.basename(p).split('frame')[-1].split('.')[0]))
video_paths = [img1] + all_pngs[1:-1] + [img2]

images_video = load_images(video_paths, size=480)
pairs_video = make_pairs(images_video, scene_graph='complete', symmetrize=True)
output_video = inference(pairs_video, model, device, batch_size=1)
scene_video = global_aligner(output_video, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
try:
    _ = scene_video.compute_global_alignment(init="mst", niter=100, schedule='cosine', lr=0.01)
except Exception as e:
    print(f"Warning: {e}")

pts3d_video = scene_video.get_pts3d()
pts_v = pts3d_video[0].detach().cpu().numpy().reshape(-1, 3)
colors_raw_v = images_video[0]['img'].squeeze().permute(1,2,0).cpu().numpy()
colors_raw_v = (colors_raw_v - colors_raw_v.min()) / (colors_raw_v.max() - colors_raw_v.min())
colors_v = colors_raw_v.reshape(-1, 3)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts_v[::10, 0], pts_v[::10, 1], pts_v[::10, 2], c=colors_v[::10], s=0.5)
ax.set_title('DUSt3R Point Cloud - idx 3781 (Video Input)')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
plt.tight_layout()
plt.savefig('/home/jtan/CommE5052/results/3781_pointcloud_video.png', dpi=150)
plt.close()
print('Saved point cloud for video.')
