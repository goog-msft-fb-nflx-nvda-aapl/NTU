import os, sys, torch
from argparse import ArgumentParser
from os import makedirs
import torchvision

sys.path.insert(0, os.path.expanduser("~/vfx-final-project/gaussian-splatting"))

from gaussian_renderer import GaussianModel, render
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text
from scene.dataset_readers import readColmapCameras
from utils.camera_utils import cameraList_from_camInfos
from arguments import ModelParams, PipelineParams

def render_test_views(model_path, test_sparse_dir, images_dir, output_dir, iteration=3000):
    parser = ArgumentParser()
    mp = ModelParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([])

    gaussians = GaussianModel(3)
    gaussians.load_ply(os.path.join(model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply"))

    cam_extrinsics = read_extrinsics_text(os.path.join(test_sparse_dir, "images.txt"))
    cam_intrinsics = read_intrinsics_text(os.path.join(test_sparse_dir, "cameras.txt"))

    cam_infos = readColmapCameras(
        cam_extrinsics, cam_intrinsics,
        depths_params=None,
        images_folder=images_dir,
        depths_folder="",
        test_cam_names_list=[]
    )
    cam_infos = sorted(cam_infos, key=lambda x: x.image_name)

    cameras = cameraList_from_camInfos(cam_infos, resolution_scale=1.0, args=args,
                                        is_nerf_synthetic=False, is_test_dataset=True)

    makedirs(output_dir, exist_ok=True)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    pipeline = pp.extract(args)

    with torch.no_grad():
        for idx, cam in enumerate(cameras):
            rendering = render(cam, gaussians, pipeline, bg)["render"]
            name = cam_infos[idx].image_name
            out_path = os.path.join(output_dir, name if name.endswith(".png") else name + ".png")
            torchvision.utils.save_image(rendering, out_path)
            print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_sparse_dir", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--iteration", type=int, default=3000)
    args = parser.parse_args()
    render_test_views(args.model_path, args.test_sparse_dir, args.images_dir, args.output_dir, args.iteration)
