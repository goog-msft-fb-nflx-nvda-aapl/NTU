import sys
sys.path.insert(0, './stable-diffusion')
sys.path.insert(0, '/home/jtan/ControlNet')

import os
import json
import cv2
import einops
import numpy as np
import torch
import argparse
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from PIL import Image


def process(model, ddim_sampler, control_image, prompt, ddim_steps=50, strength=1.0, scale=9.0, seed=42, eta=0.0, image_resolution=512):
    with torch.no_grad():
        # Prepare control image
        h, w = control_image.shape[:2]
        control_image = cv2.resize(control_image, (image_resolution, image_resolution))
        H, W = image_resolution, image_resolution

        control = torch.from_numpy(control_image.copy()).float().cuda() / 255.0
        control = einops.rearrange(control, 'h w c -> 1 c h w').clone()

        seed_everything(seed)

        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt])]
        }
        un_cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([""])]
        }
        shape = (4, H // 8, W // 8)

        model.control_scales = [strength] * 13

        samples, _ = ddim_sampler.sample(
            ddim_steps, 1, shape, cond,
            verbose=False, eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond
        )

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    return x_samples[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--config', type=str, default='/home/jtan/ControlNet/models/cldm_v15.yaml')
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--scale', type=float, default=9.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image_resolution', type=int, default=512)
    opt = parser.parse_args()

    os.makedirs(opt.output_dir, exist_ok=True)

    # Load model
    model = create_model(opt.config).cpu()
    model.load_state_dict(load_state_dict(opt.ckpt, location='cuda'))
    model = model.cuda()
    model.eval()
    ddim_sampler = DDIMSampler(model)

    # Load prompts
    data = []
    with open(opt.json_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    for item in data:
        source_path = os.path.join(opt.input_dir, item['source'])
        target_name = item['target']
        prompt = item['prompt']

        control_image = cv2.imread(source_path)
        control_image = cv2.cvtColor(control_image, cv2.COLOR_BGR2RGB)

        result = process(
            model, ddim_sampler, control_image, prompt,
            ddim_steps=opt.ddim_steps,
            strength=opt.strength,
            scale=opt.scale,
            seed=opt.seed,
            eta=0.0,
            image_resolution=opt.image_resolution
        )

        out_path = os.path.join(opt.output_dir, target_name)
        Image.fromarray(result).save(out_path)
        print(f"Saved {out_path}")

    print("Done.")