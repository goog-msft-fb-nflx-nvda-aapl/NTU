import os, sys, torch, lpips, math, argparse
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

def psnr(p,g):
    mse=F.mse_loss(p,g).item()
    return 20*math.log10(1.0/math.sqrt(mse)) if mse>0 else 100.

def ssim(pred,gt):
    C1,C2=0.01**2,0.03**2
    mu1=F.avg_pool2d(pred,11,1,5); mu2=F.avg_pool2d(gt,11,1,5)
    mu1_sq,mu2_sq,mu12=mu1**2,mu2**2,mu1*mu2
    s1=F.avg_pool2d(pred*pred,11,1,5)-mu1_sq
    s2=F.avg_pool2d(gt*gt,11,1,5)-mu2_sq
    s12=F.avg_pool2d(pred*gt,11,1,5)-mu12
    return ((2*mu12+C1)*(2*s12+C2)/((mu1_sq+mu2_sq+C1)*(s1+s2+C2))).mean().item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    args = parser.parse_args()

    loss_fn = lpips.LPIPS(net='vgg')
    to_t = T.ToTensor()
    imgs = sorted([f for f in os.listdir(args.render_dir) if f.endswith('.png')])

    psnrs, ssims, lpipss = [], [], []
    for name in imgs:
        rpath = os.path.join(args.render_dir, name)
        gpath = os.path.join(args.gt_dir, name)
        if not os.path.exists(gpath):
            print(f"GT not found for {name}, skipping")
            continue
        p = to_t(Image.open(rpath)).unsqueeze(0)
        g = to_t(Image.open(gpath)).unsqueeze(0)
        if p.shape != g.shape:
            p = F.interpolate(p, size=g.shape[-2:], mode='bilinear', align_corners=False)
        with torch.no_grad():
            lp = loss_fn(p*2-1, g*2-1).item()
        psnrs.append(psnr(p,g)); ssims.append(ssim(p,g)); lpipss.append(lp)
        print(f"{name}: PSNR={psnrs[-1]:.3f} SSIM={ssims[-1]:.4f} LPIPS={lpipss[-1]:.4f}")

    if psnrs:
        print(f"\nMean: PSNR={sum(psnrs)/len(psnrs):.3f} SSIM={sum(ssims)/len(ssims):.4f} LPIPS={sum(lpipss)/len(lpipss):.4f}")
