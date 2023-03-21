import os
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import Resize


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils import load_image
from patch_swd import PatchSWDLoss
from super_resolution.sr_utils.metrics import calculate_psnr, calculate_ssim
from super_resolution.sr_utils.image import gaussian_pyramid, laplacian_pyramid


def swd(x, y):
    n=6
    lap_pyrx = laplacian_pyramid(x, n)
    lap_pyry = laplacian_pyramid(y, n)
    pyrx = gaussian_pyramid(x, n)
    pyry = gaussian_pyramid(y, n)
    losses = []
    for i in range(n+1):
        # losses.append(PatchSWDLoss(num_proj=128)(x_, y_).item())
        losses.append(f"{pyrx[i].shape[-1]}: {PatchSWDLoss(num_proj=64, c=x.shape[1])(pyrx[i], pyry[i]).item():.5f}, "
                      f"LAP: {PatchSWDLoss(num_proj=64, c=x.shape[1])(lap_pyrx[i], lap_pyry[i]).item():.5f}")
    # return np.mean(losses)
    return "\n".join(losses)

def to_numpy(img):
    return img[0].permute(1,2,0).detach().cpu().numpy()

if __name__ == '__main__':
    gray=False
    device = torch.device("cuda:0")
    with torch.no_grad():
        s = 3
        bbox_dict = defaultdict(lambda : ((0.5, 0.5, 0.125), (0.5, 0.125, 0.125), (0.125, 0.5, 0.125)))
        bbox_dict["fox"] = [(0.5, 0.5, 0.125), (0.5, 0.125, 0.125), (0.125, 0.5, 0.125)]
        bbox_dict["00130"] = [(0.5, 0.5, 0.125), (0.5, 0.7, 0.125), (0.75, 0.5, 0.125)]

        dirpath = "outputs"
        for dirname in os.listdir("outputs"):
            img_dir = os.path.join(dirpath, dirname)

            img_name = dirname.split("_")[0]
            bboxes = bbox_dict[img_name]

            gt_img_pt = load_image(os.path.join(img_dir, "1-GT.png"), gray).to(device)
            gt_img_np = np.array(Image.open(os.path.join(img_dir, "1-GT.png")))
            # gt_img_np = to_numpy(gt_img_pt)

            paths = [os.path.join(img_dir, "1-GT.png"), os.path.join(img_dir, "2-initial_guess.png")]
            paths += sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.startswith("3")])
            paths = sorted(paths, key=lambda x: 1 if "gan" in x.lower() else 0)

            H = len(bboxes)
            W = len(paths)
            fig, axes = plt.subplots(nrows=H+1, ncols=W, figsize=(W*s, H*s))

            for i, path in enumerate(paths):
                print(path)
                img_pt = load_image(path, gray).to(device)
                dim = img_pt.shape[-2]
                img_np = np.array(Image.open(path))
                for j, (x,y,d) in enumerate(bboxes):
                    x,y,d = int(x*dim), int(y*dim), int(d * dim)
                    # if i == 0:
                    #     img_with_rects = cv2.rectangle(img, (x,y), (x+d, y+d), (255,0,0), s)
                    # else:
                    #     img_with_rects = img
                    axes[j+1, i].imshow(img_np[y:y+d, x:x+d])
                    axes[j+1, i].axis('off')
                axes[0, i].imshow(img_np)
                axes[0, i].axis('off')
                name=os.path.splitext(os.path.basename(path))[0]
                if name.startswith("3-"):
                    name = name[2:]
                if name == "2-initial_guess":
                    name = "bilinear"
                if name == "1-GT":
                    name = "GT"
                name += f"\nPSNR: {calculate_psnr(gt_img_np, img_np):.2f} " \
                        f"\nSSIM: {calculate_ssim(gt_img_np, img_np): .2f} " \
                        f"\nPyramid-SWD:\n {swd(img_pt, gt_img_pt)}"
                axes[0, i].set_title(name, fontsize=4*s)

            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, "5-comparison.png"))

# from basicsr import calculate_ssim, calculate_psnr