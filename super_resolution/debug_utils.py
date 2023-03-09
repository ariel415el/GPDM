import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from matplotlib import pyplot as plt
from torchvision.transforms import Resize

from patch_swd import PatchSWDLoss
import torch.nn.functional as F
from utils import load_image


def get_projs(image, criteria):
    rand_proj = criteria.rand.to(image.device)
    projx = F.conv2d(image, rand_proj).transpose(1, 0).reshape(criteria.num_proj, -1)
    projx = projx[0].cpu().numpy()
    return projx


def dump_hists(refernce_image, low_res, output_images, path):
    nbins = 50
    patch_sizes = [5,16,64]
    H = len(patch_sizes)
    W = len(output_images)
    fig, axes = plt.subplots(nrows=H, ncols=W, figsize=(H*3, W*3))
    for r in range(H):
        criteria = PatchSWDLoss(patch_size=patch_sizes[r], stride=1, num_proj=1, c=refernce_image.shape[1])
        axes[r,0].set_ylabel(f"Patch-size {patch_sizes[r]}", fontsize=10)
        for c, (name, output_image) in enumerate(output_images.items()):
            # axes[r, c].hist(get_projs(refernce_image, criteria), label="Ref", bins=nbins, density=True, alpha=0.5)
            # axes[r, c].hist(get_projs(low_res, criteria), label="LowRes", bins=nbins, density=True, alpha=0.5)
            # axes[r, c].hist(get_projs(output_image, criteria), label="Opt", bins=nbins, density=True, alpha=0.5)
            axes[r, c].hist([get_projs(refernce_image, criteria), get_projs(low_res, criteria), get_projs(output_image, criteria)]
                            , label=["Ref", "LowRes", "Opt"], bins=nbins, density=True, alpha=0.5)
            if r == 0:
                axes[r, c].set_title(f"Optimizer: {name}", fontsize=10)
    plt.legend()

    # plt.tight_layout()
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)
    # fig.text(0.06, 0.5, 'Patch_size', ha='center', va='center', rotation='vertical')
    plt.savefig(path)
    plt.clf()


def plot_values(val_dict, path):
    for name, val_list in val_dict.items():
        plt.plot(range(len(val_list)), val_list, label=name)
    plt.legend()
    plt.savefig(path)
    plt.clf()


if __name__ == '__main__':
    device = torch.device("cuda")
    im_size = 512
    im1 = load_image('/cs/labs/yweiss/ariel1/data/afhq/train/wild/flickr_wild_000065.jpg').to(device)
    im2 = load_image('/cs/labs/yweiss/ariel1/data/afhq/train/wild/flickr_wild_000066.jpg').to(device)
    im1 = Resize(im_size, antialias=True)(im1)
    im2 = Resize(im_size, antialias=True)(im2)
    im1_low_res = Resize(im_size//4, antialias=True)(im1)

    # plot_projections_histogram_on_top({"im1": im1, "im2": im2, "im1_low_res": im1_low_res}, f"hists.png")
