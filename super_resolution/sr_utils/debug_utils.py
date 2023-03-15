import os
import sys

import numpy as np


import torch
from matplotlib import pyplot as plt
from torchvision.transforms import Resize

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from super_resolution.sr_utils.predefined_filters import appply_filter, get_random_filters, get_gabor_filters

from utils import load_image


def plot_hists(ax, series, labels, nbins=100):
    max = np.max([x.max() for x in series])
    min = np.min([x.min() for x in series])
    width = (max-min) / nbins
    bins = np.linspace(min-width/2, max+width/2, nbins)
    # ax.hist(series, label=labels, bins=bins, density=True, alpha=0.5)
    for s, l in zip(series, labels):
        ax.hist(s, label=l, bins=bins, density=True, alpha=0.75)
    ax.set_title(f"HistSWD: {compute_swd(*series):.4f}")
    ax.legend()

def plot_img(axs, img, name):
    np_img = img.permute(1, 2, 0).numpy().copy()
    np_img -= np_img.min()
    np_img /= np_img.max()
    # img *= 255
    im = axs.imshow(np_img)
    axs.set_title(f"{name}")
    axs.axis('off')

def compute_swd(projx, projy):
    projx_pt = torch.from_numpy(projx).unsqueeze(0)
    projy_pt = torch.from_numpy(projy).unsqueeze(0)
    from patch_swd import duplicate_to_match_lengths
    projx_pt, projy_pt = duplicate_to_match_lengths(projx_pt, projy_pt)
    # Sort and compute L1 loss
    projx_pt, _ = torch.sort(projx_pt, dim=1)
    projy_pt, _ = torch.sort(projy_pt, dim=1)
    loss = np.abs(projx_pt - projy_pt).mean()
    return loss.item()


def dump_hists(refernce_image, low_res, output_images, path):
    p = 7
    nbins=100
    filters = [
         (get_random_filters(p)[0], "Random"),
         # (get_random_filters(p)[0], "Random"),
         (get_gabor_filters(p)[7], "Gabor"),
         # (get_gabor_filters(p)[15], "Gabor")
    ]

    H = len(filters)
    W = 2 + len(output_images)
    S = 3

    fig, axes = plt.subplots(nrows=H, ncols=W, figsize=(W*S, H*S))
    for i, (filter, name) in enumerate(filters):
        highres_projs = appply_filter(refernce_image, filter)
        lowres_projs = appply_filter(low_res, filter)
        plot_hists(axes[i, 0], [highres_projs, lowres_projs], ["HigRes", "LowRes"], nbins=nbins)
        axes[i, 0].set_ylabel(name)
        for j, (opt_name,opt_img) in enumerate(output_images.items()):
            opt_projs = appply_filter(opt_img, filter)
            plot_hists(axes[i, 1+j], [highres_projs, opt_projs], ["HigRes", opt_name], nbins=nbins)

        plot_img(axes[i, -1], filter, f"{name}\n{tuple(filter.shape[-2:])}")

    plt.tight_layout()
    plt.savefig(path)
    plt.clf()


def plot_values(val_dict, path):
    for name, val_list in val_dict.items():
        plt.plot(range(len(val_list)), val_list, label=name)
    plt.legend()
    plt.savefig(path)
    plt.clf()


def sanity():
    device = torch.device("cpu")
    im_size = 512
    im1 = load_image('/cs/labs/yweiss/ariel1/data/afhq/train/wild/flickr_wild_000065.jpg').to(device)
    im2 = load_image('/cs/labs/yweiss/ariel1/data/afhq/train/wild/flickr_wild_000066.jpg').to(device)
    im1 = Resize(im_size, antialias=True)(im1)
    im2 = Resize(im_size, antialias=True)(im2)
    im1_low_res = Resize(im_size//4, antialias=True)(im1)

    dump_hists(im1, im1_low_res, {'im2': im2, "im1":im1}, "asd.png")

