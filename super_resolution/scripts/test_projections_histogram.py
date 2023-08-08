import os
import sys
import torch
from matplotlib import pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sr_utils.debug_utils import plot_hists, plot_img
from sr_utils.predefined_filters import get_random_filters, get_gabor_filters, appply_filter, resize_filters, normalize_filters


def split_list_randomly(values):
    n = len(values)
    shuffled_values = values[torch.randperm(n)]
    l1 = shuffled_values[:n//2]
    l2 = shuffled_values[n//2:]
    return l1,l2


def compare_resized_histograms(factor, p, reverse=True):
    low_res = Resize(int(im_size // factor), antialias=True)(hig_res)

    if reverse:
        p = factor * p
        factor = 1/factor

    rand_projs = normalize_filters(get_random_filters(p, n=50))
    # naive_projs = get_naive_kernels(p)
    gabor_projs = normalize_filters(get_gabor_filters(p))


    resized_rand_projs = resize_filters(rand_projs, p, factor, normalize=True)
    # resized_naive_projs = resize_filters(naive_projs, p, factor, False)
    resized_gabor_projs = resize_filters(gabor_projs, p, factor, normalize=True)

    filters = [
                (rand_projs[0], resized_rand_projs[0], f"Random"),
                (rand_projs[2], resized_rand_projs[2], f"Random"),
                # (naive_projs, resized_naive_projs, f"Naive", 1),
                (gabor_projs[7], resized_gabor_projs[7], f"Gabor1"),
                (gabor_projs[15], resized_gabor_projs[15], f"Gabor2"),
             ]
    H = len(filters)
    W = 4
    S = 2

    fig, axes = plt.subplots(nrows=H, ncols=W, figsize=(W*S, H*S))
    for i, (filter, resized_filter, name) in enumerate(filters):
        plot_hists(axes[i, 0], [appply_filter(hig_res, filter), appply_filter(low_res, filter)], ["HigRes", "LowRes"], nbins=nbins)
        axes[i, 0].set_ylabel(name)

        if reverse:
            plot_hists(axes[i, 1], [appply_filter(hig_res, filter), appply_filter(low_res, resized_filter)], ["HigRes", f"LowRes_x{factor:.2f}"], nbins=nbins)
        else:
            plot_hists(axes[i, 1], [appply_filter(hig_res, resized_filter), appply_filter(low_res, filter)], [f"HigRes_x{factor:.2f}", "LowRes"], nbins=nbins)

        plot_img(axes[i, 2], filter, f"{name}\n{tuple(filter.shape[-2:])}")
        plot_img(axes[i, 3], resized_filter, f"x{factor:.2f} {name}\n{tuple(resized_filter.shape[-2:])}")
        # plot_img(axes[i, 4], hig_res[0].permute(1, 2, 0).numpy().copy(), f"HighRes")

    fig.suptitle("Downscaled" if reverse else "Upscaled")
    plt.tight_layout()
    plt.show()


def hist_sanity(factor, p):
    low_res = Resize(int(im_size // factor), antialias=True)(hig_res)
    low_res_big = Resize(im_size, antialias=True)(low_res)

    filters = [
        (get_random_filters(p)[0], "Random"),
         (get_gabor_filters(p)[7], "Gabor")
    ]
    H = len(filters)
    W=6
    S=2
    fig, axes = plt.subplots(nrows=H, ncols=W, figsize=(W*S, H*S))
    for i, (filter, name) in enumerate(filters):

        high_res_projections = appply_filter(hig_res, filter)
        low_res_projections = appply_filter(low_res, filter)
        low_res_big_projections = appply_filter(low_res_big, filter)

        plot_hists(axes[i, 0], [high_res_projections, low_res_projections], ["HigRes", "LowRes"], nbins=nbins)
        axes[i, 0].set_ylabel(name)

        plot_hists(axes[i, 1], [high_res_projections, low_res_big_projections], ["HigRes", "LowRes_big"], nbins=nbins)

        for j, (val_list, name) in enumerate([(high_res_projections, "HigRes"),
                                              (low_res_projections, "LowRes"),
                                              (low_res_big_projections, "LowRes_big")]):
            projs1, projs2 = split_list_randomly(val_list)
            plot_hists(axes[i, 2+j], [projs1, projs2], [name, name], nbins=nbins)


        plot_img(axes[i, 3+j], filter, f"{name}\n{tuple(filter.shape[-2:])}")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from torchvision.transforms import Resize
    from utils import load_image

    device = torch.device("cpu")
    im_size = 1024
    nbins = 100
    hig_res = load_image('data/images/SR/fox2.jpg').to(device)
    hig_res = Resize(im_size, antialias=True)(hig_res)

    factor = 2
    p=2
    hist_sanity(factor, p)
    compare_resized_histograms(factor, p, reverse=True)
    compare_resized_histograms(factor, p, reverse=False)