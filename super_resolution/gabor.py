import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2



def gabor(sigma, theta, Lambda, psi, gamma):
    """Gabor feature extraction."""
    sigma_x = float(sigma) / gamma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb


def get_fixed_kernels(dim):
    kernels = []
    factor = dim/5
    sigma = 1 * factor
    for theta in range(8):
        theta = theta / 8. * np.pi
        for ar in [0.5]:
            for frequency in [2*factor, 10 * factor, 20 * factor]:
                kernel_cv = cv2.getGaborKernel((dim,dim), sigma, theta, frequency, ar, 0, ktype=cv2.CV_32F)
                # kernel_cv = gabor(sigma, theta, 1/frequency, 0, ar)
                # kernel_cv = (kernel_cv - kernel_cv.min()) / (kernel_cv.max() - kernel_cv.min())
                # kernel_cv /= kernel_cv.sum()
                kernels.append(kernel_cv)
                # plt.imshow(kernel_cv)
                # plt.title(f"{theta}, {ar}, {frequency}")
                # plt.show()
    kernels = np.stack(kernels)
    kernels = torch.from_numpy(kernels)[:, None]
    kernels = kernels.repeat(1,3,1,1)
    return kernels


def get_naive_kernels(dim):
    kernels = []

    ker = np.zeros((dim, dim))
    ker[0, dim // 2] = 1
    ker[-1, dim // 2] = -1
    kernels.append(ker)

    ker = np.zeros((dim, dim))
    ker[dim // 2, 0] = 1
    ker[dim // 2, -1] = -1
    kernels.append(ker)


    kernels = np.stack(kernels)
    kernels = torch.from_numpy(kernels)[:, None]
    kernels = kernels.repeat(1,3,1,1).float()
    return kernels


def get_random_projections(p, n=1):
    rand = torch.randn(n, 3 * p ** 2)  # (slice_size**2*ch)
    rand = rand / torch.norm(rand, dim=1, keepdim=True)  # noramlize to unit directions
    rand = rand.reshape(n, 3, p, p)
    return rand


def get_projs(image, rand):
    projx = torch.nn.functional.conv2d(image, rand.to(image.device).unsqueeze(0)).transpose(1, 0).reshape(-1)
    projx = projx.cpu().numpy()
    return projx


def plot_hists(ax, series, labels, nbins=100):
    max = np.max([x.max() for x in series])
    min = np.min([x.min() for x in series])
    width = (max-min) / nbins
    bins = np.linspace(min-width/2, max+width/2, nbins)
    # ax.hist(series, label=labels, bins=bins, density=True, alpha=0.5)
    for s, l in zip(series, labels):
        ax.hist(s, label=l, bins=bins, density=True, alpha=0.75)
    ax.legend()

def plot_img(axs, img, name):
    img -= img.min()
    img /= img.max()
    # img *= 255
    axs.imshow(img)
    axs.set_title(f"{name}")
    axs.axis('off')


def normalize_projections(projs):
    for i in range(len(projs)):
        if projs[i].sum() != 0:
            projs[i] /= projs[i].sum()
    return projs
def resize_projections(projections, p, factor, normalize=True):
    resized_projections = Resize(int(p * factor), antialias=True)(projections)
    if normalize:
        resized_projections = normalize_projections(resized_projections)
        projections = normalize_projections(projections)
    return projections, resized_projections

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from torchvision.transforms import Resize
    from utils import load_image

    device = torch.device("cpu")
    im_size = 1024
    p=5
    nbins=100
    factor = 4

    hig_res = load_image('../data/images/SR/fox2.jpg').to(device)
    hig_res = Resize(im_size, antialias=True)(hig_res)
    low_res = Resize(int(im_size // 4), antialias=True)(hig_res)

    rand_projs = get_random_projections(p, n=50)
    naive_projs = get_naive_kernels(p)
    gabor_projs = get_fixed_kernels(p)
    rand_projs, resized_rand_projs = resize_projections(rand_projs, p, factor)
    naive_projs, resized_naive_projs = resize_projections(naive_projs, p, factor, False)
    gabor_projs, resized_gabor_projs = resize_projections(gabor_projs, p, factor)

    plots = [
                (rand_projs, resized_rand_projs, f"Random", 0),
                (rand_projs, resized_rand_projs, f"Random", 2),
                # (naive_projs, resized_naive_projs, f"Naive", 1),
                (gabor_projs, resized_gabor_projs, f"Gabor1", 7),
                (gabor_projs, resized_gabor_projs, f"Gabor2", 15),
             ]

    fig, axes = plt.subplots(nrows=len(plots), ncols=4, figsize=(10, 5))
    for i, (k, bk, name, idx) in enumerate(plots):
        plot_hists(axes[i, 0], [get_projs(hig_res, k[idx]), get_projs(low_res, k[idx])], ["HigRes", "LowRes"], nbins=nbins)
        axes[i, 0].set_ylabel(name)
        plot_hists(axes[i, 1], [get_projs(hig_res, bk[idx]), get_projs(low_res, k[idx])], [f"HigRes_x{factor:.2f}", "LowRes"], nbins=nbins)
        # plot_hists(axes[i, 1], [get_projs(hig_res, k[idx]), get_projs(low_res, bk[idx])], ["HigRes", f"LowRes_x{factor:.2f}"], nbins=nbins)

        plot_img(axes[i, 2], k[idx].permute(1, 2, 0).numpy().copy(), f"{name}\n{tuple(k[idx].shape[-2:])}")
        plot_img(axes[i, 3], bk[idx].permute(1, 2, 0).numpy().copy(), f"x{factor:.2f} {name}\n{tuple(bk[idx].shape[-2:])}")
        # plot_img(axes[i, 4], hig_res[0].permute(1, 2, 0).numpy().copy(), f"HighRes")

    plt.tight_layout()
    plt.show()
    # kernels = get_fixed_kernels(9)
    # print(len(kernels))
    # for ker in kernels:
    #     plt.imshow(ker.permute(1, 2, 0).numpy())
    #     plt.show()