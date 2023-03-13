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
            for frequency in [factor, 0.1 * factor, 1 * factor, 10 * factor]:
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
    ker[-1, dim // 2] = 1
    kernels.append(ker)

    ker = np.zeros((dim, dim))
    ker[dim // 2, 0] = 1
    ker[dim // 2, -1] = 1
    kernels.append(ker)


    kernels = np.stack(kernels)
    kernels = torch.from_numpy(kernels)[:, None]
    kernels = kernels.repeat(1,3,1,1)
    return kernels


def get_random_projections(p, n=1):
    rand = torch.randn(n, 3 * p ** 2)  # (slice_size**2*ch)
    rand = rand / torch.norm(rand, dim=1, keepdim=True)  # noramlize to unit directions
    rand = rand.reshape(n, 3, p, p)
    return rand


def get_projs(image, rand):
    rand /= rand.sum()
    projx = torch.nn.functional.conv2d(image, rand.to(image.device).unsqueeze(0)).transpose(1, 0).reshape(-1)
    projx = projx.cpu().numpy()
    return projx


def plot(ax, series, labels, nbins=100):
    max = np.max([x.max() for x in series])
    min = np.min([x.min() for x in series])
    bins = np.linspace(min, max, nbins)
    # ax.hist(series, label=labels, bins=bins, density=True, alpha=0.5)
    for s, l in zip(series, labels):
        ax.hist(s, label=l, bins=bins, density=True, alpha=0.75)
    ax.legend()

def plot_img(axs, img, name):
    img -= img.mean()
    img /= img.max()
    # img *= 255
    axs.imshow(img)
    axs.set_title(f"{name}")
    axs.axis('off')

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from torchvision.transforms import Resize
    from utils import load_image

    device = torch.device("cpu")
    im_size = 512
    p=5

    hig_res = load_image('/cs/labs/yweiss/ariel1/data/afhq/train/wild/flickr_wild_000065.jpg').to(device)
    hig_res = Resize(im_size, antialias=True)(hig_res)
    low_res = Resize(im_size // 4, antialias=True)(hig_res)

    rand_projs = get_random_projections(p, n=50)
    kernels = get_fixed_kernels(p)
    big_rand_kernels = Resize(p*4, antialias=True)(rand_projs)
    big_kernels = Resize(p*4, antialias=True)(kernels)

    plots = [
                (rand_projs, big_rand_kernels, f"Random", 0),
                # (rand_projs, big_rand_kernels, f"random", 15),
                (kernels, big_kernels, f"Gabor", 15),
                # (kernels, big_kernels, f"gabor", 2)
             ]

    fig, axes = plt.subplots(nrows=len(plots), ncols=4, figsize=(10, 5))
    for i, (k, bg, name, idx) in enumerate(plots):
        plot(axes[i, 0], [get_projs(hig_res, k[idx]), get_projs(low_res, k[idx])], ["HigRes", "LowRes"], nbins=100)
        axes[i, 0].set_ylabel(name)
        plot(axes[i, 1], [get_projs(hig_res, bg[idx]), get_projs(low_res, k[idx])], ["HigRes_upscaled", "LowRes"], nbins=100)

        plot_img(axes[i, 2], k[idx].permute(1, 2, 0).numpy(), f"{name}")
        plot_img(axes[i, 3], bg[idx].permute(1, 2, 0).numpy(), f"Upscaled {name}")

    plt.tight_layout()
    plt.show()
    # kernels = get_fixed_kernels(9)
    # print(len(kernels))
    # for ker in kernels:
    #     plt.imshow(ker.permute(1, 2, 0).numpy())
    #     plt.show()