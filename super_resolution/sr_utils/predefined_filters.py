import numpy as np
import torch
import cv2
from torchvision.transforms import Resize


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


def get_gabor_filters(dim):
    filters = []
    factor = dim/5
    sigma = 1 * factor
    for theta in range(8):
        theta = theta / 8. * np.pi
        for ar in [0.5]:
            for frequency in [2*factor, 10 * factor, 20 * factor]:
                gabor_filter = cv2.getGaborKernel((dim,dim), sigma, theta, frequency, ar, 0, ktype=cv2.CV_32F)
                filters.append(gabor_filter)
                # plt.imshow(kernel_cv)
                # plt.title(f"{theta}, {ar}, {frequency}")
                # plt.show()
    filters = np.stack(filters)
    filters = torch.from_numpy(filters)[:, None]
    filters = filters.repeat(1,3,1,1)
    return normalize_filters(filters)


def get_naive_kernels(dim):
    filters = []

    ker = np.zeros((dim, dim))
    ker[0, dim // 2] = 1
    ker[-1, dim // 2] = -1
    filters.append(ker)

    ker = np.zeros((dim, dim))
    ker[dim // 2, 0] = 1
    ker[dim // 2, -1] = -1
    filters.append(ker)


    filters = np.stack(filters)
    filters = torch.from_numpy(filters)[:, None]
    filters = filters.repeat(1,3,1,1).float()
    return filters


def get_random_filters(p, n=1):
    rand = torch.randn(n, 3 * p ** 2)  # (slice_size**2*ch)
    rand = rand / torch.norm(rand, dim=1, keepdim=True)  # noramlize to unit directions
    rand = rand.reshape(n, 3, p, p)
    rand /= torch.sum(rand, dim=(1,2,3), keepdim=True)
    return rand


def normalize_filters(projs):
    for i in range(len(projs)):
        if projs[i].sum() != 0:
            projs[i] /= projs[i].sum()
    return projs


def resize_filters(filters, p, factor, normalize=True):
    resized_filters = Resize(int(p * factor), antialias=True)(filters)
    if normalize:
        resized_filters = normalize_filters(resized_filters)
        # filters = normalize_filters(filters)
    return resized_filters


def appply_filter(image, rand):
    projx = torch.nn.functional.conv2d(image, rand.to(image.device).unsqueeze(0)).transpose(1, 0).reshape(-1)
    projx = projx.cpu().numpy()
    return projx

