import os
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from torchvision import transforms
import torch
import torch.nn.functional as F


def get_kernel_gauss(size=5, sigma=1.0, n_channels=1):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")

    grid = np.float32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    return kernel


def conv_gauss(img, kernel):
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)


def cv2pt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float64) / 255.
    img = img * 2 - 1
    # img *= 2
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()

    return img

def aspect_ratio_resize(img, max_dim=256):
    y, x, c = img.shape
    if x > y:
        return cv2.resize(img, (max_dim, int(y/x*max_dim)))
    else:
        return cv2.resize(img, (int(x/y*max_dim), max_dim))


def downscale(img, pyr_factor):
    assert 0 < pyr_factor < 1
    # y, x, c = img.shape
    c, y, x = img.shape
    new_x = int(x * pyr_factor)
    new_y = int(y * pyr_factor)

    # return cv2.resize(img, (new_x, new_y), interpolation=cv2.INTER_AREA)
    return transforms.Resize((new_y, new_x), antialias=True)(img)


def get_pyramid(img, n_levels, pyr_factor):
    res = [img]
    for i in range(n_levels):
        img = downscale(img, pyr_factor)
        res = [img] + res
    return res


def quantize_image(img, N_colors):
    return np.round_(img*(N_colors/255))*(255/N_colors)


def get_file_name(path):
    return os.path.splitext(os.path.basename(path))[0]


@dataclass
class SyntesisConfigurations:
    aspect_ratio: Tuple[float, float] = (1.,1.)
    resize: int = 256
    pyr_factor: float = 0.7
    n_scales: int = 5
    lr: float = 0.05
    num_steps: int = 500
    init: str = 'noise'
    blur_loss: float = 0.0
    tv_loss: float = 0.0
    device: str = 'cuda:0'

    def get_conf_tag(self):
        init_name = 'img' if os.path.exists(self.init) else self.init
        if self.blur_loss > 0:
            init_name += f"_BL({self.blur_loss})"
        if self.tv_loss > 0:
            init_name += f"_TV({self.tv_loss})"
        return f'AR-{self.aspect_ratio}_R-{self.resize}_S-{self.pyr_factor}x{self.n_scales}_I-{init_name}'