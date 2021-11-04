import os
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
import torch.nn.functional as F
import torchvision.utils as vutils


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
    return F.conv2d(img, kernel.to(img.device), groups=n_channels)


def cv2pt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float64) / 255.
    img = img * 2 - 1
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()

    return img

def aspect_ratio_resize(img, max_dim=256):
    y, x, c = img.shape
    if x > y:
        return cv2.resize(img, (max_dim, int(y/x*max_dim)))
    else:
        return cv2.resize(img, (int(x/y*max_dim), max_dim))


def match_image_sizes(input, target):
    """resize and crop input image sot that it has the same aspect ratio as target"""
    assert(len(input.shape) == len(target.shape) and len(target.shape) == 4)
    input_h, input_w = input.shape[-2:]
    target_h, target_w = target.shape[-2:]
    input_scale_factor = input_h / input_w
    target_scale_factor = target_h / target_w
    if target_scale_factor > input_scale_factor:
        input = transforms.Resize((target_h, int(input_w/input_h*target_h)), antialias=True)(input)
        pixels_to_cut = input.shape[-1] - target_w
        if pixels_to_cut > 0:
            input = input[:, :, :, int(pixels_to_cut / 2):-int(pixels_to_cut / 2)]

    else:
        input = transforms.Resize((int(input_h/input_w*target_w), target_w), antialias=True)(input)
        pixels_to_cut = input.shape[-2] - target_h
        if pixels_to_cut > 0:
            input = input[:, :, int(pixels_to_cut / 2):-int(pixels_to_cut / 2)]

    input = transforms.Resize(target.shape[-2:], antialias=True)(input)

    return input


def downscale(img, pyr_factor):
    assert 0 < pyr_factor < 1
    # y, x, c = img.shape
    c, y, x = img.shape
    new_x = int(x * pyr_factor)
    new_y = int(y * pyr_factor)

    # return cv2.resize(img, (new_x, new_y), interpolation=cv2.INTER_AREA)
    return transforms.Resize((new_y, new_x), antialias=True)(img)


# def get_pyramid(img, n_levels, pyr_factor):
#     res = [img]
#     for i in range(n_levels):
#         img = downscale(img, pyr_factor)
#         res = [img] + res
#     return res

def get_pyramid(img, min_size, pyr_factor):
    res = [img]
    while True:
        img = downscale(img, pyr_factor)
        if img.shape[-2] < min_size:
            break
        res = [img] + res
    return res

def quantize_image(img, N_colors):
    return np.round_(img*(N_colors/255))*(255/N_colors)


def get_file_name(path):
    return os.path.splitext(os.path.basename(path))[0]

def _fspecial_gauss_1d(size, sigma):
    """Create 2-D gauss kernel"""
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    w = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    w /= w.sum()

    w = (w.reshape(size, 1) * w.reshape(1, size))

    return w


class LossesList(torch.nn.Module):
    def __init__(self, losses, weights, name=None):
        super(LossesList, self).__init__()
        self.weights = weights

        self.losses = torch.nn.ModuleList(losses)

        self.name = name if name else "+".join([f"{w}*{l.name}" for l,w in zip(self.losses, self.weights)])

    def forward(self, x, y):
        return sum([self.losses[i](x, y) * self.weights[i] for i in range(len(self.losses))])


class GrayLevelLoss(torch.nn.Module):
    def __init__(self, img, resize):
        super(GrayLevelLoss, self).__init__()
        self.img = None
        self.resize = resize
        self.img = transforms.Resize((resize, resize), antialias=True)(img)
        self.name = f'GrayLevelLoss({resize})'
    def forward(self, x, y):
        if self.img is None:
            raise ValueError("Uninitialized with image")
        from torchvision import transforms
        img = transforms.Resize((x.shape[-2], x.shape[-1]), antialias=True)(self.img.to(x.device))
        # return ((img.mean(0) - x[0].mean(0))**2).mean()
        return ((img - x[0])**2).mean()


def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vutils.save_image(torch.clip(img, -1, 1), path, normalize=True)


def plot_loss(losses, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    # ax.plot(np.arange(len(losses)), losses)
    ax.plot(np.arange(len(losses)), np.log(losses))
    fig1.savefig(path)
    plt.close(fig1)