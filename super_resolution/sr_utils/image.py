import numpy as np
import torch
from torch.nn import functional as F


def extract_patches(src_img, patch_size, stride):
    """
    Splits the image to overlapping patches and returns a pytorch tensor of size (N_patches, 3*patch_size**2)
    """
    channels = src_img.shape[1]
    patches = F.unfold(src_img, kernel_size=patch_size, stride=stride) # shape (b, 3*p*p, N_patches)
    patches = patches.squeeze(dim=0).permute((1, 0)).reshape(-1, channels,  patch_size, patch_size)
    return patches


def get_gaussian_kernel(device="cpu"):
    kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]], np.float32) / 256.0
    gaussian_k = torch.as_tensor(kernel.reshape(1, 1, 5, 5)).to(device)
    return gaussian_k


def pyramid_down(image):
    gaussian_k = get_gaussian_kernel(device=image.device)
    # channel-wise conv(important)
    multiband = [F.conv2d(image[:, i:i + 1, :, :], gaussian_k, padding=2, stride=2) for i in range(image.shape[1])]
    down_image = torch.cat(multiband, dim=1)
    return down_image


def pyramid_up(image):
    gaussian_k = get_gaussian_kernel(device=image.device)
    upsample = F.interpolate(image, scale_factor=2)
    multiband = [F.conv2d(upsample[:, i:i + 1, :, :], gaussian_k, padding=2) for i in range(image.shape[1])]
    up_image = torch.cat(multiband, dim=1)
    return up_image


def gaussian_pyramid(original, n_pyramids):
    x = original
    # pyramid down
    pyramids = [original]
    for i in range(n_pyramids):
        x = pyramid_down(x)
        pyramids.append(x)
    return pyramids


def laplacian_pyramid(original, n_pyramids):
    # create gaussian pyramid
    pyramids = gaussian_pyramid(original, n_pyramids)

    # pyramid up - diff
    laplacian = []
    for i in range(len(pyramids) - 1):
        diff = pyramids[i] - pyramid_up(pyramids[i + 1])
        laplacian.append(diff)
    # Add last gaussian pyramid
    laplacian.append(pyramids[len(pyramids) - 1])
    return laplacian
