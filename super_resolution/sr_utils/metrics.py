from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1, img2):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    max_value = 255
    mse = ((img1.astype(np.float32)  - img2.astype(np.float32)) ** 2).mean()
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def calculate_ssim(img1, img2):
    return np.sum([ssim(img1[...,i], img2[...,i]) for i in range(img1.shape[2])])


# Gaussian blur kernel
def get_gaussian_kernel(device="cpu"):
    kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]], np.float32) / 256.0
    gaussian_k = torch.as_tensor(kernel.reshape(1, 1, 5, 5)).to(device)
    return gaussian_k


def pyramid_down(image, device="cpu"):
    gaussian_k = get_gaussian_kernel(device=device)
    # channel-wise conv(important)
    multiband = [F.conv2d(image[:, i:i + 1, :, :], gaussian_k, padding=2, stride=2) for i in range(3)]
    down_image = torch.cat(multiband, dim=1)
    return down_image


def pyramid_up(image, device="cpu"):
    gaussian_k = get_gaussian_kernel(device=device)
    upsample = F.interpolate(image, scale_factor=2)
    multiband = [F.conv2d(upsample[:, i:i + 1, :, :], gaussian_k, padding=2) for i in range(3)]
    up_image = torch.cat(multiband, dim=1)
    return up_image


def gaussian_pyramid(original, n_pyramids, device="cpu"):
    x = original
    # pyramid down
    pyramids = [original]
    for i in range(n_pyramids):
        x = pyramid_down(x, device=device)
        pyramids.append(x)
    return pyramids


def laplacian_pyramid(original, n_pyramids, device="cpu"):
    # create gaussian pyramid
    pyramids = gaussian_pyramid(original, n_pyramids, device=device)

    # pyramid up - diff
    laplacian = []
    for i in range(len(pyramids) - 1):
        diff = pyramids[i] - pyramid_up(pyramids[i + 1], device=device)
        laplacian.append(diff)
    # Add last gaussian pyramid
    laplacian.append(pyramids[len(pyramids) - 1])
    return laplacian


def minibatch_laplacian_pyramid(image, n_pyramids, batch_size, device="cpu"):
    n = image.size(0) // batch_size + np.sign(image.size(0) % batch_size)
    pyramids = []
    for i in range(n):
        x = image[i * batch_size:(i + 1) * batch_size]
        p = laplacian_pyramid(x.to(device), n_pyramids, device=device)
        p = [x.cpu() for x in p]
        pyramids.append(p)
    del x
    result = []
    for i in range(n_pyramids + 1):
        x = []
        for j in range(n):
            x.append(pyramids[j][i])
        result.append(torch.cat(x, dim=0))
    return result

