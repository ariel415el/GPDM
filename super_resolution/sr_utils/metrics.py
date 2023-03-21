import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from super_resolution.sr_utils.image import laplacian_pyramid


def calculate_psnr(img1, img2):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    max_value = 255
    mse = ((img1.astype(np.float32)  - img2.astype(np.float32)) ** 2).mean()
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def calculate_ssim(img1, img2):
    return np.sum([ssim(img1[...,i], img2[...,i]) for i in range(img1.shape[2])])


if __name__ == '__main__':
    device = torch.device("cuda:0")
    image = torch.ones(64,3,128,128).to(device)
    pyr = laplacian_pyramid(image, 5, device)
    print([x.shape for x in pyr])