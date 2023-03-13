import matplotlib.pyplot as plt
import numpy as np
import torch


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
    sigma = 1
    for theta in range(4):
        theta = theta / 4. * np.pi
        for frequency in (0.05, 1):
            for ar in [0.25, 0.75]:
                kernel_cv = cv2.getGaborKernel((dim,dim), sigma, theta, frequency, ar, 0, ktype=cv2.CV_32F)
                kernel_cv = (kernel_cv - kernel_cv.min()) / (kernel_cv.max() - kernel_cv.min())
                kernels.append(kernel_cv)
    return kernels


def get_naive_kernels(dim):
    kernels = []
    for x in [0.1, 0.25, 0.5]:
        for y in [0.1, 0.25, 0.5]:
            ker = np.zeros((dim, dim))
            ker[int(np.floor(x*dim)), int(np.floor(y*dim))] = 1
            ker[-int(np.ceil(x*dim)), -int(np.ceil(y*dim))] = 1
            kernels.append(ker)
    kernels = np.stack(kernels)
    kernels = torch.from_numpy(kernels)[:, None]
    kernels = kernels.repeat(1,3,1,1)
    return kernels


if __name__ == '__main__':
    import cv2
    for ker in get_naive_kernels(7):
        plt.imshow(ker)
        plt.show()