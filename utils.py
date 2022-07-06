import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import save_image


def load_image(path):
    return cv2pt(cv2.imread(path))


def dump_images(images, out_dir):
    if os.path.exists(out_dir):
        i = len(out_dir)
    else:
        i = 0
        os.makedirs(out_dir)
    for j in range(images.shape[0]):
        save_image(images[j], os.path.join(out_dir, f"{i}.png"), normalize=True)
        i += 1


def get_pyramid_scales(max_height, min_height, step):
    cur_scale = max_height
    scales = [cur_scale]
    while cur_scale > min_height:
        if type(step) == float:
            cur_scale = int(cur_scale * step)
        else:
            cur_scale -= step
        scales.append(cur_scale)

    return scales[::-1]



def cv2pt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float64) / 255.
    img = img * 2 - 1
    img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)

    return img


def plot_loss(losses, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.plot(np.arange(len(losses)), losses)
    # ax.plot(np.arange(len(losses)), np.log(losses))
    fig1.savefig(path)
    plt.close(fig1)