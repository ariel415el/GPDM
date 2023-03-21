import os

from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import save_image


def load_image(path, gray=False):
    img = np.array(Image.open(path))
    if gray:
        img = np.mean(img, axis=-1, keepdims=True)
    img = img - img.min()
    img = img / img.max()
    img = img * 2 - 1
    if len(img.shape) == 2:
        img = img[..., None]
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).float()
    return img


def read_data(path):
    if os.path.isdir(path):
        refernce_images = torch.cat([load_image(f'{path}/{x}') for x in os.listdir(path)], dim=0)
    else:
        refernce_images = load_image(path)
    return refernce_images


def dump_images(images, out_dir):
    if os.path.exists(out_dir):
        i = len(os.listdir(out_dir))
    else:
        i = 0
        os.makedirs(out_dir)
    for j in range(images.shape[0]):
        save_image(images[j], os.path.join(out_dir, f"{i}.png"), normalize=True)
        i += 1

def show_nns(images, ref_images, out_dir):
    nn_indices = []
    for i in range(len(images)):
        dists = torch.mean((ref_images - images[i].unsqueeze(0))**2, dim=(1,2,3))
        j = dists.argmin().item()
        nn_indices.append(j)
    debug_image = torch.cat([images, ref_images[nn_indices] ], dim=0)
    save_image(debug_image, os.path.join(out_dir, f"NNs.png"), normalize=True, nrow=len(images))


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


def plot_loss(losses, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig1 = plt.figure()
    fig1.suptitle(f'Last loss: {losses[-1]}')
    ax = fig1.add_subplot(111)
    ax.plot(np.arange(len(losses)), losses)
    # ax.plot(np.arange(len(losses)), np.log(losses))
    fig1.savefig(path)
    plt.close(fig1)