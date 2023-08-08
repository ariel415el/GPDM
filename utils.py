import os
from math import sqrt

from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from torchvision import transforms as T


def get_transforms(center_crop):
    transforms = [T.ToTensor()]
    if center_crop is not None: transforms += [T.CenterCrop(size=center_crop)]
    transforms+=[T.Normalize((0.5,), (0.5,))]
    return T.Compose(transforms)


def load_image(path, make_square=False):
    img = Image.open(path).convert('RGB')
    transforms = get_transforms(center_crop=min(img.size[:2]) if make_square else None)
    img = transforms(img).unsqueeze(0)
    return img


def read_data(path, max_inputs):
    if os.path.isdir(path):
        paths = [f'{path}/{x}' for x in os.listdir(path)]
        if max_inputs is not None:
            paths = paths[:max_inputs]
        refernce_images = torch.cat([load_image(p, make_square=True) for p in paths], dim=0)
        print("# Warning! Center cropping non square inputs if any")
    else:
        refernce_images = load_image(path)
    return refernce_images


def dump_images(batch, out_dir):
    nrow = int(sqrt(len(batch)))
    save_image((batch + 1)/2, os.path.join(out_dir, "outputs.png"), nrow=nrow, normalize=False, pad_value=1, scale_each=True)

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