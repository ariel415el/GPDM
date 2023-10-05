import os
from math import sqrt

from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from torchvision import transforms as T


def get_transforms(center_crop, gray_scale):
    transforms = [T.ToTensor()]
    if center_crop is not None: transforms += [T.CenterCrop(size=center_crop)]
    if gray_scale: transforms += [T.Grayscale()]
    transforms+=[T.Normalize((0.5,), (0.5,))]
    return T.Compose(transforms)


def load_image(path, make_square=False, gray_scale=False):
    img = Image.open(path).convert('RGB')
    transforms = get_transforms(center_crop=min(img.size[:2]) if make_square else None, gray_scale=gray_scale)
    img = transforms(img).unsqueeze(0)
    return img


def read_data(path, max_inputs, gray_scale):
    if os.path.isdir(path):
        paths = [f'{path}/{x}' for x in os.listdir(path)]
        if max_inputs is not None:
            paths = paths[:max_inputs]
        data = []
        for p in tqdm(paths):
            data.append(load_image(p, make_square=True, gray_scale=gray_scale))
        refernce_images = torch.cat(data, dim=0)
        print("# Warning! Center cropping non square inputs if any")
    else:
        refernce_images = load_image(path, gray_scale=gray_scale)
    return refernce_images


def dump_images(batch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    nrow = int(sqrt(len(batch)))
    save_image((batch + 1)/2, os.path.join(out_dir, "outputs.png"), nrow=nrow, normalize=False, pad_value=1, scale_each=True)


def to_np(img):
    if img.shape[0] == 1:
        img = img.repeat(3,1,1)
    img = img.add_(1).div(2).mul(255).clamp_(0, 255)
    if len(img.shape) == 3:
        img = img.permute(1, 2, 0)
    return img.to("cpu", torch.uint8).cpu().numpy()


def show_nns(outputs, ref_images, out_dir, n=16):
    # nn_indices = []
    s=2
    n = min(n,len(outputs))
    fig, axes = plt.subplots(2, n, figsize=(s * n, s * 2))
    for i in range(n):
        dists = torch.mean((ref_images - outputs[i].unsqueeze(0))**2, dim=(1,2,3))
        j = dists.argmin().item()
        axes[0, i].imshow(to_np(outputs[i]))
        axes[0, i].axis('off')
        axes[1, i].imshow(to_np(ref_images[j]))
        axes[1, i].axis('off')
        axes[1, i].set_title(f"NN L2: {(outputs[i] - ref_images[j]).pow(2).sum():.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"NNs.png"))
        # nn_indices.append(j)
    # debug_image = torch.cat([outputs[:n], ref_images[nn_indices]], dim=0)
    # save_image(debug_image, os.path.join(out_dir, f"NNs.png"), normalize=True, nrow=n)


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