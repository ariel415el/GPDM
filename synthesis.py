import os
from tqdm import tqdm

from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch

from utils import aspect_ratio_resize, get_pyramid, cv2pt, conv_gauss, get_kernel_gauss, match_image_sizes
import torchvision.utils as vutils
from torchvision import transforms


# def tv_loss(img):
#     w_variance = torch.mean(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
#     h_variance = torch.mean(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
#     return h_variance + w_variance



def match_patch_distributions(input_img, target_img, criteria, content_loss, conf, output_dir):
    """
    :param images: tensor of shape (H, W, C)
    """
    device = torch.device(conf.device)
    criteria = criteria.to(device)
    os.makedirs(output_dir, exist_ok=True)

    optimized_variable = input_img.clone().unsqueeze(0).to(device)
    optimized_variable.requires_grad_(True)
    optim = torch.optim.Adam([optimized_variable], lr=conf.lr)

    target_img_pt = target_img.unsqueeze(0).to(device)
    vutils.save_image(torch.clip(optimized_variable, -1, 1), f"{output_dir}/output-0.png", normalize=True)

    all_losses = []
    all_means = []
    for i in tqdm(range(1, conf.num_steps + 1)):
        optim.zero_grad()

        loss = criteria(optimized_variable, target_img_pt)
        all_losses.append(loss.item())

        if content_loss:
            loss += content_loss(optimized_variable)

        loss.backward()
        optim.step()

        if i % 100 == 0:
            for g in optim.param_groups:
                g['lr'] *= 0.9
        if i % 50 == 0:
            os.makedirs(output_dir, exist_ok=True)
            vutils.save_image(torch.clip(optimized_variable, -1, 1), f"{output_dir}/output-{i}.png", normalize=True)
        if i % 10 == 0:
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)

            all_means.append(np.mean(all_losses[-100:]))
            # ax.plot(np.arange(len(all_losses)), np.log(all_losses), label=f'{criteria.name}: {all_means[-1]:.6f}')
            ax.plot(np.arange(len(all_losses)), all_losses, label=f'{criteria.name}: {all_means[-1]:.6f}')
            ax.legend()
            fig1.savefig(f"{output_dir}/train_loss.png")
            plt.close(fig1)

    return torch.clip(optimized_variable.detach()[0], -1, 1)


def get_initial_image(conf, h, w, target_img):
    if conf.init == 'noise':
        synthesis = torch.randn((3, h, w)) * 0.1
    elif os.path.exists(conf.init):
        synthesis = cv2pt(cv2.imread(conf.init))
        synthesis = match_image_sizes(synthesis, target_img)
        synthesis = transforms.Resize((h, w), antialias=True)(synthesis)
    elif conf.init == '+noise':
        synthesis = transforms.Resize((h, w), antialias=True)(target_img)
        synthesis += torch.normal(0, 1, size=(h, w))[None, :]
    elif conf.init == 'blur':
        synthesis = transforms.Resize((h, w), antialias=True)(target_img)
        synthesis = conv_gauss(synthesis.unsqueeze(0), get_kernel_gauss(size=9, sigma=5, n_channels=3))[0]
        # synthesis += torch.randn((3, h, w)) * 0.5
        synthesis += torch.normal(0, 0.75, size=(h, w))[None, :]
    else:
        raise ValueError("No such init mode")
    return synthesis


def synthesize_image(target_img_path, criteria, content_loss, conf, output_dir):
    while os.path.exists(output_dir):
        output_dir += '#'
    os.makedirs(output_dir, exist_ok=True)

    target_img = cv2.imread(target_img_path)
    if conf.resize:
        target_img = aspect_ratio_resize(target_img, max_dim=conf.resize)
    target_img = cv2pt(target_img)
    target_pyramid = get_pyramid(target_img, conf.n_scales, conf.pyr_factor)

    for lvl, lvl_target_img in enumerate(target_pyramid):
        print(f"Starting lvl {lvl}")
        h, w = int(lvl_target_img.shape[1] * conf.aspect_ratio[0]), int(lvl_target_img.shape[2] * conf.aspect_ratio[1])
        if lvl == 0:
            synthesis = get_initial_image(conf, h, w, target_img)
            if content_loss:
                content_loss.init(synthesis)
        else:
            synthesis = transforms.Resize((h, w), antialias=True)(synthesis)

        vutils.save_image(lvl_target_img, os.path.join(output_dir, f"target-{lvl}.png"), normalize=True)
        vutils.save_image(synthesis, os.path.join(output_dir, f"org-{lvl}.png"), normalize=True)

        synthesis = match_patch_distributions(synthesis, lvl_target_img, criteria, content_loss, conf, output_dir=os.path.join(output_dir, str(lvl)))
        # conf.num_steps += 50
        vutils.save_image(synthesis, os.path.join(output_dir, f"final-{lvl}.png"), normalize=True)

    vutils.save_image(synthesis, output_dir + ".png", normalize=True)

    return synthesis
