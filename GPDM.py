import os

from torchvision.utils import save_image
from tqdm import tqdm
import torch
from torchvision.transforms import Resize as tv_resize
from utils import plot_loss, load_image


def generate(reference_images,
             criteria,
             init_from: str = 'zeros',
             pyramid_scales=(32, 64, 128, 256),
             lr: float = 0.01,
             num_steps: int = 300,
             aspect_ratio=(1, 1),
             additive_noise_sigma=0.0,
             device: str = 'cuda:0',
             debug_dir=None):
    """
    Run the GPDM model to generate an image/s with a similar patch distribution to reference_images/s with a given criteria.
    This manages the coarse to fine optimization steps.
    """
    pbar = GPDMLogger(num_steps, len(pyramid_scales))
    criteria = criteria.to(device)

    reference_images = reference_images.to(device)
    synthesized_images = get_fist_initial_guess(reference_images, init_from, additive_noise_sigma).to(device)
    initial_image_shape = synthesized_images.shape[-2:]

    for scale in pyramid_scales:
        pbar.new_lvl()
        lvl_references = tv_resize(scale)(reference_images)
        lvl_output_shape = get_output_shape(initial_image_shape, scale, aspect_ratio)
        synthesized_images = tv_resize(lvl_output_shape)(synthesized_images)

        synthesized_images = _match_patch_distributions(synthesized_images, lvl_references, criteria, num_steps, lr,
                                                        pbar, debug_dir)
        # Decrease learning rate
        lr *= 0.9

        if debug_dir:
            save_image(lvl_references, os.path.join(debug_dir, f'references-lvl-{pbar.lvl}.png'), normalize=True)
            save_image(synthesized_images, os.path.join(debug_dir, f'outputs-lvl-{pbar.lvl}.png'), normalize=True)

    pbar.pbar.close()
    return synthesized_images


def _match_patch_distributions(synthesized_images, reference_images, criteria, num_steps, lr, pbar, debug_dir=None):
    """
    Minimizes criteria(synthesized_images, reference_images) for num_steps SGD steps by differentiating self.synthesized_images
    :param reference_images: tensor of shape (b, C, H1, W1)
    :param synthesized_images: tensor of shape (b, C, H2, W2)
    :param debug_dir:
    """
    synthesized_images.requires_grad_(True)
    optim = torch.optim.Adam([synthesized_images], lr=lr)
    losses = []
    for i in range(num_steps):
        # Optimize image
        optim.zero_grad()
        batch_indices = torch.randperm(len(reference_images))
        loss = criteria(synthesized_images, reference_images[batch_indices])
        loss.backward()
        optim.step()

        # Update staus
        losses.append(loss.item())
        pbar.step()
        pbar.print()

    # write optimization debut images
    if debug_dir:
        plot_loss(losses, os.path.join(debug_dir, f'train_losses-lvl-{pbar.lvl}.png'))

    return torch.clip(synthesized_images.detach(), -1, 1)


class GPDMLogger:
    """Keeps track of the levels and steps of optimization. Logs it via TQDM"""

    def __init__(self, n_steps, n_lvls):
        self.n_steps = n_steps
        self.n_lvls = n_lvls
        self.lvl = -1
        self.lvl_step = 0
        self.steps = 0
        self.pbar = tqdm(total=self.n_lvls * self.n_steps, desc='Starting')

    def step(self):
        self.pbar.update(1)
        self.steps += 1
        self.lvl_step += 1

    def new_lvl(self):
        self.lvl += 1
        self.lvl_step = 0

    def print(self):
        self.pbar.set_description(f'Lvl {self.lvl}/{self.n_lvls - 1}, step {self.lvl_step}/{self.n_steps}')


def get_fist_initial_guess(reference_images, init_from, additive_noise_sigma):
    if init_from == "zeros":
        synthesized_images = torch.zeros_like(reference_images)
    elif init_from == "target":
        synthesized_images = reference_images.clone()
    elif os.path.exists(init_from):
        synthesized_images = load_image(init_from)
        synthesized_images = synthesized_images.repeat(reference_images.shape[0], 1, 1, 1)
    else:
        raise ValueError("Bad init mode", init_from)
    if additive_noise_sigma:
        synthesized_images += torch.randn_like(synthesized_images) * additive_noise_sigma
    return synthesized_images


def get_output_shape(initial_image_shape, size, aspect_ratio):
    """Get the size of the output pyramid level"""
    h, w = initial_image_shape
    h, w = int(size * aspect_ratio[0]), int((w * size / h) * aspect_ratio[1])
    return h, w
