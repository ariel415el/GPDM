import os
from time import sleep
from typing import Tuple

from tqdm import tqdm

import cv2
import torch

import sys
sys.path.append('.')
from utils import aspect_ratio_resize, get_pyramid, cv2pt, conv_gauss, get_kernel_gauss, match_image_sizes, plot_loss, \
    save_image
import torchvision.utils as vutils
from torchvision import transforms


class logger:
    def __init__(self, n_steps, n_lvls):
        self.n_steps = n_steps
        self.n_lvls = n_lvls + 1
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
        # pass
        self.pbar.set_description(f'Lvl {self.lvl}/{self.n_lvls}, step {self.lvl_step}/{self.n_steps}')


class GPDM:
    def __init__(self,
            aspect_ratio: Tuple[float, float] = (1., 1.),
            resize: int = None,
            pyr_factor: float = 0.7,
            n_scales: int = 5,
            lr: float = 0.05,
            num_steps: int = 500,
            init: str = 'noise',
            noise_sigma: float = 0.75,
            device: str = 'cuda:0',
    ):
        self.aspect_ratio = aspect_ratio
        self.resize = resize
        self.pyr_factor = pyr_factor
        self.n_scales = n_scales
        self.lr = lr
        self.num_steps = num_steps
        self.init = init
        self.noise_sigma = noise_sigma
        self.device = torch.device(device)

        init_name = 'img' if os.path.exists(self.init) else self.init
        self.name = f'AR-{aspect_ratio}_R-{resize}_S-{pyr_factor}x{n_scales}_I-{init_name}+I(0,{noise_sigma})'

    def _get_target_pyramid(self, target_img_path):
        target_img = cv2.imread(target_img_path)
        if self.resize:
            target_img = aspect_ratio_resize(target_img, max_dim=self.resize)
        target_img = cv2pt(target_img)
        target_pyramid = get_pyramid(target_img, self.n_scales, self.pyr_factor)
        target_pyramid = [x.unsqueeze(0).to(self.device) for x in target_pyramid]
        return target_pyramid

    def get_synthesis_size(self, lvl):
        lvl_img = self.target_pyramid[lvl]
        h, w = lvl_img.shape[-2:]
        h, w = int(h * self.aspect_ratio[0]), int(w * self.aspect_ratio[1])
        return h, w

    def _get_initial_image(self):
        target_img = self.target_pyramid[-1]
        h, w = self.get_synthesis_size(lvl=0)
        if os.path.exists(self.init):
            initial_iamge = cv2pt(cv2.imread(self.init))
            initial_iamge = match_image_sizes(initial_iamge, target_img)
            initial_iamge = transforms.Resize((h, w), antialias=True)(initial_iamge)
        elif self.init == 'target':
            initial_iamge = transforms.Resize((h, w), antialias=True)(target_img)
        elif self.init == 'blured_target':
            initial_iamge = transforms.Resize((h, w), antialias=True)(target_img)
            initial_iamge = conv_gauss(initial_iamge, get_kernel_gauss(size=9, sigma=5, n_channels=3))
        else:
            initial_iamge = torch.zeros(1, 3, h, w)

        initial_iamge = initial_iamge.to(self.device)

        if self.noise_sigma > 0:
            initial_iamge += torch.normal(0, 0.75, size=(h, w)).reshape(1, 1, h, w).to(self.device)

        return initial_iamge

    def match_patch_distributions(self, target_img, debug_dir):
        """
        :param images: tensor of shape (H, W, C)
        """
        optim = torch.optim.Adam([self.synthesized_image], lr=self.lr)
        losses = []
        for i in range(self.num_steps):
            # write debut images
            if debug_dir and i % 100 == 0:
                save_image(torch.clip(self.synthesized_image, -1, 1),
                           os.path.join(debug_dir, f'lvl-{self.pbar.lvl}-{self.pbar.lvl_step}.png'))
                plot_loss(losses, os.path.join(debug_dir, f'lvl-{self.pbar.lvl}-train_loss.png'))

            # Optimize image
            optim.zero_grad()
            loss = self.criteria(self.synthesized_image, target_img)
            loss.backward()
            optim.step()

            # Update staus
            losses.append(loss)
            self.pbar.step()
            self.pbar.print()

            if i != 0 and i % 100 == 0:
                for g in optim.param_groups:
                    g['lr'] *= 0.9

    def run(self, target_img_path, criteria, debug_dir=None):
        self.pbar = logger(self.num_steps, self.n_scales)
        self.target_pyramid = self._get_target_pyramid(target_img_path)
        self.criteria = criteria.to(self.device)
        self.synthesized_image = self._get_initial_image()
        self.synthesized_image.requires_grad_(True)

        for lvl, lvl_target_img in enumerate(self.target_pyramid):
            self.pbar.new_lvl()
            if lvl > 0:
                with torch.no_grad():
                    h, w = self.get_synthesis_size(lvl=lvl)
                    self.synthesized_image = transforms.Resize((h, w), antialias=True)(self.synthesized_image)
                    self.synthesized_image.requires_grad_(True)

            self.match_patch_distributions(lvl_target_img, debug_dir)
            self.synthesized_image = torch.clip(self.synthesized_image, -1, 1)

        if debug_dir:
            vutils.save_image(torch.cat([self.synthesized_image, self.target_pyramid[-1]], dim=-1),
                              os.path.join(debug_dir, "debug.png"), normalize=True)
        self.pbar.pbar.close()
        return self.synthesized_image.detach()[0]

