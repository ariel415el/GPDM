import sys
import os
from typing import Tuple
from tqdm import tqdm

import cv2
import torch
from torchvision import transforms

sys.path.append('.')
from utils import aspect_ratio_resize, get_pyramid, cv2pt, conv_gauss, get_kernel_gauss, match_image_sizes, plot_loss,  save_image


class logger:
    """Keeps track of the levels and steps of optimization. Logs it via TQDM"""
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
    """An image generation model according to "Generating natural images with direct Patch Distributions Matching"""
    def __init__(self,
            scale_factor: Tuple[float, float] = (1., 1.),
            resize: int = None,
            pyr_factor: float = 0.7,
            n_scales: int = 5,
            lr: float = 0.05,
            num_steps: int = 500,
            init: str = 'noise',
            noise_sigma: float = 0.75,
            device: str = 'cuda:0',
    ):
        """
        :param scale_factor: scale of the output in relation to input
        :param resize: max size of input image dimensions
        :param pyr_factor: Downscale ratio of each pyramid level
        :param n_scales: number of pyramid scale downs
        :param lr: optimization starting learning rate
        :param num_steps: number of optimization steps at each pyramid level
        :param init: Initialization mode ('noise' / 'target' / 'blured_target' / <image_path>)
        :param noise_sigma: standard deviation of the zero mean normal noise added to the initialization
        :param device: cuda/cpu
        """
        self.scale_factor = scale_factor
        self.resize = resize
        self.pyr_factor = pyr_factor
        self.n_scales = n_scales
        self.lr = lr
        self.num_steps = num_steps
        self.init = init
        self.noise_sigma = noise_sigma
        self.device = torch.device(device)

        init_name = 'img' if os.path.exists(self.init) else self.init
        self.name = f'AR-{scale_factor}_R-{resize}_S-{pyr_factor}x{n_scales}_I-{init_name}+I(0,{noise_sigma})'

    def _get_target_pyramid(self, target_img_path):
        """Reads an image and create a pyraimd out of it. Ordered in increasing image size"""
        target_img = cv2.imread(target_img_path)
        if self.resize:
            target_img = aspect_ratio_resize(target_img, max_dim=self.resize)
        target_img = cv2pt(target_img)
        target_pyramid = get_pyramid(target_img, self.n_scales, self.pyr_factor)
        target_pyramid = [x.unsqueeze(0).to(self.device) for x in target_pyramid]
        return target_pyramid

    def get_synthesis_size(self, lvl):
        """Get the size of the output pyramid level"""
        lvl_img = self.target_pyramid[lvl]
        h, w = lvl_img.shape[-2:]
        h, w = int(h * self.scale_factor[0]), int(w * self.scale_factor[1])
        return h, w

    def _get_initial_image(self):
        """Prepare the initial image for optimization"""
        target_img = self.target_pyramid[-1]
        h, w = self.get_synthesis_size(lvl=0)
        if os.path.exists(self.init):
            initial_iamge = cv2pt(cv2.imread(self.init)).unsqueeze(0)
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
        Minimizes self.criteria(self.synthesized_image, target_img) for self.num_steps SGD steps
        :param target_img: tensor of shape (1, C, H, W)
        :param debug_dir:
        """
        optim = torch.optim.Adam([self.synthesized_image], lr=self.lr)
        losses = []
        for i in range(self.num_steps):

            # write optimization debut images
            if debug_dir and i % 100 == 0 or i == self.num_steps -1:
                save_image(self.synthesized_image,
                           os.path.join(debug_dir, 'optimization', f'lvl-{self.pbar.lvl}-{self.pbar.lvl_step}.png'))
                plot_loss(losses, os.path.join(debug_dir, 'train_losses',f'lvl-{self.pbar.lvl}-train_loss.png'))

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

        self.synthesized_image = torch.clip(self.synthesized_image, -1, 1)

    def run(self, target_img_path, criteria, debug_dir=None):
        """
        Run the GPDM model to generate an image with a similar patch distribution to target_img_path with a given criteria.
        This manages the coarse to fine optimization steps.
        """
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

            if debug_dir:
                save_image(lvl_target_img, os.path.join(debug_dir, f'target-lvl-{self.pbar.lvl}.png'))
                save_image(self.synthesized_image, os.path.join(debug_dir, f'output-lvl-{self.pbar.lvl}.png'))

        self.pbar.pbar.close()
        return self.synthesized_image.detach()[0]

