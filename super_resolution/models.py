from sklearn.mixture import GaussianMixture
from torchvision.transforms import Resize
from tqdm import tqdm
import os
import sys

from super_resolution.GMM import GMM1D, GMMnD

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from patch_swd import PatchSWDLoss


class DirectSWD:
    def __init__(self, ref_image, p=5, s=1, n_proj=64, mode="Resample", num_steps=500, lr=0.001,
                 gradient_projector=None, name=None):
        self.name = f"{mode}-{n_proj}-{p}x{s}"
        if name is not None:
            self.name += "_" +name
        self.p = p
        self.s = s
        self.n_proj = n_proj
        self.mode = mode
        self.num_steps = num_steps
        self.lr = lr
        self.ref_image = ref_image
        self.gradient_projector = gradient_projector
        self.criteria = PatchSWDLoss(patch_size=self.p, stride=self.s, num_proj=self.n_proj, c=self.ref_image.shape[1])

    def run(self, init_image):
        if self.mode == "Fixed-opt":
            DirectSWD.optimize_directions(init_image, self.ref_image, self.criteria, num_steps=self.num_steps, lr=self.lr)

        return self.match_patch_distributions(init_image)

    def match_patch_distributions(self, synthesized_images):
        synthesized_images.requires_grad_(True)
        optim = torch.optim.Adam([synthesized_images], lr=self.lr)
        losses = []
        pbar = tqdm(range(self.num_steps))
        for _ in pbar:
            if self.mode == "Resample":
                self.criteria.init()
            optim.zero_grad()
            loss = self.criteria(synthesized_images, self.ref_image)
            loss.backward()
            if self.gradient_projector is not None:
                    synthesized_images.grad = self.gradient_projector(synthesized_images.grad)
            optim.step()

            losses.append(loss.item())
            pbar.set_description(f"Loss: {loss.item()}")

        synthesized_images = torch.clip(synthesized_images.detach(), -1, 1)
        return synthesized_images, losses

    @staticmethod
    def optimize_directions(synthesized_images, reference_images, criteria, num_steps=300, lr=0.001):
        criteria.rand.requires_grad_(True)
        optim = torch.optim.Adam([criteria.rand], lr=lr)
        losses = []
        pbar = tqdm(range(num_steps))
        for i in pbar:
            # Optimize image
            optim.zero_grad()
            loss = -criteria(synthesized_images, reference_images)
            loss.backward()
            optim.step()
            losses.append(loss.item())
            # Update staus
            pbar.set_description(f"Loss: {loss.item()}")

        return losses


class GMMSWD:
    def __init__(self, ref_image, p=5, s=1, n_proj=64, num_steps=500, lr=0.001, n_components=5, mode="Fixed",
                 gradient_projector=None, name=None):
        self.name = f"{mode}_{n_components}-GMM-{n_proj}-{p}x{s}"
        if name is not None:
            self.name += "_" +name
        self.p = p
        self.s = s
        self.n_proj = n_proj
        self.num_steps = num_steps
        self.lr = lr
        self.mode = mode
        self.gradient_projector = gradient_projector

        criteria = PatchSWDLoss(patch_size=self.p, stride=self.s, num_proj=self.n_proj, c=ref_image.shape[1])
        criteria.init()

        self.rand = criteria.rand.to(ref_image.device)
        projx = F.conv2d(ref_image, self.rand).transpose(1,0).reshape(self.n_proj, -1)
        self.gmms = GMMnD(n_components, projx)
        self.projx = None

    def loss(self, image):
        projy = F.conv2d(image, self.rand).transpose(1,0).reshape(self.n_proj, -1)
        if self.projx is None or self.mode == "Resample":
            self.projx = self.gmms.sample(projy.shape[1])
            self.projx, _ = torch.sort(self.projx, dim=1)

        projy, _ = torch.sort(projy, dim=1)

        loss = torch.abs(self.projx - projy).mean()
        return loss

    def run(self, init_image):
        return self.match_patch_distributions(init_image)

    def match_patch_distributions(self, synthesized_image):
        synthesized_image.requires_grad_(True)
        optim = torch.optim.Adam([synthesized_image], lr=self.lr)
        losses = []

        pbar = tqdm(range(self.num_steps))
        for _ in pbar:
            optim.zero_grad()
            loss = self.loss(synthesized_image)
            loss.backward()
            if self.gradient_projector is not None:
                    synthesized_image.grad = self.gradient_projector(synthesized_image.grad)
            optim.step()

            losses.append(loss.item())
            pbar.set_description(f"Loss: {loss.item()}")

        synthesized_images = torch.clip(synthesized_image.detach(), -1, 1)
        return synthesized_images, losses


class gradient_projector:
    def __init__(self, low_dim, high_dim, resize_kwargs):
        self.low_dim = low_dim
        self.resize_kwargs = resize_kwargs
        self.high_dim = high_dim

    def __call__(self, grad):
        with torch.no_grad():
            grad_down = Resize(self.low_dim, **self.resize_kwargs)(grad)
            grad -= Resize(self.high_dim, **self.resize_kwargs)(grad_down)
            return grad