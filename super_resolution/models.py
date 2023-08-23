from tqdm import tqdm
import os
import sys

from super_resolution.sr_utils.GMM import GMMnD
from super_resolution.sr_utils.image import extract_patches
from super_resolution.sr_utils.predefined_filters import get_naive_kernels, get_gabor_filters

import torch
import torch.nn.functional as F
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from patch_swd import PatchSWDLoss, duplicate_to_match_lengths


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

    def loss(self, image):
        return self.criteria(image, self.ref_image)

    def run(self, init_image):
        img, losses = match_patch_distributions(init_image, self.loss, self.num_steps, self.lr)
        if self.gradient_projector is not None:
            img = self.gradient_projector(img)
        img = torch.clip(img.detach(), -1, 1)
        return img, losses

class predefinedDirectSWD(DirectSWD):
    def __init__(self, ref_image, p=5, s=1, n_proj=64, num_steps=500, lr=0.001, name=None):
        super(predefinedDirectSWD, self).__init__(ref_image, p, s, n_proj, mode="Fixed", num_steps=num_steps, lr=lr, name=name)
        self.criteria.rand = get_gabor_filters(self.p, c=self.ref_image.shape[1]).to(self.criteria.rand.device).float()
        self.p = self.criteria.p = self.criteria.rand.shape[-1]
        self.n_proj = self.criteria.num_proj = self.criteria.rand.shape[0]
        self.name = "Predefined" + self.name


class LapSWD(DirectSWD):
    def __init__(self, ref_image, low_res_big, p=5, s=1, n_proj=64, mode="Resampled", num_steps=500, lr=0.001,
                 gradient_projector=None, name=None):
        super(LapSWD, self).__init__(ref_image - low_res_big, p, s, n_proj, mode, num_steps=num_steps, lr=lr,
                                     gradient_projector=gradient_projector, name=name)
        self.name = "LapSWD" + self.name

    def loss(self, image):
        return self.criteria(image, self.ref_image)

    def match_patch_distributions(self, synthesized_image, loss_func, num_steps, lr):
        from torchvision.transforms import Resize
        d = synthesized_image.shape[-1]

        synthesized_image.requires_grad_(True)
        optim = torch.optim.Adam([synthesized_image], lr=lr)
        losses = []

        pbar = tqdm(range(num_steps))
        for _ in pbar:
            optim.zero_grad()
            loss = loss_func(synthesized_image - Resize(d, antialias=True)(Resize(d//4, antialias=True)(synthesized_image)).clone())
            loss.backward()
            optim.step()

            losses.append(loss.item())
            pbar.set_description(f"Loss: {loss.item()}")

        synthesized_image = torch.clip(synthesized_image.detach(), -1, 1)
        return synthesized_image, losses

    def run(self, init_image):
        img, losses = self.match_patch_distributions(init_image, self.loss, self.num_steps, self.lr)
        if self.gradient_projector is not None:
            img = self.gradient_projector(img)
        return img, losses


class WindowDirectSWD:
    def __init__(self, ref_image, w, ws, p=5, ps=1, n_proj=64, mode="Resample", num_steps=500, lr=0.001,
                 gradient_projector=None, name=None):
        self.name = f"{mode}-{n_proj}-{w}x{ws}-{p}x{ps}"
        if name is not None:
            self.name += "_" +name
        self.p = p
        self.ps = ps
        self.w = w
        self.ws = ws
        self.n_proj = n_proj
        self.mode = mode
        self.num_steps = num_steps
        self.lr = lr
        self.ref_windows = extract_patches(ref_image, self.w, self.ws)

        self.gradient_projector = gradient_projector
        self.criteria = PatchSWDLoss(patch_size=self.p, stride=self.ps, num_proj=self.n_proj, c=ref_image.shape[1])

    def loss(self, image):
        windows = extract_patches(image, self.w, self.ws)
        loss = 0
        for i in range(windows.shape[0]):
            loss += self.criteria(windows[i], self.ref_windows[i])
        return loss

    def run(self, init_image):
        img, losses = match_patch_distributions(init_image, self.loss, self.num_steps, self.lr)
        if self.gradient_projector is not None:
            img = self.gradient_projector(img)
        img = torch.clip(img.detach(), -1, 1)
        return img, losses


class MSSWD:
    def __init__(self, ref_image, ps=(5,9), s=1, n_proj=64, mode="Resample", num_steps=500, lr=0.001,
                 gradient_projector=None, name=None):
        self.name = f"MS_{mode}-{n_proj}-{ps}x{s}"
        if name is not None:
            self.name += "_" + name
        self.s = s
        self.n_proj = n_proj
        self.mode = mode
        self.num_steps = num_steps
        self.lr = lr
        self.ref_image = ref_image
        self.gradient_projector = gradient_projector
        self.criterias = [PatchSWDLoss(patch_size=p, stride=self.s, num_proj=self.n_proj, c=self.ref_image.shape[1]) for p in ps]

    def loss(self, image):
        return sum(criteria(image, self.ref_image) for criteria in self.criterias)

    def run(self, init_image):
        img, losses = match_patch_distributions(init_image, self.loss, self.num_steps, self.lr)
        if self.gradient_projector is not None:
            img = self.gradient_projector(img)
        return img, losses


class GMMSWD(DirectSWD):
    def __init__(self, ref_image, p=5, s=1, n_proj=64, num_steps=500, lr=0.001, n_components=5, mode="Fixed",
                 gradient_projector=None, name=None):
        super().__init__(self)
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


class TwoScalesSWD:
    def __init__(self, ref_image, scale_factor, p=20, s=1, n_proj=64, num_steps=500, lr=0.001, n_components=5, mode="Fixed",
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
        self.scale_factor = scale_factor
        self.ref_image = ref_image

        self.criteria = PatchSWDLoss(patch_size=self.p, stride=self.s, num_proj=self.n_proj, c=ref_image.shape[1])

    def loss(self, image):
        from torchvision.transforms import Resize
        self.criteria.sample_projections()
        # sr_filters = self.criteria.rand.to(image.device)
        # lr_filters = Resize(self.p//self.scale_factor, antialias=True)(sr_filters.clone())
        lr_filters = self.criteria.rand.to(image.device)
        sr_filters = Resize(self.p*self.scale_factor, antialias=True)(lr_filters.clone())
        projy = F.conv2d(image, sr_filters).transpose(1,0).reshape(self.n_proj, -1)
        projx = F.conv2d(self.ref_image, lr_filters).transpose(1,0).reshape(self.n_proj, -1)

        projx, projy = duplicate_to_match_lengths(projx, projy)

        projx, _ = torch.sort(projx, dim=1)
        projy, _ = torch.sort(projy, dim=1)

        loss = torch.abs(projx - projy).mean()
        return loss

    def run(self, init_image):
        img, losses = match_patch_distributions(init_image, self.loss, self.num_steps, self.lr)
        if self.gradient_projector is not None:
            img = self.gradient_projector(img)
        img = torch.clip(img.detach(), -1, 1)
        return img, losses

class GD_gradient_projector:
    def __init__(self, corrupt_image, operator, n_steps=100, lr=0.0001, reg_weight=0):
        self.corrupt_image = corrupt_image
        self.operator = operator
        self.n_steps = n_steps
        self.lr = lr
        self.reg_weight = reg_weight

    def __call__(self, im):
        # im.requires_grad_(True)
        init_im = im.clone().detach()
        optim = torch.optim.Adam([im], lr=self.lr)
        for i in range(self.n_steps):
            loss = ((self.operator(im) - self.corrupt_image)**2).mean()
            if self.reg_weight > 0:
                loss += self.reg_weight * ((im - init_im) ** 2).mean()
            loss.backward()
            optim.step()

        im = torch.clip(im, -1, 1)
        return im.detach()


class back_projector:
    def __init__(self, corrupt_image, operator, n_steps=10):
        self.corrupt_image = corrupt_image
        self.operator = operator
        self.n_steps = n_steps

    def __call__(self, im):
        for i in range(self.n_steps):
            im = im + self.operator.naive_reverse(self.corrupt_image - self.operator(im))
        return im


def match_patch_distributions(synthesized_image, loss_func, num_steps, lr):
    synthesized_image.requires_grad_(True)
    optim = torch.optim.Adam([synthesized_image], lr=lr)
    losses = []

    pbar = tqdm(range(num_steps))
    for _ in pbar:
        optim.zero_grad()
        loss = loss_func(synthesized_image)
        loss.backward()
        optim.step()

        losses.append(loss.item())
        pbar.set_description(f"Loss: {loss.item()}")

    return synthesized_image, losses