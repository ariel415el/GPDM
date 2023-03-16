from tqdm import tqdm
import os
import sys

from super_resolution.sr_utils.GMM import GMMnD
from super_resolution.sr_utils.predefined_filters import get_naive_kernels, get_gabor_filters

import torch
import torch.nn.functional as F
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
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

    def loss(self, image):
        return self.criteria(image, self.ref_image)

    def run(self, init_image):
        return match_patch_distributions(init_image, self.loss, self.num_steps, self.lr, self.gradient_projector)


class predefinedDirectSWD(DirectSWD):
    def __init__(self, ref_image, p=5, s=1, n_proj=64, num_steps=500, lr=0.001, name=None):
        super(predefinedDirectSWD, self).__init__(ref_image, p, s, n_proj, mode="Fixed", num_steps=num_steps, lr=lr, name=name)
        self.criteria.rand = get_gabor_filters(self.p).to(self.criteria.rand.device).float()
        self.p = self.criteria.p = self.criteria.rand.shape[-1]
        self.n_proj = self.criteria.num_proj = self.criteria.rand.shape[0]
        self.name = "Predefined" + self.name


class LapSWD(DirectSWD):
    def __init__(self, ref_image, low_res_big, p=5, s=1, n_proj=64, mode="Resampled", num_steps=500, lr=0.001, name=None):
        super(LapSWD, self).__init__(ref_image - low_res_big, p, s, n_proj, mode, num_steps=num_steps, lr=lr, name=name)
        self.name = "LapSWD" + self.name
        self.low_res_big = low_res_big

    def loss(self, image):
        return self.criteria(image, self.ref_image)

    def run(self, init_image):
        d = init_image.shape[-1]
        from torchvision.transforms import Resize
        from torchvision.utils import save_image
        init_image -= Resize(d, antialias=True)(Resize(d//4, antialias=True)(init_image))
        save_image(init_image, "init.png", normalize=True)
        save_image(self.ref_image, "ref.png", normalize=True)
        img, losses = match_patch_distributions(init_image, self.loss, self.num_steps, self.lr, self.gradient_projector)
        save_image(img, "final.png", normalize=True)
        return self.low_res_big + img, losses
        # return img, losses


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
        return match_patch_distributions(init_image, self.loss, self.num_steps, self.lr, self.gradient_projector)


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
        return match_patch_distributions(init_image, self.loss, self.num_steps, self.lr, self.gradient_projector)


class GD_gradient_projector:
    def __init__(self, corrupt_image, operator, n_steps=100, lr=0.0001, reg_weight=0):
        self.corrupt_image = corrupt_image
        self.operator = operator
        self.n_steps = n_steps
        self.lr = lr
        self.reg_weight = reg_weight

    def __call__(self, im):
        init_im = im.clone().detach()
        optim = torch.optim.Adam([im], lr=self.lr)
        for i in range(self.n_steps):
            loss = ((self.operator(im) - self.corrupt_image)**2).mean()
            if self.reg_weight > 0:
                loss += self.reg_weight * ((im - init_im) ** 2).mean()
            loss.backward()
            optim.step()
        return im


# class LaplacianSharpening:
#     def run(self, ini_image):
#         F.conv2d(batched_channels, k, padding=size // 2)

def match_patch_distributions(synthesized_image, loss_func, num_steps, lr, gradient_projector):
    synthesized_image.requires_grad_(True)
    optim = torch.optim.Adam([synthesized_image], lr=lr)
    losses = []

    pbar = tqdm(range(num_steps))
    for _ in pbar:
        optim.zero_grad()
        loss = loss_func(synthesized_image)
        loss.backward()
        optim.step()
        if gradient_projector is not None:
            #         synthesized_image.grad = self.gradient_projector(synthesized_image.grad)
            synthesized_image = gradient_projector(synthesized_image)

        losses.append(loss.item())
        pbar.set_description(f"Loss: {loss.item()}")

    synthesized_image = torch.clip(synthesized_image.detach(), -1, 1)
    return synthesized_image, losses