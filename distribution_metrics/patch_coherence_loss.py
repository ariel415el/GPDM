import torch
from distribution_metrics.common import extract_patches
from utils import _fspecial_gauss_1d, conv_gauss, get_kernel_gauss


def compute_ssim(x_patches, y_patches, patch_size, K=(0.01, 0.03)):
    C1, C2 = 0.01**2, 0.03**2
    filter = _fspecial_gauss_1d(patch_size, 1.5).reshape(-1).to(x_patches.device) / 3

    x_patches = x_patches.reshape(-1, 3, patch_size**2).transpose(1,0)
    y_patches = y_patches.reshape(-1, 3, patch_size**2).transpose(1,0)
    mu_x = x_patches @ filter
    mu_y = y_patches @ filter

    mu_x_mu_y = (mu_x[:, None, :] * mu_y[:, :, None])
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x_sq = x_patches**2 @ filter - mu_x_sq
    sigma_y_sq = y_patches**2 @ filter - mu_y_sq
    sigma_xy = (x_patches[:, None, :] * y_patches[:, :, None]) @ filter - mu_x_mu_y

    cs_map = (2 * sigma_xy + C2) / (sigma_x_sq[:, None, :] + sigma_y_sq[:, :, None] + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu_x_mu_y + C1) / (mu_x_sq[:, None, :] + mu_y_sq[:, :, None] + C1))

    ssim_map *= cs_map

    return ssim_map.mean(0)

def compute_patch_coherence(input_patches, target_patches, patch_size, mode='l2'):
    if mode == 'cosine':
        input_patches = torch.clip((input_patches + 1) / 2, 0, 1)
        target_patches = torch.clip((target_patches + 1) / 2, 0, 1)
        dist_matrix = 0.5 - compute_ssim(input_patches, target_patches, patch_size) / 2

    elif mode == 'l2':
        dist_matrix = ((input_patches[None, :] - target_patches[:, None])**2).mean(2) # (dist_matrix[i,j] = d(target_patches[i], input_patches[j])

    else:
        loss = 0
        for i in range(input_patches.shape[0]):
            loss += ((input_patches[i].unsqueeze(0) - target_patches)**2).mean(1).min()
        return loss

    alpha = 0.05
    dist_matrix /= (torch.min(dist_matrix, dim=1)[0] + alpha)[:, None]  # reduces distance to target patches with no similar input patche
    loss = torch.min(dist_matrix, dim=0)[0].mean()



    return loss


class PatchCoherentLoss(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, mode='l2', batch_reduction='mean'):
        super(PatchCoherentLoss, self).__init__()
        self.name = f"PatchCoheren(p-{patch_size}:{stride}_M-{mode})"
        self.patch_size = patch_size
        self.stride = stride
        self.batch_reduction = batch_reduction
        self.mode = mode

    def forward(self, x, y):
        b, c, h, w = x.shape
        x = x.half()
        y = y.half()
        x_patches = extract_patches(x, self.patch_size, self.stride, 'none')
        y_patches = extract_patches(y, self.patch_size, self.stride, 'none')

        # x_patches = (((1 + x_patches) / 2) * 255).to(torch.uint8)
        # y_patches = (((1 + y_patches) / 2) * 255).to(torch.uint8)

        results = []
        for i in range(b):
            results.append(compute_patch_coherence(x_patches[i], y_patches[i], self.patch_size, self.mode))

        results = torch.stack(results)

        if self.batch_reduction == 'mean':
            return results.mean()
        else:
            return results

if __name__ == '__main__':
    input_image = torch.randn((3,90,90))
    target_image = torch.randn((3,90,90)) * 2


    compute_patch_coherence(input_image, target_image, 5, 3)
