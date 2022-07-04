from random import randint

import torch

from distribution_metrics.patch_swd import extract_patches


def efficient_compute_distances(x, y):
    dist = (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist

def compute_dists(x, y):
    dist = torch.sum((x[:, None] - y[None, :]) **2, -1)
    return dist

def dist_mat(input_patches, target_patches):
    dist_matrix = torch.zeros((len(input_patches), len(target_patches)), dtype=torch.float16).to(input_patches.device)
    b = 16
    n_batches = len(input_patches) // b
    for i in range(n_batches):
        # dist_matrix[i * b:(i + 1) * b] = torch.cdist(input_patches[i * b:(i + 1) * b], target_patches) **2
        dist_matrix[i * b:(i + 1) * b] = efficient_compute_distances(input_patches[i * b:(i + 1) * b], target_patches)
    if len(input_patches) % b != 0:
        # dist_matrix[n_batches * b:] = torch.cdist(input_patches[n_batches * b:], target_patches)**2
        dist_matrix[n_batches * b:] = efficient_compute_distances(input_patches[n_batches * b:], target_patches)
    return dist_matrix

def compute_patch_coherence(input_patches, target_patches, mode='detached'):
    # dist_matrix = torch.cdist(target_patches, input_patches)
    dist_matrix = dist_mat(target_patches, input_patches)
    min_indices = torch.min(dist_matrix, dim=0)[1]

    if mode == 'detached':
        return ((input_patches - target_patches[min_indices]) ** 2).mean()
    else:
        alpha = 0.05
        dist_matrix /= (torch.min(dist_matrix, dim=1)[0] + alpha)[:, None]  # reduces distance to target patches with no similar input patche
        loss = torch.min(dist_matrix, dim=0)[0].mean()
        return loss


class PatchCoherentLoss(torch.nn.Module):
    """For each patch in input image x find its NN in target y and sum the their distances"""
    def __init__(self, patch_size=7, stride=1, mode='detached', batch_reduction='mean'):
        super(PatchCoherentLoss, self).__init__()
        self.name = f"PatchCoheren(p-{patch_size}:{stride}_M-{mode})"
        self.patch_size = patch_size
        self.stride = stride
        self.batch_reduction = batch_reduction
        self.mode = mode

    def forward(self, x, y, mask=None):
        b, c, h, w = x.shape

        if self.stride > 1:
            rows_offset = randint(0, self.stride -1)
            cols_offset = randint(0, self.stride -1)
            x = x[:, :, rows_offset:, cols_offset:]
            y = y[:, :, rows_offset:, cols_offset:]

        x_patches = extract_patches(x, self.patch_size, self.stride)
        y_patches = extract_patches(y, self.patch_size, self.stride)

        results = []
        for i in range(b):
            results.append(compute_patch_coherence(x_patches[i], y_patches[i], self.mode))

        results = torch.stack(results)

        if self.batch_reduction == 'mean':
            return results.mean()
        else:
            return results

if __name__ == '__main__':
    input_image = torch.randn((1, 3,250,250)).cuda()
    target_image = torch.randn((1, 3,250,250)).cuda() * 2

    from time import time
    start = time()
    loss = PatchCoherentLoss(5, 3, 'batched_detached-l2').cuda()
    for i in range(10):
        loss(input_image, target_image)
    print(f"Time: {(time() - start) / 10}")
