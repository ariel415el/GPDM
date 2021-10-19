import torch
from distribution_metrics.common import extract_patches


def compute_swd(x, y, num_proj=256):
    """Compute a Sliced Wasserstein distance between two equal size sets of vectors using num_proj projections"""
    assert len(x.shape) == len(y.shape) and len(y.shape) == 2
    rand = torch.randn(x.size(1), num_proj).to(x.device)  # (slice_size**2*ch)
    rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize
    # projection into (batch-zie, num_projections)
    proj1 = torch.matmul(x, rand)
    proj2 = torch.matmul(y, rand)

    # sort by first dimension means each column is sorted separately
    proj1, _ = torch.sort(proj1, dim=0)
    proj2, _ = torch.sort(proj2, dim=0)


    d = torch.abs(proj1 - proj2)
    return torch.mean(d)


def find_nearest(array, value):
    # idx = torch.searchsorted(sorted_sequence=array.contiguous(), input=value, right=False)
    import numpy as np
    idx = np.searchsorted(array.cpu().detach().numpy(), value.item(), side='left')
    if idx > 0 and (idx == len(array) or torch.abs(value - array[idx - 1]) < torch.abs(value - array[idx])):
        return array[idx - 1]
    else:
        return array[idx]


def compute_coherence_swd(input, target, num_proj=256):
    """Compute a Sliced Wasserstein distance between two equal size sets of vectors using num_proj projections"""
    rand = torch.randn(input.size(1), num_proj).to(input.device)  # (slice_size**2*ch)
    rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize

    # projection into (batch-zie, num_projections)
    projected_input = torch.matmul(input, rand)
    proj_target = torch.matmul(target, rand)

    # sort by first dimension means each column is sorted separately
    proj_target = torch.argsort(proj_target, dim=0)
    projected_input = torch.argsort(projected_input, dim=0)

    loss = 0
    for i in range(num_proj):
        loss += torch.abs(input[projected_input[:, i]] - target[proj_target[:, i]]).mean()
    return loss / num_proj
    x = 1
    # proj_target = proj_target.contiguous()
    # loss = 0
    # for i in range(num_proj):
    #     for j in range(projected_input.shape[0]):
    #         nearest_target_projection = find_nearest(proj_target[:, i], projected_input[j, i])
    #         loss += torch.abs(nearest_target_projection - projected_input[j, i])
    # return loss / (num_proj * projected_input.shape[0])


class PatchSWDLoss(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, num_proj=256, normalize_patch='none', batch_reduction='mean'):
        super(PatchSWDLoss, self).__init__()
        self.name = f"PatchSWD(p-{patch_size}:{stride})"
        self.patch_size = patch_size
        self.stride = stride
        self.num_proj = num_proj
        self.batch_reduction = batch_reduction
        self.normalize_patch = normalize_patch

    def forward(self, x, y):
        b, c, h, w = x.shape
        x_patches = extract_patches(x, self.patch_size, self.stride, self.normalize_patch)
        y_patches = extract_patches(y, self.patch_size, self.stride, self.normalize_patch)

        if y_patches.shape[1] > x_patches.shape[1]:
            indices = torch.randperm(y_patches.shape[1])[:x_patches.shape[1]]
            y_patches = y_patches[:, indices]
            x_patches = x_patches[:, torch.randperm(x_patches.shape[1])]

        elif x_patches.shape[1] > y_patches.shape[1]:
            indices = torch.randperm(x_patches.shape[1])[:y_patches.shape[1]]
            x_patches = x_patches[:, indices]
            y_patches = y_patches[:, torch.randperm(y_patches.shape[1])]

        results = []
        for i in range(b):
            results.append(compute_swd(x_patches[i], y_patches[i], num_proj=self.num_proj))

        results = torch.stack(results)

        if self.batch_reduction == 'mean':
            return results.mean()
        else:
            return results


class PatchCoherentSWDLoss(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, num_proj=256, normalize_patch='none', batch_reduction='mean'):
        super(PatchCoherentSWDLoss, self).__init__()
        self.name = f"PatchSWD(p-{patch_size}:{stride})"
        self.patch_size = patch_size
        self.stride = stride
        self.num_proj = num_proj
        self.batch_reduction = batch_reduction
        self.normalize_patch = normalize_patch

    def forward(self, x, y):
        b, c, h, w = x.shape
        x_patches = extract_patches(x, self.patch_size, self.stride, self.normalize_patch)
        y_patches = extract_patches(y, self.patch_size, self.stride, self.normalize_patch)

        results = []
        for i in range(b):
            results.append(compute_coherence_swd(x_patches[i], y_patches[i], num_proj=self.num_proj))

        results = torch.stack(results)

        if self.batch_reduction == 'mean':
            return results.mean()
        else:
            return results


if __name__ == '__main__':
    x = torch.ones((5, 3, 64, 64))
    y = torch.ones((5, 3, 64, 64)) * 3
    loss = PatchSWDLoss(batch_reduction='none')

    print(loss(x, y))
