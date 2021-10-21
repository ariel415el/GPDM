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


    # d = torch.abs(proj1 - proj2)
    d = ((proj1 - proj2)**2).mean()
    return torch.mean(d)


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


