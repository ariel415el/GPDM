import torch
from distribution_metrics.common import extract_patches



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
    projected_target = torch.matmul(target, rand)

    # sort by first dimension means each column is sorted separately
    projected_input = torch.argsort(projected_input, dim=0)
    projected_target = torch.argsort(projected_target, dim=0)

    projected_target = projected_target.contiguous()
    loss = 0
    for j in range(num_proj):
        for i in range(len(projected_input)):
            nearest = find_nearest(projected_target[:, j], projected_input[i, j])
            loss += (projected_input[i, j] - nearest)**2
        return loss / len(projected_input)
    return loss / num_proj



class PatchCoherentSWDLoss(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, num_proj=1, normalize_patch='none', batch_reduction='mean'):
        super(PatchCoherentSWDLoss, self).__init__()
        self.name = f"PatchCoherentSWD(p-{patch_size}:{stride})"
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
    loss = PatchCoherentSWDLoss(batch_reduction='none')

    print(loss(x, y))
