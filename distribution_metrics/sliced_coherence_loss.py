import torch
from distribution_metrics.common import extract_patches
import torch.nn.functional as F

def compute_coherence_1d_distance(input, target):
    # dist_matrix = ((input[None, :] - target[:, None]) ** 2)  # (dist_matrix[i,j] = d(target_patches[i], input_patches[j])
    # return torch.min(dist_matrix, dim=0)[0].mean()
    dist_matrix = ((input[None, :].detach() - target[:, None]) ** 2)  # (dist_matrix[i,j] = d(target_patches[i], input_patches[j])
    return (input - target[torch.min(dist_matrix, dim=0)[1]]).mean()


def gaussian(size, sigma):
    x = torch.arange(size)
    return torch.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2

def blur_1d(vec):
    f_vec = vec.unsqueeze(0).unsqueeze(0).float()
    filter = gaussian(size=7, sigma=1.2).unsqueeze(0).unsqueeze(0).to(vec.device)
    # filter /= torch.sum(filter)
    return F.conv1d(f_vec, filter)[0,0]


def compute_coherence_hist_distance(x, y, b=2**16):
    min_v, max_v = min(y.min(), x.min()), max(y.max(), x.max())
    y_bin_indices = torch.floor((b-1) * (y - min_v) / (max_v - min_v)).long()
    y_hist = torch.zeros(b)
    y_hist[y_bin_indices] = 1
    y_hist = blur_1d(y_hist)
    y_hist = torch.minimum(y_hist, torch.ones_like(y_hist))

    y_hist = y_hist.reshape(1,1,1,-1)

    grid = torch.zeros((1, 1, len(x), 2))
    x_bin_indices = 2 * (x - min_v) / (max_v - min_v) - 1
    grid[:,:,:,0] = x_bin_indices.reshape(1, 1, -1)
    loss = -F.grid_sample(y_hist,  grid, align_corners=False).sum()

    return loss

def compute_coherence_swd(input, target, num_proj):
    """Compute a Sliced Wasserstein distance between two equal size sets of vectors using num_proj projections"""
    rand = torch.randn(input.size(1), num_proj).to(input.device)  # (slice_size**2*ch)
    rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize

    # projection into (batch-zie, num_projections)
    projected_input = torch.matmul(input, rand)
    projected_target = torch.matmul(target, rand)

    loss = 0
    for j in range(num_proj):
        loss += compute_coherence_hist_distance(projected_input[:, j], projected_target[:, j])

    return loss / num_proj



class PatchSCD(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, num_proj=16, normalize_patch='none', batch_reduction='mean'):
        super(PatchSCD, self).__init__()
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
