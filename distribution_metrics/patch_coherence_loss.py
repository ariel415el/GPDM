import torch
from distribution_metrics.common import extract_patches

def compute_patch_coherence(input_patches, target_patches, mode='l2'):
    if mode == 'l2':
        dist_matrix = ((input_patches[None, :] - target_patches[:, None])**2).mean(2) # (dist_matrix[i,j] = d(target_patches[i], input_patches[j])
    elif mode == 'batched_detached-l2':
        # compute dist_matrix in mini-batches to store NxM matrix instead of NxMxPsize
        dist_matrix = torch.zeros((len(input_patches), len(target_patches)), dtype=torch.float16).to(input_patches.device)
        b = 64
        n_batches = len(input_patches) // b
        for i in range(n_batches):
            dist_matrix[:, i*b:(i+1)*b] = ((input_patches[None, i*b:(i+1)*b].detach() - target_patches[:, None]) ** 2).mean(2)
        if len(input_patches) % b != 0:
            dist_matrix[:, n_batches*b:] = ((input_patches[None, n_batches*b:].detach() - target_patches[:, None]) ** 2).mean(2)
        min_indices = torch.min(dist_matrix, dim=0)[1]
        return ((input_patches - target_patches[min_indices])**2).mean()
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
    """For each patch in input image x find its NN in target y and sum the their distances"""
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

        results = []
        for i in range(b):
            results.append(compute_patch_coherence(x_patches[i], y_patches[i], self.mode))

        results = torch.stack(results)

        if self.batch_reduction == 'mean':
            return results.mean()
        else:
            return results

if __name__ == '__main__':
    input_image = torch.randn((1, 3,90,90)).cuda()
    target_image = torch.randn((1, 3,90,90)).cuda() * 2

    from time import time
    start = time()
    loss = PatchCoherentLoss(5, 1, 'batched_detached-l2').cuda()
    for i in range(10):
        loss(input_image, target_image)
    print(f"Time: {(time() - start) / 10}")
