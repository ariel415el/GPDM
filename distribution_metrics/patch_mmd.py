import numpy as np
import torch

from distribution_metrics.common import extract_patches


def get_distance_matrix(X):
    XX = torch.matmul(X, X.t())

    X_norms = torch.sum(X ** 2, 1, keepdim=True)

    # exp[a,b] = (X[a] @ X[a])^2 -2(X[a] @ X[b]) + (X[b] @ X[b])^2 = || X[a] - X[b] ||^2
    return X_norms - 2 * XX + X_norms.t()


class MultiBandWitdhRbfKernel:
    def __init__(self, sigmas=None):
        self.name = '-MultiBandWitdhRbfKernel'
        if sigmas is None:
            self.sigmas = [2, 5, 10, 20, 40, 80]
        else:
            self.sigmas = sigmas

    def __call__(self, X, S, **kwargs):
        squared_l2_dist_mat = get_distance_matrix(X)
        loss = 0
        for s in self.sigmas:
            rbf_gram_matrix = torch.exp(squared_l2_dist_mat / (-2 * s ** 2))
            # rbf_gram_matrix = torch.exp(1.0 / v * squared_l2_dist_mat)
            loss += torch.sum(S * rbf_gram_matrix)
        # return torch.sqrt(loss)
        return loss


class RbfKernel:
    def __init__(self, sigma):
        self.name = '-RbfKernel'
        if sigma is None:
            self.sigma = [2, 5, 10, 20, 40, 80]
        else:
            self.sigma = sigma

    def __call__(self, X, S, **kwargs):
        rbf_gram_matrix = torch.exp(get_distance_matrix(X) / (-2 * self.sigma ** 2))
        loss = torch.sum(S * rbf_gram_matrix)
        return loss


class DotProductKernel:
    def __init__(self):
        self.name = '-DotProductKernel'

    def __call__(self, X, S, **kwargs):
        XX = torch.matmul(X, X.t())
        loss = torch.sum(S * XX)
        return loss


def get_scale_matrix(M, N):
    """
    return an (N+M)x(N+M) matrix where the the TL and BR NxN and MxM blocks are 1/N^2 and 1/M^2 equivalently
    and the other two blocks are -1/NM
    """
    s1 = (torch.ones((N, 1)) * 1.0 / N)
    s2 = (torch.ones((M, 1)) * -1.0 / M)
    s = torch.cat((s1, s2), 0)
    return torch.matmul(s, s.t())


def compute_MMD(x_patches, y_patches, kernel):
    """
    :param x_patches:
    :param y_patches:
    :param kernel: a function f: (M+N,3xpxp) -> (M+N, M+N) appliess a kernel for all ofh the (M+N)x(M+N) pairs of patches
    """
    # Compute signed scale matrix to sum up the right entries in the gram matrix for MMD loss
    M = x_patches.size()[0]
    N = y_patches.size()[0]
    S = get_scale_matrix(M, N).to(x_patches.device)
    all_patches = torch.cat((x_patches, y_patches), 0)
    # S[:N,:N] *= 0.95
    # S[N:,N:] *= 0.95
    # S[:N,N:] *= 1.25
    # S[N:,:N] *= 1.25
    return kernel(all_patches, S)


class PatchMMD(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, normalize_patch='none', batch_reduction='mean'):
        super(PatchMMD, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.batch_reduction = batch_reduction
        self.normalize_patch = normalize_patch
        self.name = f"PatchMMD{self.kernel.name if self.kernel else ''}(p-{patch_size}:{stride})"

    def forward(self, x, y):
        b, c, h, w = x.shape
        x_patches = extract_patches(x, self.patch_size, self.stride, self.normalize_patch)
        y_patches = extract_patches(y, self.patch_size, self.stride, self.normalize_patch)

        results = []
        for i in range(b):
            results.append(compute_MMD(x_patches[i], y_patches[i], self.kernel))

        results = torch.stack(results)

        if self.batch_reduction == 'mean':
            return results.mean()
        else:
            return results


class PatchMMDMultiRBF(PatchMMD):
    def __init__(self, patch_size=7, stride=1, batch_reduction='mean', normalize_patch='none', sigmas=None):
        if sigmas == None:
            sigmas = [0.1, 0.05, 0.025, 0.01]
        sigmas = np.array(sigmas) * patch_size ** 2
        self.kernel = MultiBandWitdhRbfKernel(sigmas)

        super(PatchMMDMultiRBF, self).__init__(patch_size=patch_size, stride=stride, normalize_patch=normalize_patch, batch_reduction=batch_reduction)


class PatchMMD_RBF(PatchMMD):
    def __init__(self, patch_size=7, stride=1, batch_reduction='mean', normalize_patch='none', sigma=None):
        if sigma == None:
            sigma = 0.05
        sigma *= patch_size ** 2
        self.kernel = RbfKernel(sigma)

        super(PatchMMD_RBF, self).__init__(patch_size=patch_size, stride=stride, normalize_patch=normalize_patch, batch_reduction=batch_reduction)

class PatchMMD_DotProd(PatchMMD):
    def __init__(self, patch_size=7, stride=1, batch_reduction='mean', normalize_patch='none'):
        self.kernel = DotProductKernel()

        super(PatchMMD_DotProd, self).__init__(patch_size=patch_size, stride=stride, normalize_patch=normalize_patch, batch_reduction=batch_reduction)



if __name__ == '__main__':
    from time import time
    import distribution_metrics
    for loss in [
        PatchMMD_RBF(patch_size=11, stride=5),
        distribution_metrics.MMDApproximate(patch_size=11, strides=5, r=128),
        distribution_metrics.PatchSWDLoss(patch_size=11, stride=5),
    ]:

        x = torch.randn((16, 3, 128, 128)).cuda()
        y = torch.ones(16, 3, 128, 128).cuda()
        print(loss(x, y))

        start = time()
        for i in range(10):
            loss(x, y)
        print(f"{loss.name}: {(time()-start)/10} s per inference")
