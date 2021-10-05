import torch
from torch.nn.functional import conv2d, pad
import numpy as np

# try:
#     import pykeops as pko
#     # pko.clean_pykeops()
#     from pykeops.torch import LazyTensor
# except:
# print("couldn't import pykeops")

pko = None


def get_reduction_fn(reduction):
    if reduction == "mean":
        return torch.mean
    elif reduction == "sum":
        return torch.sum
    elif reduction == "none":
        def no_reduce(tensor, *args, **kwargs):  # support the reduction API
            return tensor

        return no_reduce
    else:
        raise ValueError("Invalid reduction type")


class MMDApproximate(torch.nn.Module):
    def _get_w_b(self, n_channels):
        self.w = torch.randn(self.r, n_channels, self.ksize, self.ksize) / self.sigma
        if self.normalize_patch == 'channel_mean':
            self.w -= self.w.mean(dim=(2, 3), keepdim=True)
        elif self.normalize_patch == 'mean':
            self.w -= self.w.mean(dim=(1, 2, 3), keepdim=True)
        self.b = torch.rand(self.r) * (2 * np.pi)
        return self.w, self.b

    def __init__(self,
                 patch_size=3,
                 strides=1,
                 sigma=0.06,
                 r=1024,
                 pool_size=32,
                 pool_strides=16,
                 batch_reduction='mean',
                 spatial_reduction='mean',
                 pad_image=True,
                 normalize_patch='none',
                 name=None):
        super(MMDApproximate, self).__init__()
        self.r = r
        self.pool_size = pool_size
        if pool_size > 1:
            self.pool = torch.nn.AvgPool2d(kernel_size=pool_size, stride=pool_strides)
        elif pool_size == -1:
            def GAP(x):
                return torch.nn.functional.avg_pool2d(x, x.shape[-2:], 1, 0)
            self.pool = GAP
        else:
            def no_op(x):
                return x
            self.pool = no_op
        self.ksize = patch_size
        self.strides = strides
        self.w = None
        self.b = None
        self.normalize_patch = normalize_patch
        self.averaging_kernel = None
        self.batch_reduction = get_reduction_fn(batch_reduction)
        self.spatial_reduction = get_reduction_fn(spatial_reduction)
        self.sigma = sigma * patch_size ** 2  # sigma is defined per pixel

        self.padding = self.ksize // 2 if pad_image else 0

        self.name = f"MMD-prox(p={patch_size},win={pool_size}:{pool_strides},rf={r},s={sigma})" if name is None else name

    def get_activations(self, x):
        if self.padding > 0:
            x = pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        c = x.shape[1]
        if self.w is None:
            self._get_w_b(c)
        act_x = torch.cos(conv2d(x, self.w.to(x.device), self.b.to(x.device), self.strides))
        x_feats = self.pool(act_x)
        return x_feats

    def forward(self, x, y):
        if self.padding > 0:
            x = pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
            y = pad(y, (self.padding, self.padding, self.padding, self.padding), mode='reflect')

        c = x.shape[1]
        w, b = self._get_w_b(c)
        w = w.to(x.device)
        b = b.to(x.device)
        act_x = torch.cos(conv2d(x, w, b, self.strides))
        x_feats = self.pool(act_x)
        act_y = torch.cos(conv2d(y, w, b, self.strides))
        y_feats = self.pool(act_y)
        distance = self.spatial_reduction((x_feats - y_feats).pow(2).mean(dim=1, keepdim=True), dim=(1, 2, 3))

        return self.batch_reduction(distance, dim=0)


class GlobalPoolDistance(torch.nn.Module):
    def __init__(self, patch_size=3, strides=1, rbf_sigma=0.06, batch_reduction='mean', use_keops=False,
                 normalize_patch='none'):
        super().__init__()
        self.patch_size = patch_size
        if normalize_patch == 'none':
            self.x_unfolder = torch.nn.Unfold(kernel_size=patch_size, stride=strides)
            self.y_unfolder = torch.nn.Unfold(kernel_size=patch_size, stride=strides)
        elif normalize_patch in ['channel_mean', 'mean']:
            x_unfolder = torch.nn.Unfold(kernel_size=patch_size, stride=strides)
            y_unfolder = torch.nn.Unfold(kernel_size=patch_size, stride=strides)
            mean_dims = (2, 3) if normalize_patch == 'channel_mean' else (1, 2, 3)

            def unfold_without_mean(base_unfolder):
                def new_unfolder(x):
                    bs, c, h, w = x.shape
                    unfolded = base_unfolder(x).reshape(bs, c, patch_size, patch_size, -1)
                    unfolded = unfolded - unfolded.mean(dim=mean_dims, keepdims=True)
                    return unfolded.reshape(bs, c * patch_size * patch_size, -1)

                return new_unfolder

            self.x_unfolder = unfold_without_mean(x_unfolder)
            self.y_unfolder = unfold_without_mean(y_unfolder)
        self.rbf_sigma = rbf_sigma * (patch_size ** 2)  # sigma is defined per pixel
        self.batch_reduction = get_reduction_fn(batch_reduction)
        self.pad = torch.nn.ReflectionPad2d(patch_size // 2)
        self.use_keops = use_keops
        self.normalize_patch = normalize_patch

    def _pooled_distance_kpo(self, x, y):
        x_i = self.x_unfolder(x).permute(dims=(0, 2, 1)).unsqueeze(dim=-2).contiguous()
        x_i = LazyTensor(x_i)
        d = x_i.shape[1]
        y_j = LazyTensor(self.y_unfolder(y).permute(dims=(0, 2, 1)).unsqueeze(dim=-3).contiguous())
        D_ij = ((x_i - y_j) ** 2).sum(dim=-1)
        K_ij = ((-1 / self.rbf_sigma ** 2) * D_ij).exp()
        K_ij = K_ij.sum(dim=1) / d
        return K_ij.mean(dim=1)

    def _pooled_distance_unnormalized(self, x, y):
        x_uf = self.x_unfolder(x)
        y_uf = self.y_unfolder(y)
        xy_diffs = (x_uf.unsqueeze(dim=-1) - y_uf.unsqueeze(dim=-2)).pow(2).sum(dim=1)  # [bs, p_dim, L1, L2]
        xy_k = (xy_diffs / (-1 * self.rbf_sigma ** 2)).exp()
        return xy_k

    def forward(self, x, y):
        if self.use_keops:
            batch_loss = -2 * self._pooled_distance_kpo(x, y) + \
                         self._pooled_distance_kpo(x, x) + \
                         self._pooled_distance_kpo(y, y)
        else:
            batch_loss = -2 * self._pooled_distance_unnormalized(x, y).mean(dim=(1, 2)) + \
                         self._pooled_distance_unnormalized(x, x).mean(dim=(1, 2)) + \
                         self._pooled_distance_unnormalized(y, y).mean(dim=(1, 2))

        return self.batch_reduction(batch_loss)


class MMDExact(torch.nn.Module):
    def __init__(self, window_size=32, window_stride=16, batch_reduction='mean', pad=True, **pooling_kwargs):
        super().__init__()
        self.name = 'MMDExact'
        self.window_stride = window_stride

        self.pooler = GlobalPoolDistance(**pooling_kwargs, batch_reduction='none')

        self.window_size = window_size if not pad else window_size + (self.pooler.patch_size // 2) * 2
        self.window_unfold = torch.nn.Unfold(self.window_size, stride=self.window_stride)
        self.batch_reduction = get_reduction_fn(batch_reduction)
        self.pad = pad

    def forward(self, x, y):
        c = x.shape[1]
        x_bs = x.shape[0]
        y_bs = y.shape[0]
        if x_bs != y_bs:
            if x_bs == 1:
                x = x.repeat(y_bs, 1, 1, 1)
                x_bs = y_bs
            else:
                y = y.repeat(x_bs, 1, 1, 1)
                y_bs = x_bs
        h, w = x.shape[2:]
        p = torch.nn.ReflectionPad2d(self.pooler.patch_size // 2)
        x = p(x)
        y = p(y)

        window_shape = (c, self.window_size, self.window_size)
        x_uf = self.window_unfold(x).permute(dims=(0, 2, 1)).reshape(x_bs, -1, *window_shape)
        y_uf = self.window_unfold(y).permute(dims=(0, 2, 1)).reshape(y_bs, -1, *window_shape)
        L = x_uf.shape[1]
        x_uf = x_uf.reshape(x_bs * L, *window_shape)
        y_uf = y_uf.reshape(y_bs * L, *window_shape)
        total_loss = self.pooler(x_uf, y_uf).view(x_bs, L).mean(dim=1)
        return self.batch_reduction(total_loss)


if __name__ == '__main__':
    x = torch.ones(2, 3, 64, 64)
    y = torch.zeros(2, 3, 64, 64)
    loss = MMDExact()

    print(loss(x, y))
