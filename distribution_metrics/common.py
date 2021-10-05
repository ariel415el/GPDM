import torch


def extract_patches(x, patch_size, stride, normalize_patch):
    """Extract normalized patches from an image"""
    b, c, h, w = x.shape
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=stride)
    x_patches = unfold(x).transpose(1, 2).reshape(b, -1, 3, patch_size, patch_size)
    if normalize_patch != 'none':
        dims = (0, 1, 2, 3, 4) if normalize_patch == 'mean' else (0, 1, 3, 4)
        x_std, x_mean = torch.std_mean(x_patches, dim=dims, keepdim=True)
        x_patches = (x_patches - x_mean)
        if normalize_patch == 'normalize':
            x_patches /= (x_std + 1e-8)
    return x_patches.view(b, -1, 3 * patch_size ** 2)


