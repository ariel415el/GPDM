import torch
import torch.nn.functional as F


def duplicate_to_match_lengths(arr1, arr2):
    """
    Duplicates entries of the smaller array to match its size to the bigger one
    :param arr1: (r, n) torch tensor
    :param arr2: (r, m) torch tensor
    :return: (r,max(n,m)) torch tensor
    """
    if len(arr1) == len(arr2):
        return arr1, arr2
    elif len(arr1) < len(arr2):
        tmp = arr1
        arr1 = arr2
        arr2 = tmp

    b = arr1.shape[1] // arr2.shape[1]
    arr2 = torch.cat([arr2] * b, dim=1)
    if arr1.shape[1] > arr2.shape[1]:
        indices = torch.randperm(arr2.shape[1])[:arr1.shape[1] - arr2.shape[1]]
        arr2 = torch.cat([arr2, arr2[:, indices]], dim=1)

    return arr1, arr2

def extract_patches(x, patch_size, stride):
    """Extract normalized patches from an image"""
    b, c, h, w = x.shape
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=stride)
    x_patches = unfold(x).transpose(1, 2).reshape(b, -1, 3, patch_size, patch_size)
    return x_patches.view(b, -1, 3 * patch_size ** 2)

class PatchSWDLoss(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, num_proj=256, use_convs=True):
        super(PatchSWDLoss, self).__init__()
        self.name = f"ConvSWDLoss(p-{patch_size}:{stride})"
        self.patch_size = patch_size
        self.stride = stride
        self.num_proj = num_proj
        self.use_convs = use_convs

    def forward(self, x, y):
        b, c, h, w = x.shape
        assert b == 1, "Batches not implemented"
        rand = torch.randn(self.num_proj, 3, self.patch_size, self.patch_size).to(x.device) # (slice_size**2*ch)
        if self.num_proj > 1:
            rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize

        if self.use_convs:
            projx = F.conv2d(x, rand).reshape(self.num_proj, -1)
            projy = F.conv2d(y, rand).reshape(self.num_proj, -1)
        else:
            projx = torch.matmul(extract_patches(x, self.patch_size, self.stride), rand)
            projy = torch.matmul(extract_patches(y, self.patch_size, self.stride), rand)

        projx, projy = duplicate_to_match_lengths(projx, projy)

        projx, _ = torch.sort(projx, dim=1)
        projy, _ = torch.sort(projy, dim=1)

        loss = torch.abs(projx - projy).mean()

        return loss

