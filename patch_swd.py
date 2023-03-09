import torch
import torch.nn.functional as F


class PatchSWDLoss(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, num_proj=256, c=3, l2=False):
        super(PatchSWDLoss, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.num_proj = num_proj
        self.l2 = l2
        self.c = c
        self.init()

    def init(self):
        # Sample random normalized projections
        rand = torch.randn(self.num_proj, self.c*self.patch_size**2) # (slice_size**2*ch)
        rand = rand / torch.norm(rand, dim=1, keepdim=True)  # noramlize to unit directions
        self.rand = rand.reshape(self.num_proj, self.c, self.patch_size, self.patch_size)

    def forward(self, x, y):
        self.rand = self.rand.to(x.device)
        # Project patches
        projx = F.conv2d(x, self.rand).transpose(1,0).reshape(self.num_proj, -1)
        projy = F.conv2d(y, self.rand).transpose(1,0).reshape(self.num_proj, -1)

        # Duplicate patches if number does not equal
        projx, projy = duplicate_to_match_lengths(projx, projy)

        # Sort and compute L1 loss
        projx, _ = torch.sort(projx, dim=1)
        projy, _ = torch.sort(projy, dim=1)

        if self.l2:
            loss = ((projx - projy)**2).mean()
        else:
            loss = torch.abs(projx - projy).mean()

        return loss


def duplicate_to_match_lengths(arr1, arr2):
    """
    Duplicates randomly selected entries from the smaller array to match its size to the bigger one
    :param arr1: (r, n) torch tensor
    :param arr2: (r, m) torch tensor
    :return: (r,max(n,m)) torch tensor
    """
    if arr1.shape[1] == arr2.shape[1]:
        return arr1, arr2
    elif arr1.shape[1] < arr2.shape[1]:
        tmp = arr1
        arr1 = arr2
        arr2 = tmp

    b = arr1.shape[1] // arr2.shape[1]
    arr2 = torch.cat([arr2] * b, dim=1)
    if arr1.shape[1] > arr2.shape[1]:
        indices = torch.randperm(arr2.shape[1])[:arr1.shape[1] - arr2.shape[1]]
        arr2 = torch.cat([arr2, arr2[:, indices]], dim=1)

    return arr1, arr2