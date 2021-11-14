import torch
import torch.nn.functional as F


class PatchSWDLoss(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, num_proj=256):
        super(PatchSWDLoss, self).__init__()
        self.name = f"ConvSWDLoss(p-{patch_size}:{stride})"
        self.patch_size = patch_size
        self.stride = stride
        self.num_proj = num_proj

    def forward(self, x, y):
        b, c, h, w = x.shape
        assert b == 1, "Batches not implemented"
        rand = torch.randn(self.num_proj, 3, self.patch_size, self.patch_size).to(x.device)  # (slice_size**2*ch)
        rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize

        projx = F.conv2d(x, rand).reshape(self.num_proj, -1)
        projy = F.conv2d(y, rand).reshape(self.num_proj, -1)

        # while projx.shape[1] > projy.shape[1]:

        if projx.shape[1] > projy.shape[1]:
            d = projx.shape[1] // projy.shape[1]
            projy = torch.cat([projy] * d, dim=1)
            if projx.shape[1] > projy.shape[1]:
                indices = torch.randperm(projy.shape[1])[:projx.shape[1] - projy.shape[1]]
                projy = torch.cat([projy, projy[:, indices]], dim=1)

        elif projx.shape[1] < projy.shape[1]:
            d = projy.shape[1] // projx.shape[1]
            projx = torch.cat([projx] * d, dim=1)
            if projx.shape[1] < projy.shape[1]:
                indices = torch.randperm(projx.shape[1])[:projy.shape[1] - projx.shape[1]]
                projx = torch.cat([projx, projx[:, indices]], dim=1)

        projx, _ = torch.sort(projx, dim=1)
        projy, _ = torch.sort(projy, dim=1)

        # loss = ((projx - projy) ** 2).mean()
        loss = torch.abs(projx - projy).mean()

        return loss

