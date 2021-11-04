from distribution_metrics.approximate_patch_mmd import MMDApproximate
from distribution_metrics.patch_swd import PatchSWDLoss
from distribution_metrics.patch_coherence_loss import PatchCoherentLoss
from distribution_metrics.old_patch_swd import OLDPatchSWDLoss
from distribution_metrics.sliced_coherence_loss import PatchSCD



# Test speed
# if __name__ == '__main__':
#     import torch
#     from time import time
#     x = torch.ones(1,3,512,512).cuda()
#     y = torch.ones(1,3,512,512).cuda()
#
#     for loss in [PatchSWDLoss().cuda(), ConvSWDLoss().cuda()]:
#         start = time()
#         for i in range(100):
#             loss(x, y)
#         print(f"{loss.name} took : {(time() - start) / 100}")