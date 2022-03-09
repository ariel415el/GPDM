import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import distribution_metrics
from GPDM import GPDM
from utils import save_image

criteria = distribution_metrics.PatchSWDLoss(patch_size=11, stride=1, num_proj=64)
model = GPDM(pyr_factor=0.85, coarse_dim=44, lr=0.03, num_steps=500, init='noise', noise_sigma=0.15, resize=1024,
             scale_factor=(1, 1))

def reshuffle_hq(image_paths, out_dir, n_reps=1):
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=256)
    model = GPDM(pyr_factor=0.85, coarse_dim=35, lr=0.05, num_steps=300, init='noise', noise_sigma=1.5, resize=0)

    for input_image_path in image_paths:
        for i in range(n_reps):

            result = model.run(input_image_path, criteria, debug_dir=None)

            save_image(result, os.path.join(out_dir, model.name, str(i), os.path.basename(input_image_path)))

if __name__ == '__main__':

    image_paths = []
    dataset_dir = '/home/ariel/university/GPDM/images/HQ_16'
    image_paths += [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]
    image_paths = [x for x in image_paths if ('rio' in x or 'lambs' in x or 'safari' in x)]

    out_dir = "outputs/reshuffle_HQ"
    reshuffle_hq(image_paths, out_dir)
