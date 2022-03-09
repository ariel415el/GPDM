import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import distribution_metrics
from GPDM import GPDM
from utils import save_image

def reshuffle(dataset_dir, out_dir, n_reps=1):
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=256)
    model = GPDM(pyr_factor=0.85, coarse_dim=35, lr=0.05, num_steps=300, init='noise', noise_sigma=1.5, resize=0)

    image_paths = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]
    for input_image_path in image_paths:
        for i in range(n_reps):

            result = model.run(input_image_path, criteria, debug_dir=None)

            save_image(result, os.path.join(out_dir, f"{os.path.basename(dataset_dir)}_{model.name}", str(i), os.path.basename(input_image_path)))

if __name__ == '__main__':
    out_dir = "outputs/reshuffle"
    reshuffle('/home/ariel/university/GPDM/images/SIGD16', out_dir)
    reshuffle('/home/ariel/university/GPDM/images/Places50', out_dir)
    reshuffle('/home/ariel/university/GPDM/images/reshuffle', out_dir)
