import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import distribution_metrics
from GPDM import GPDM
from utils import save_image


def retarget(image_paths, out_dir, n_reps=1):
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=256)

    for input_image_path in image_paths:
        for scale_factor in [(1, 2), (1, 0.5), (2, 0.5)]:
            for i in range(n_reps):
                model = GPDM(resize=0, coarse_dim=28, pyr_factor=0.75, scale_factor=scale_factor, lr=0.05, num_steps=300, init='target', noise_sigma=1.5)

                result = model.run(input_image_path, criteria, debug_dir=None)

                save_image(result, os.path.join(out_dir, model.name, str(i), str(scale_factor) + os.path.basename(input_image_path)))


if __name__ == '__main__':
    image_pats = [
        'images/retargeting/fish.png',
        'images/retargeting/fruit.png',
        'images/retargeting/corn.png',
        'images/retargeting/pinguins.png',
        'images/retargeting/SupremeCourt.jpeg',
        'images/retargeting/mountains.jpg',
        'images/retargeting/jerusalem.jpg',
        'images/retargeting/teo.jpg',
        'images/retargeting/house.jpg',
        'images/Places50/50.jpg',
    ]
    out_dir = "outputs/retarget"
    retarget(image_pats, out_dir, 1)
