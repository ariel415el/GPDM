import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import distribution_metrics
from GPDM import GPDM
from utils import save_image


def resample_textures(image_paths, out_dir, n_reps=1):
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
    image_paths = [
        'images/textures/olives.png',
        'images/textures/tomatos.png',
        'images/textures/green_waves.jpg',
        'images/textures/cobbles.jpeg',
        'images/style_transfer/style/brick.jpg',
        'images/style_transfer/style/mondrian.jpg',
        'images/style_transfer/style/rug.jpeg',
    ]

    out_dir = "outputs/resample_textures"
    resample_textures(image_paths, out_dir)
