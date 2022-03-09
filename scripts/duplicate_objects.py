import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import distribution_metrics
from GPDM import GPDM
from utils import save_image, get_file_name


def edit_images(input_and_target_images, out_dir):
    """
    Smaller patch size adds variablility but may ruin large objects
    """

    criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=256, mask_patches_factor=3)

    for (target_image_path, mask_image_path, coarse_dim) in input_and_target_images:
        model = GPDM(coarse_dim=coarse_dim, pyr_factor=0.85, lr=0.03, num_steps=300, init='noise', noise_sigma=0, resize=256)

        result = model.run(target_image_path, criteria, mask_img_path=mask_image_path, debug_dir=None)
        output_path = os.path.join(out_dir, f'{get_file_name(target_image_path)}.png')
        save_image(result, output_path)


if __name__ == '__main__':
    input_and_target_images = [
        ('images/retargeting/teo.jpg', 'images/edit_inputs/teo_mask.png', 28),
        ('images/edit_inputs/oranges.jpg', 'images/edit_inputs/oranges_mask.jpg', 28),
        ('images/edit_inputs/tree.png', 'images/edit_inputs/tree_mask.png', 28),
        ('images/manipulations/soccer2_org.jpg', 'images/manipulations/soccer2_mask.jpg', 28),
    ]
    outputs_dir = f'outputs/duplicate_images'
    edit_images(input_and_target_images, outputs_dir)