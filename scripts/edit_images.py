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
    criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=256)

    for (target_image_path, input_image_path, coarse_dim) in input_and_target_images:
        model = GPDM(coarse_dim=coarse_dim, pyr_factor=0.85, lr=0.03, num_steps=300, init=input_image_path, noise_sigma=0, resize=256)

        result = model.run(target_image_path, criteria, debug_dir=None)
        output_path = os.path.join(out_dir, f'{get_file_name(input_image_path)}.png')
        save_image(result, output_path)


if __name__ == '__main__':
    input_and_target_images = [
        ('images/edit_inputs/tree.png', 'images/edit_inputs/tree_edit.png', 128),
        ('images/SIGD16/7.jpg', 'images/edit_inputs/balls_edit.jpg', 128),
        ('images/SIGD16/4.jpg', 'images/edit_inputs/birds_dusk_edit.jpg', 128),
        ('images/Places50/15.jpg', 'images/edit_inputs/stone_edit.png', 256),
        ('images/edit_inputs/swiming.jpg', 'images/edit_inputs/swiming_edit.jpg', 32),
        ('images/retargeting/teo.jpg', 'images/edit_inputs/teo_edit.jpg', 64),
    ]
    outputs_dir = f'outputs/edit_images'
    edit_images(input_and_target_images, outputs_dir)