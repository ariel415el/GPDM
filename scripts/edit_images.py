import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import distribution_metrics
from GPDM import GPDM
from utils import save_image, get_file_name

input_and_target_images = [
    ('images/edit_inputs/tree.png', 'images/edit_inputs/tree_edit.png', 128),
    ('images/SIGD16/7.jpg', 'images/edit_inputs/balls_edit.jpg', 128),
    ('images/SIGD16/4.jpg', 'images/edit_inputs/birds_dusk_edit.jpg', 128),
    ('images/Places50/15.jpg', 'images/edit_inputs/stone_edit.png', 256),
    ('images/edit_inputs/swiming.jpg', 'images/edit_inputs/swiming_edit.jpg', 128),
    ('images/retargeting/teo.jpg', 'images/edit_inputs/teo_edit.jpg', 64),
]

def main():
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=256)
    # criteria = distribution_metrics.PatchCoherentLoss(patch_size=7, stride=3, mode='detached')
    outputs_dir = 'outputs/image_editing'
    for (target_image_path, input_image_path, coarse_dim) in input_and_target_images:
            model = GPDM(coarse_dim=coarse_dim, pyr_factor=0.85, lr=0.03, num_steps=250, init=input_image_path, noise_sigma=0, resize=0)
            fname, ext = os.path.splitext(os.path.basename(input_image_path))[:2]

            debug_dir = f'{outputs_dir}/debug_images/{get_file_name(input_image_path)}-to-{get_file_name(target_image_path)}/{criteria.name}_{model.name}'
            result = model.run(target_image_path, criteria, debug_dir)

            save_image(result, f'{outputs_dir}/generated_images/{fname}${0}{ext}')


if __name__ == '__main__':
    main()