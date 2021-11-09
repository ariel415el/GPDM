import os

import distribution_metrics
from GPDM import GPDM
from utils import save_image, get_file_name

input_and_target_images = [
    ('../images/resampling/balls.jpg', '../images/edit_inputs/balls_green_and_black.jpg'),
    ('../images/resampling/birds.png', '../images/edit_inputs/birds_edit_1.jpg'),
    ('../images/resampling/birds.png', '../images/edit_inputs/birds_edit_2.png'),
    ('../images/SIGD16/birds_on_branch.jpg', '../images/edit_inputs/birds-on-tree_edit_1.jpg'),
    ('../images/retargeting/house.jpg', '../images/edit_inputs/house_edit2.jpg'),
    ('../images/retargeting/house.jpg', '../images/edit_inputs/house_edit1.jpg'),
    ('../images/resampling/balloons.png', '../images/edit_inputs/balloons_edit.jpg'),
    ('../images/image_editing/stone.png', '../images/edit_inputs/stone_edit.png'),
    ('../images/image_editing/tree.png', '../images/edit_inputs/tree_edit.png'),
    ('../images/retargeting/mountins2.jpg', '../images/edit_inputs/mountins2_edit.jpg'),
    ('../images/retargeting/mountins2.jpg', '../images/edit_inputs/mountins2_edit2.jpg'),
    ('../images/image_editing/swiming1.jpg', '../images/edit_inputs/swiming1_edit.jpg'),
]

def main():
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    criteria = distribution_metrics.PatchSWDLoss(patch_size=11, stride=1, num_proj=256)
    # criteria = distribution_metrics.PatchCoherentLoss(patch_size=7, stride=3, mode='detached')
    for (target_image_path, input_image_path) in input_and_target_images:
            model = GPDM(coarse_dim=64, pyr_factor=0.85, lr=0.03, num_steps=250, init=input_image_path, noise_sigma=0)
            fname, ext = os.path.splitext(os.path.basename(input_image_path))[:2]

            debug_dir = f'outputs/image_editing/{get_file_name(input_image_path)}-to-{get_file_name(target_image_path)}/{criteria.name}_{model.name}'
            result = model.run(target_image_path, criteria, debug_dir)
            # result = model.run(target_image_path, criteria, None)

            save_image(result, f'outputs/image_editing/generated_images/{fname}${0}{ext}')


if __name__ == '__main__':
    main()