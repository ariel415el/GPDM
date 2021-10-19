import distribution_metrics
from retarget_image import retarget_image
from utils import SyntesisConfigurations, get_file_name

input_and_target_images = [
    # ('../images/resampling/balls.jpg', '../images/edit_inputs/balls_green_and_black.jpg'),
    # ('../images/resampling/birds.png', '../images/edit_inputs/birds_edit_1.jpg'),
    # ('../images/resampling/birds.png', '../images/edit_inputs/birds_edit_2.png'),
    # ('../images/retargeting/birds2.jpg', '../images/edit_inputs/birds-on-tree_edit_1.jpg'),
    ('../images/retargeting/house.jpg', '../images/edit_inputs/house_edit1.jpg'),
    # ('../images/retargeting/house.jpg', '../images/edit_inputs/house_edit1.jpg'),
    # ('../images/resampling/balloons.png', '../images/edit_inputs/balloons_edit.jpg'),
    # ('../images/image_editing/stone.png', '../images/edit_inputs/stone_edit.png'),
    # ('../images/image_editing/tree.png', '../images/edit_inputs/tree_edit.png'),
    # ('../images/image_editing/swiming1.jpg', '../images/edit_inputs/swiming1_edit.jpg'),
    # ('../images/retargeting/mountins2.jpg', '../images/edit_inputs/mountins2_edit.jpg'),
    # ('../images/retargeting/mountins2.jpg', '../images/edit_inputs/mountins2_edit2.jpg'),
]

def main():
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    # criteria = distribution_metrics.PatchSWDLoss(patch_size=11, stride=1, num_proj=256, normalize_patch='none')
    criteria = distribution_metrics.PatchCoherentLoss(patch_size=3, stride=1)
    for (style_image_path, content_image_path) in input_and_target_images:

            conf = SyntesisConfigurations(n_scales=0, pyr_factor=0.75, lr=0.05, num_steps=500, init=content_image_path, resize=128)

            outputs_dir = f'outputs/image_editing/{get_file_name(content_image_path)}-to-{get_file_name(style_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

            # content_loss = GrayLevelLoss(content_image_path, 256)
            retarget_image(style_image_path, criteria, None, conf, outputs_dir)


if __name__ == '__main__':
    main()