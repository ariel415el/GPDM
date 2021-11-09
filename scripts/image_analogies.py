import os
import distribution_metrics
from synthesis import synthesize_image
from utils import SyntesisConfigurations, get_file_name

input_and_target_images = [
    ('../images/analogies/duck_mosaic.jpg', '../images/analogies/S_char.jpg'),
    ('../images/analogies/S_char.jpg', '../images/analogies/duck_mosaic.jpg'),
    ('../images/analogies/kanyon2.jpg', '../images/analogies/tower.jpg'),
    ('../images/analogies/tower.jpg', '../images/analogies/kanyon2.jpg'),
    ('../images/analogies/maps_a.png', '../images/analogies/maps_b.png'),
]

def main():
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    criteria = distribution_metrics.PatchCoherentLoss(patch_size=3, stride=1, mode='batched_detached-l2')
    # criteria = distribution_metrics.PatchSWDLoss(patch_size=3, stride=1, num_proj=256, normalize_patch='none')
    conf = SyntesisConfigurations(n_scales=0, lr=0.05, num_steps=30, resize=512)
    for style_image_path, content_image_path in input_and_target_images:
        conf.init = content_image_path

        outputs_dir = f'outputs_old/image_analogies/{get_file_name(content_image_path)}-to-{get_file_name(style_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

        # content_loss = GrayLevelLoss(content_image_path, 32)
        content_loss = None
        synthesize_image(style_image_path, criteria, content_loss, conf, outputs_dir)


if __name__ == '__main__':
    main()