import os
import distribution_metrics
from retarget_image import retarget_image
from utils import SyntesisConfigurations, get_file_name

images = [
    '../images/retargeting/mountains3.png',
    '../images/SIGD16/birds_on_branch.jpg',
    '../images/retargeting/birds2.jpg',
    '../images/retargeting/fish.png',
    '../images/retargeting/fruit.png',
    '../images/retargeting/corn.png',
    '../images/retargeting/mountins2.jpg',
    '../images/retargeting/colusseum.png',
    '../images/retargeting/mountains.jpg',
    '../images/retargeting/SupremeCourt.jpeg',
    '../images/retargeting/kanyon.jpg',
    '../images/retargeting/corn.png',
    '../images/resampling/balloons.png',
    '../images/resampling/birds.png',
    '../images/retargeting/pinguins.png',
    '../images/resampling/birds.png',
]

def main():
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    # criteria = distribution_metrics.PatchCoherentLoss(patch_size=15, stride=5)
    criteria = distribution_metrics.PatchSWDLoss(patch_size=11, stride=1, num_proj=1024, normalize_patch='channel_mean')

    for input_image_path in images:
        for aspect_ratio in [(1, 1.5), (1, 0.4)]:
            for _ in range(2):
                conf = SyntesisConfigurations(pyr_factor=0.75, n_scales=5, aspect_ratio=aspect_ratio, lr=0.01, num_steps=200,
                                              init="blur", resize=256, blur_loss=0)

                outputs_dir = f'outputs/retarget_images/{get_file_name(input_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

                # content_loss = GrayLevelLoss(input_image_path, 256)
                retarget_image(input_image_path, criteria, None, conf, outputs_dir)

if __name__ == '__main__':
    main()