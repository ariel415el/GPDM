import os
import distribution_metrics
from synthesis import synthesize_image
from utils import SyntesisConfigurations, get_file_name

images = [
    # '../images/retargeting/mountains3.png',
    # '../images/SIGD16/birds_on_branch.jpg',
    # '../images/retargeting/fish.png',
    # '../images/retargeting/fruit.png',
    # '../images/retargeting/corn.png',
    # '../images/retargeting/mountins2.jpg',
    # '../images/retargeting/colusseum.png',
    # '../images/retargeting/mountains.jpg',
    # '../images/retargeting/SupremeCourt.jpeg',
    # '../images/retargeting/kanyon.jpg',
    '../images/retargeting/corn.png',
    # '../images/resampling/balloons.png',
    # '../images/resampling/birds.png',
    # '../images/retargeting/pinguins.png',
]

def main():
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    # criteria = distribution_metrics.PatchCoherentLoss(patch_size=3, stride=1)
    # criteria = distribution_metrics.PatchSWDLoss(patch_size=11, stride=1, num_proj=512, normalize_patch='none')
    # criteria = distribution_metrics.PatchMMD_RBF(patch_size=11, stride=1, normalize_patch='mean')
    criteria = distribution_metrics.PatchMMD_Inverse(patch_size=11, stride=1, normalize_patch='mean')
    # criteria = distribution_metrics.MMDApproximate(patch_size=11, strides=1, pool_size=-1, sigma=0.03, r=512, normalize_patch='mean')

    for input_image_path in images:
        for aspect_ratio in [(1, 0.5), (1, 1.5)]:
            for _ in range(1):
                conf = SyntesisConfigurations(pyr_factor=0.75, n_scales=7, aspect_ratio=aspect_ratio, lr=0.02, num_steps=500,
                                              init="blur", resize=256)

                outputs_dir = f'outputs/retarget_images/{get_file_name(input_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

                # content_loss = GrayLevelLoss(256)
                synthesize_image(input_image_path, criteria, None, conf, outputs_dir)

if __name__ == '__main__':
    main()