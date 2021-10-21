import os
import distribution_metrics
from synthesis import synthesize_image
from utils import SyntesisConfigurations, get_file_name

images = [
    '../images/resampling/cows.png',
    '../images/resampling/balloons.png',
    '../images/resampling/people_on_the_beach.jpg',
    '../images/resampling/balls.jpg',
    '../images/resampling/birds.png',
    '../images/resampling/jerusalem2.jpg',
]

def main():
    """
    Smaller patch size adds variablility but may ruin large objects
    """

    criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=256, normalize_patch='mean')
    for input_image_path in images:
        for i in range(3):

            conf = SyntesisConfigurations(pyr_factor=0.75, n_scales=5, lr=0.05, num_steps=150, init='blur', resize=256)

            outputs_dir = f'outputs/resampling_const_background/{get_file_name(input_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

            synthesize_image(input_image_path, criteria, None, conf, outputs_dir)


if __name__ == '__main__':
    main()