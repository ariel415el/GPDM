import os
import distribution_metrics
from synthesis import synthesize_image
from utils import SyntesisConfigurations, get_file_name


def main():
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    paintings_dir = '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/style/'

    criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=256, normalize_patch='mean')
    for img_name in os.listdir(paintings_dir):
        input_image_path = os.path.join(paintings_dir, img_name)

        for i in range(3):

            conf = SyntesisConfigurations(pyr_factor=0.75, n_scales=9, lr=0.05, num_steps=150, init='blur', resize=512)

            outputs_dir = f'outputs/resampling_paintings/{get_file_name(input_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

            synthesize_image(input_image_path, criteria, None, conf, outputs_dir)


if __name__ == '__main__':
    main()