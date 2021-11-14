import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import distribution_metrics
from GPDM import GPDM
from utils import save_image

image_paths = []
dataset_dir = '/home/ariel/university/GPDM/images/HQ_16'
image_paths += [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]


def main():
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    criteria = distribution_metrics.PatchSWDLoss(patch_size=11, stride=1, num_proj=64)
    model = GPDM(pyr_factor=0.85, coarse_dim=44, lr=0.03, num_steps=500, init='noise', noise_sigma=0.15, resize=1024, scale_factor=(1,1))
    output_dir = f'outputs/reshuffle/{os.path.basename(dataset_dir)}_{criteria.name}_{model.name}'
    for input_image_path in image_paths:
        for i in range(1):
            fname, ext = os.path.splitext(os.path.basename(input_image_path))[:2]

            debug_dir = f'{output_dir}/debug_images/{fname}-{i}'

            result = model.run(input_image_path, criteria, debug_dir)

            save_image(result, f'{output_dir}/generated_images/{fname}${i}{ext}')

if __name__ == '__main__':
    main()