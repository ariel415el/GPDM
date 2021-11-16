import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import distribution_metrics
from GPDM import GPDM
from utils import save_image


input_and_target_images = [
    ('images/retargeting/teo.jpg', 'images/edit_inputs/teo_mask.png', 28),
    ('images/edit_inputs/oranges.jpg', 'images/edit_inputs/oranges_mask.jpg', 28),
]
def main():
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=128)
    model = GPDM(pyr_factor=0.85, coarse_dim=28, lr=0.05, num_steps=300, init='noise', noise_sigma=1.5, resize=0)
    output_dir = f'outputs/remove_objects/{criteria.name}_{model.name}'
    for (target_image_path, input_image_path, coarse_dim) in input_and_target_images:
        fname, ext = os.path.splitext(os.path.basename(target_image_path))[:2]

        debug_dir = f'{output_dir}/debug_images/{fname}'

        result = model.run(target_image_path, criteria, debug_dir, mask_img_path=input_image_path)

        save_image(result, f'{output_dir}/generated_images/{fname}{ext}')

if __name__ == '__main__':
    main()