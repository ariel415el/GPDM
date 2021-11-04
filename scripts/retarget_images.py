import os
import distribution_metrics
from GPDM import GPDM
from utils import get_file_name, save_image

images = [
    '../images/retargeting/mountains3.png',
    # '../images/SIGD16/birds_on_branch.jpg',
    '../images/retargeting/fish.png',
    '../images/retargeting/fruit.png',
    '../images/retargeting/corn.png',
    # '../images/retargeting/mountins2.jpg',
    # '../images/retargeting/colusseum.png',
    # '../images/retargeting/mountains.jpg',
    # '../images/retargeting/SupremeCourt.jpeg',
    # '../images/retargeting/kanyon.jpg',
    # '../images/retargeting/corn.png',
    # '../images/resampling/balloons.png',
    # '../images/resampling/birds.png',
    # '../images/retargeting/pinguins.png',
]

def main():
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    criteria = distribution_metrics.PatchSWDLoss(patch_size=11, stride=1, num_proj=128)
    # criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=8, normalize_patch='mean')
    for input_image_path in images:
        for scale_factor in [(1, 1.5)]: # (1, 0.5)
            for i in range(3):
                model = GPDM(coarse_dim=22, pyr_factor=0.75, scale_factor=scale_factor, lr=0.02, num_steps=400, init='blured_target', noise_sigma=1.5)
                # model = GPDM(pyr_factor=0.85, n_scales=13, scale_factor=scale_factor, lr=0.05, num_steps=150, init='blured_target', noise_sigma=1.5)

                debug_dir = f'outputs/retarget_images/debug_images/{get_file_name(input_image_path)}/{criteria.name}_{model.name}'

                result = model.run(input_image_path, criteria, debug_dir)

                fname, ext = os.path.splitext(os.path.basename(input_image_path))[:2]
                save_image(result, f'outputs/retarget_images/generated_images/{fname}_{scale_factor}${i}{ext}')
if __name__ == '__main__':
    main()