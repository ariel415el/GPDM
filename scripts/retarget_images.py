import os
import distribution_metrics
from GPDM import GPDM
from synthesis import synthesize_image
from utils import SyntesisConfigurations, get_file_name, save_image

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
    criteria = distribution_metrics.PatchSWDLoss(patch_size=11, stride=1, num_proj=512, normalize_patch='none')
    # criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=8, normalize_patch='mean')
    for input_image_path in images:
        for aspect_ratio in [(1, 0.5), (1, 1.5)]:
            for i in range(3):
                model = GPDM(pyr_factor=0.75, n_scales=7, aspect_ratio=aspect_ratio, lr=0.02, num_steps=400, init='blured_target', noise_sigma=1.5)
                # model = GPDM(pyr_factor=0.85, n_scales=13, aspect_ratio=aspect_ratio, lr=0.05, num_steps=150, init='blured_target', noise_sigma=1.5)

                debug_dir = f'outputs/retarget_images/{get_file_name(input_image_path)}/{criteria.name}_{model.name}'

                result = model.run(input_image_path, criteria, debug_dir)

                fname, ext = os.path.splitext(os.path.basename(input_image_path))[:2]
                save_image(result, f'outputs/retarget_images/generated_images/{fname}_{aspect_ratio}${i}{ext}')
if __name__ == '__main__':
    main()