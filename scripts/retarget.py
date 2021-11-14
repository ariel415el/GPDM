import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import distribution_metrics
from GPDM import GPDM
from utils import save_image

images = [
    'images/retargeting/fish.png',
    'images/retargeting/fruit.png',
    'images/retargeting/corn.png',
    'images/retargeting/pinguins.png',
    'images/retargeting/SupremeCourt.jpeg',
    'images/retargeting/mountains.jpg',
    'images/retargeting/jerusalem.jpg',
    'images/retargeting/teo.jpg',
    'images/retargeting/house.jpg',
    'images/Places50/50.jpg',

]
images += [os.path.join('images/SIGD16/', x) for x in os.listdir('images/SIGD16/')]

def main():
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    outputs_dir = 'outputs/retarget/'
    criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=256)
    for input_image_path in images:
        for scale_factor in [(2,0.5), (1,1.5), (1,2.5), (1,0.5)]:
            for i in range(2):
                model = GPDM(resize=256, coarse_dim=21, pyr_factor=0.85, scale_factor=scale_factor, lr=0.05, num_steps=150, decay_steps=150, init='target', noise_sigma=1.5)

                fname, ext = os.path.splitext(os.path.basename(input_image_path))[:2]
                debug_dir = f'{outputs_dir}/{criteria.name}_{model.name}/debug_images/{fname}-{scale_factor}'

                result = model.run(input_image_path, criteria, debug_dir)

                save_image(result, f'{outputs_dir}/{criteria.name}_{model.name}/generated_images/{fname}_{scale_factor}${i}{ext}')
if __name__ == '__main__':
    main()