import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import distribution_metrics
from GPDM import GPDM
from utils import save_image

image_paths = [
        '../images/textures/olives.png',
        '../images/textures/tomatos.png',
        '../images/textures/green_waves.jpg',
        '../images/textures/cobbles.jpeg',
        '../images/style_transfer/style/brick.jpg',
        '../images/style_transfer/style/mondrian.jpg',
        '../images/style_transfer/style/rug.jpeg',

    ]

def main():
    criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=128)
    for input_image_path in image_paths:
        model = GPDM(pyr_factor=0.75, coarse_dim=32, resize=256, scale_factor=(2, 2), lr=0.03, num_steps=200, init='noise', noise_sigma=1.5, decay_steps=100)
        output_dir = f'outputs/periodic_textures/{criteria.name}_{model.name}'
        fname, ext = os.path.splitext(os.path.basename(input_image_path))[:2]
        debug_dir = f'{output_dir}/debug/{fname}'
        result = model.run(input_image_path, criteria, debug_dir)

        save_image(result, f'{output_dir}/generated_images/{fname}{ext}')

if __name__ == '__main__':
    main()