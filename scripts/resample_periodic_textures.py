import os

import distribution_metrics
from GPDM import GPDM
from utils import save_image

image_paths = [
        '../images/textures/olives.png',
        '../images/textures/tomatos.png',
        '../images/textures/green_waves.jpg',
        '../images/textures/cobbles.jpeg'
    ]

def main():
    criteria = distribution_metrics.PatchSWDLoss(patch_size=11, stride=1, num_proj=512, normalize_patch='none')
    model = GPDM(pyr_factor=0.75, n_scales=8, scale_factor=(2, 2), lr=0.01, num_steps=500, init='noise', noise_sigma=1.5, resize=256)
    output_dir = f'outputs/periodic_textures/{criteria.name}_{model.name}'
    for input_image_path in image_paths:
        fname, ext = os.path.splitext(os.path.basename(input_image_path))[:2]
        debug_dir = f'{output_dir}/debug/{fname}'
        result = model.run(input_image_path, criteria, debug_dir)

        save_image(result, f'{output_dir}/generated_images/{fname}.{ext}')

if __name__ == '__main__':
    main()