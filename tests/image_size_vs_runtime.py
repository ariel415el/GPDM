import sys
import os
from time import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import distribution_metrics
from GPDM import GPDM
from utils import save_image


def main():
    input_image_path = '/home/ariel/university/GPDM/images/HQ_16/lambs.jpg'
    output_dir = f'outputs/reshuffle_size-runtime'
    os.makedirs(output_dir, exist_ok=True)
    f = open(os.path.join(output_dir,'sfid.txt'), 'w+')
    num_proj = 128
    num_steps = 300

    criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=num_proj)
    for resize in [128,256,512, 1024]:
        model = GPDM(pyr_factor=0.85, coarse_dim=28, lr=0.05, num_steps=num_steps, init='noise', noise_sigma=1.5, resize=resize)

        start = time()
        result = model.run(input_image_path, criteria, None)
        runtime = time() - start

        save_image(result, os.path.join(output_dir, os.path.basename(input_image_path)))

        f.write(f"{resize}: {runtime}\n")

if __name__ == '__main__':
    main()