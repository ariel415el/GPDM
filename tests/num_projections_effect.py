import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import distribution_metrics
from GPDM import GPDM
from utils import save_image

from tests.SIFID.sifid_score import calculate_sifid_given_paths
from tests.compute_diversity import compute_images_diversity
import numpy as np

def main(dataset_dir):
    image_paths = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]
    file_extension = image_paths[0].split(".")[-1]

    output_dir = os.path.join(f'outputs','num_projections_effect')
    os.makedirs(output_dir, exist_ok=True)
    f = open(os.path.join(output_dir, 'results.txt'), 'w+')

    num_steps = 300
    for num_proj in [1, 3, 4, 8, 16, 32, 64, 128, 256, 512]:
        criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=num_proj)
        model = GPDM(pyr_factor=0.85, coarse_dim=28, lr=0.05, num_steps=num_steps, init='noise', noise_sigma=1.5, resize=0)
        sfid_scores = []
        generated_dirs = []
        for i in range(3):
            generation_dir = os.path.join(output_dir, f'generated_images_p-{num_proj}_{i}')
            generated_dirs.append(generation_dir)
            for input_image_path in image_paths:
                result = model.run(input_image_path, criteria, None)
                save_image(result, os.path.join(generation_dir, os.path.basename(input_image_path)))

            sfid = np.asarray(calculate_sifid_given_paths(dataset_dir, generation_dir, 1, False, 64, file_extension)).mean()
            sfid_scores.append(sfid)

        diversity = compute_images_diversity(generated_dirs, dataset_dir)
        f.write(f"{num_proj}: SFID={np.mean(sfid_scores)}, Diversity={diversity}\n")


if __name__ == '__main__':
    main('images/SIGD16')