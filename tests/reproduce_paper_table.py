import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import distribution_metrics
from GPDM import GPDM
from utils import save_image

from tests.SIFID.sifid_score import calculate_sifid_given_paths
from tests.compute_diversity import compute_images_diversity
import numpy as np


def get_sifd_and_diversity(images_dir, output_dir):
    image_paths = [os.path.join(images_dir, x) for x in os.listdir(images_dir)]
    file_extension = image_paths[0].split(".")[-1]

    sfid_scores = []
    result_images_dirs = []
    num_steps = 150
    num_proj = 256
    criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=num_proj)
    model = GPDM(pyr_factor=0.85, coarse_dim=28, lr=0.05, num_steps=num_steps, init='noise', noise_sigma=1.5, resize=0)
    for i in range(2):
        generation_dir = os.path.join(output_dir, f'generated_images_{i}')
        result_images_dirs.append(generation_dir)
        for input_image_path in image_paths:
            result = model.run(input_image_path, criteria, None)
            save_image(result, os.path.join(generation_dir, os.path.basename(input_image_path)))

        sfid = np.asarray(calculate_sifid_given_paths(images_dir, generation_dir, 1, False, 64, file_extension)).mean()
        sfid_scores.append(sfid)

    sfid = np.mean(sfid_scores)
    diversity = compute_images_diversity(result_images_dirs, images_dir)

    return sfid, diversity


if __name__ == '__main__':
    output_dir = os.path.join(f'outputs', 'reshuffle_table')
    f = open(os.path.join(output_dir,'table.txt'), 'w+')
    f.write("Dataset, SFID, Diversity")
    for dataset_dir in ["images/SIGD16", "images/Places50"]:
        sfid, diversity = get_sifd_and_diversity(dataset_dir, output_dir)
        f.write(f"{os.path.basename(dataset_dir)}, {sfid}, {diversity}")
