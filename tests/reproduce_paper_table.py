import sys
import os
from time import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append((os.path.dirname(os.path.abspath(__file__))))

import distribution_metrics
from GPDM import GPDM
from utils import save_image

from tests.SIFID.sifid_score import calculate_sifid_given_paths
from tests.compute_diversity import compute_images_diversity
import numpy as np


def get_sifd_and_diversity(images_dir, output_dir, num_proj, num_steps, num_reps):
    image_paths = [os.path.join(images_dir, x) for x in os.listdir(images_dir)]
    file_extension = image_paths[0].split(".")[-1]

    run_times = []
    sfid_scores = []
    result_images_dirs = []
    criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=num_proj)
    model = GPDM(pyr_factor=0.85, coarse_dim=28, lr=0.05, num_steps=num_steps, init='noise', noise_sigma=1.5, resize=0)
    for i in range(num_reps):
        generation_dir = os.path.join(output_dir, f'generated_images_{i}')
        result_images_dirs.append(generation_dir)
        for j,input_image_path in enumerate(image_paths):
            print(f"Status: Dataset={os.path.basename(images_dir)}, iter={i}, im={j}/{len(image_paths)}")

            start = time()
            result = model.run(input_image_path, criteria, None)
            run_times.append(time() - start)

            save_image(result, os.path.join(generation_dir, os.path.basename(input_image_path)))

        sfid = np.asarray(calculate_sifid_given_paths(images_dir, generation_dir, 1, False, 64, file_extension)).mean()
        sfid_scores.append(sfid)

    sfid = np.mean(sfid_scores)
    diversity = compute_images_diversity(result_images_dirs, images_dir)
    runtime = np.mean(run_times)

    return sfid, diversity, runtime


if __name__ == '__main__':
    num_proj = 64
    num_steps = 300
    num_reps = 3
    output_dir = os.path.join(f'outputs', f'reshuffle_table_#Projs-{num_proj}_#Steps-{num_proj}_#Reps-{num_reps}')
    os.makedirs(output_dir, exist_ok=True)
    f = open(os.path.join(output_dir, f'table.txt'), 'w+')
    f.write("Dataset, SFID, Diversity, runtime\n")
    for dataset_dir in [
                        "images/SIGD16",
                        # "images/Places50"
                        ]:
        sfid, diversity, runtime = get_sifd_and_diversity(dataset_dir, output_dir, num_proj, num_steps, num_reps)
        f.write(f"{os.path.basename(dataset_dir)}, {sfid}, {diversity}, {runtime}")
