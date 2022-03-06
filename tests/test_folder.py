import os

import numpy as np

from tests.SIFID.sifid_score import calculate_sifid_given_paths
from tests.compute_diversity import compute_images_diversity


def test(reference_dir, synthetic_root):
    all_synthetic_dirs = [os.path.join(synthetic_root, x) for x in os.listdir(synthetic_root)]

    diversity = compute_images_diversity(reference_dir, all_synthetic_dirs)
    scores = []
    for synthetic_dir in all_synthetic_dirs:
        sfid = calculate_sifid_given_paths(reference_dir, synthetic_dir, 1, False, 64)
        scores.append(sfid)

    print(f"SFID: {np.mean(scores)} +- {np.std(scores)}, Diversity: {diversity}")


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    reference_dir = '/home/ariel/university/GPDM/images/SIGD16'
    synthetic_root = '/home/ariel/university/Efficient-GPNN/scripts/outputs/reshuffle/SIGD16_alpha=1'
    test(reference_dir, synthetic_root)
    synthetic_root = '/home/ariel/university/Efficient-GPNN/scripts/outputs/reshuffle/SIGD16_alpha=0.005'
    test(reference_dir, synthetic_root)
    synthetic_root = '/home/ariel/university/Efficient-GPNN/scripts/outputs/reshuffle/SIGD16_faissIVF-50'
    test(reference_dir, synthetic_root)

    synthetic_root = '/home/ariel/university/GPDM/scripts/outputs/reshuffle/SIGD16_28-noise'
    test(reference_dir, synthetic_root)

    reference_dir = '/home/ariel/university/GPDM/images/Places50'
    synthetic_root = '/home/ariel/university/Efficient-GPNN/scripts/outputs/reshuffle/Places50_alpha=1'
    test(reference_dir, synthetic_root)
    synthetic_root = '/home/ariel/university/Efficient-GPNN/scripts/outputs/reshuffle/Places50_alpha=0.005'
    test(reference_dir, synthetic_root)
    synthetic_root = '/home/ariel/university/Efficient-GPNN/scripts/outputs/reshuffle/Places50_faissIVF-50'
    test(reference_dir, synthetic_root)

    synthetic_root = '/home/ariel/university/GPDM/scripts/outputs/reshuffle/Places50_28-noise'
    test(reference_dir, synthetic_root)
