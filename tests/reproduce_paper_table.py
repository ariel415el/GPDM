import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from time import time
import GPDM
from patch_swd import PatchSWDLoss
from utils import load_image, dump_images, get_scales
from tests.sifid_score import compute_SIFID


def compute_diversity(refernce_images, generated_images):
    """
    :param refernce_images: (1, c, h, w)
    :param generated_images: (b, c, h, w)
    """
    gray_ref = torch.mean(refernce_images, dim=1)
    gray_images = torch.mean(generated_images, dim=1)
    diversity = torch.std(gray_images, dim=0).mean() / torch.std(gray_ref)

    return  diversity.item()


if __name__ == '__main__':
    dataset_path = "images/SIGD16"
    output_dir = "test_oputputs/SIGD16"
    num_repetitions = 50
    batch_size = 50
    os.makedirs(output_dir, exist_ok=True)
    f = open(os.path.join(output_dir, f'table.txt'), 'w+')
    f.write("Image, SFID, Diversity, images/sec\n")

    criteria = PatchSWDLoss(patch_size=7, stride=1, num_proj=64)

    sifid_scores = []
    diversities = []
    for img_name in os.listdir(dataset_path):
        refernce_image = load_image(os.path.join(dataset_path, img_name))
        img_name = os.path.basename(img_name)

        start = time()
        generated_images = []
        for n_batches in range(num_repetitions // batch_size):

            scales = get_scales(refernce_image.shape[-2], 28, 0.85)
            new_images = GPDM.generate(refernce_image.repeat(batch_size, 1, 1, 1), criteria, scales=scales, lr=0.05, additive_noise_sigma=1.5)

            generated_images.append(new_images)
        generated_images = torch.cat(generated_images, dim=0)
        dump_images(generated_images, os.path.join(output_dir, img_name))

        run_time = time() - start
        diversity = compute_diversity(refernce_image, generated_images)
        SIFID = compute_SIFID(refernce_image, generated_images, n_convblocks=0)
        diversities.append(diversity)
        sifid_scores.append(SIFID)
        f.write(f"{img_name}, {SIFID:.3f}, {diversity:.3f}, {len(generated_images)/run_time:.3f}\n")

    f.write(f"Avg, {np.mean(sifid_scores):.3f}, {np.mean(diversities):.3f}\n")
    f.close()