import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from time import time
import GPDM
from patch_swd import PatchSWDLoss
from utils import load_image, dump_images, get_pyramid_scales
from tests.sifid_score import compute_SIFID


def compute_pixel_diversity(refernce_images, generated_images):
    """
    To quantify the diversity of the generated images, for each training example we calculated the standard
    deviation of the intensity values of each pixel over N generated images, averaged it over all pixels and normalize
    it by the std of the intensity values of the reference image.

    :param refernce_images: (1, c, h, w)
    :param generated_images: (b, c, h, w)
    """
    gray_ref = torch.mean(refernce_images, dim=1)
    gray_images = torch.mean(generated_images, dim=1)
    diversity = torch.std(gray_images, dim=0).mean() / torch.std(gray_ref)

    return  diversity.item()


if __name__ == '__main__':
    dataset_name = "Places50"
    dataset_path = f"images/{dataset_name}"
    n_images = 50
    batch_size = 1  # Generate <n_images> images in batches. should divide num_repetitions
    patch_size=7
    min_dim=21
    lr=0.05
    pyr_factor=0.85
    criteria=PatchSWDLoss(patch_size=patch_size, stride=1, num_proj=64)

    output_dir = f"test_oputputs/{dataset_name}_p={patch_size}_lr={lr}_min={min_dim}_bs={batch_size}"
    os.makedirs(output_dir, exist_ok=True)
    f = open(os.path.join(output_dir, f'table.txt'), 'w+')
    f.write("Image, SFID, Diversity, images/sec\n")

    sifid_scores = []
    diversities = []
    for img_name in os.listdir(dataset_path):
        refernce_image = load_image(os.path.join(dataset_path, img_name))
        img_name = os.path.basename(img_name)

        generated_images = []
        start = time()
        for n_batches in range(n_images // batch_size):

            new_images = GPDM.generate(refernce_image.repeat(batch_size, 1, 1, 1),
                                       criteria=criteria,
                                       pyramid_scales=get_pyramid_scales(refernce_image.shape[-2], min_dim, pyr_factor),
                                       lr=lr,
                                       additive_noise_sigma=0.75)

            generated_images.append(new_images)
        run_time = time() - start
        generated_images = torch.cat(generated_images, dim=0)

        # Write generated images
        dump_images(generated_images, os.path.join(output_dir, img_name))

        # Compute and write SIFID and diversity
        diversity = compute_pixel_diversity(refernce_image, generated_images)
        diversities.append(diversity)
        SIFID = compute_SIFID(refernce_image, generated_images, n_convblocks=1)
        sifid_scores.append(SIFID)
        f.write(f"{img_name}, {SIFID:.3f}, {diversity:.3f}, {len(generated_images)/run_time:.3f}\n")

    f.write(f"Avg, {np.mean(sifid_scores):.3f}, {np.mean(diversities):.3f}, None\n")
    f.close()
