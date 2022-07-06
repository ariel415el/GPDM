import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from os.path import join, basename
import GPDM
from patch_swd import PatchSWDLoss
from utils import load_image, dump_images, get_pyramid_scales

if __name__ == '__main__':
    n_images = 5
    image_path = 'images/SIGD16/4.jpg'
    refernce_images = load_image(image_path).repeat(n_images, 1, 1, 1)

    criteria = PatchSWDLoss(patch_size=8, stride=1, num_proj=64)

    new_images = GPDM.generate(refernce_images, criteria,
                               pyramid_scales=get_pyramid_scales(refernce_images.shape[-2], 32, 16),
                               init_from="mean",
                               additive_noise_sigma=1.5
                              )

    dump_images(new_images, join("outputs", "reshuffle", basename(image_path)))