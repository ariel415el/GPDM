import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from os.path import join, basename
import GPDM
from patch_swd import PatchSWDLoss
from utils import load_image, dump_images

if __name__ == '__main__':
    n_images = 1  # does not make much sense to generate more than 1 but if so increase the additive noise to get different iamges
    style_image_path = 'images/style_transfer/style/thick_oil.jpg'
    style_image = load_image(style_image_path).repeat(n_images, 1, 1, 1)
    content_image_path = 'images/style_transfer/content/hillary1.jpg'

    criteria = PatchSWDLoss(patch_size=11, stride=1, num_proj=64)
    new_iamges = GPDM.generate(style_image,
                               criteria,
                               scales=[1024],
                               init_from=content_image_path,
                               additive_noise_sigma=0.05,
                               debug_dir=None)

    dump_images(new_iamges, join("outputs", "style-transfer", basename(content_image_path) + "-to-" + basename(style_image_path)))