from os.path import join, basename, dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import GPDM
from patch_swd import PatchSWDLoss
from utils import load_image, dump_images, get_pyramid_scales

if __name__ == '__main__':
    n_images = 10
    image_path = 'images/SIGD16/16.jpg'
    # image_path = 'images/Places50/50.jpg'
    refernce_images = load_image(image_path).repeat(n_images, 1, 1, 1)

    criteria = PatchSWDLoss(patch_size=7, stride=1, num_proj=64)

    new_images = GPDM.generate(refernce_images, criteria,
                               pyramid_scales=get_pyramid_scales(refernce_images.shape[-2], 21, 0.85),
                               init_from="zeros",
                               additive_noise_sigma=1.5,
                               lr=0.05,
                              )

    dump_images(new_images, join("outputs", "reshuffle", basename(image_path)))