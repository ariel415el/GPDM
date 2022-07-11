from os.path import join, basename, dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import GPDM
from patch_swd import PatchSWDLoss
from utils import load_image, dump_images, get_pyramid_scales

if __name__ == '__main__':
    n_images = 5
    image_path = 'images/SIGD16/4.jpg'
    refernce_images = load_image(image_path).repeat(n_images, 1, 1, 1)

    criteria = PatchSWDLoss(patch_size=8, stride=1, num_proj=64)

    new_iamges = GPDM.generate(refernce_images, criteria,
                               pyramid_scales=get_pyramid_scales(refernce_images.shape[-2], 32, 16),
                               aspect_ratio=(1, 2),
                               init_from="target",
                               additive_noise_sigma=0.01
                               )
    dump_images(new_iamges, join("outputs", "retarget", basename(image_path)))