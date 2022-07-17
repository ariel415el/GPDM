from os.path import join, basename, dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import GPDM
from patch_swd import PatchSWDLoss
from utils import load_image, dump_images, get_pyramid_scales

if __name__ == '__main__':
    n_images = 1  # does not make much sense to generate more than 1 but if so increase the additive noise to get different iamges
    style_image_path = 'data/images/style_transfer/style/thick_oil.jpg'
    style_image = load_image(style_image_path).repeat(n_images, 1, 1, 1)
    content_image_path = 'data/images/style_transfer/content/hillary1.jpg'

    criteria = PatchSWDLoss(patch_size=7, stride=1, num_proj=64)
    new_iamges = GPDM.generate(style_image,
                               criteria,
                               pyramid_scales=get_pyramid_scales(800, 800, 0.75),
                               init_from=content_image_path,
                               lr=0.01)

    dump_images(new_iamges, join("outputs", "style-transfer", basename(content_image_path) + "-to-" + basename(style_image_path)))