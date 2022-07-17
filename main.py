from os.path import join, basename, dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import GPDM
from patch_swd import PatchSWDLoss
from utils import load_image, dump_images, get_pyramid_scales
import argparse

# IO
parser = argparse.ArgumentParser(description='Run GPDM')
parser.add_argument('target_image', help="This image has the reference patch distribution to be matched")
parser.add_argument('--output_dir', default="Outputs", help="Where to put the results")
parser.add_argument('--debug_dir', default=None, help="If not None, debug images are dumped to this path")

# SWD parameters
parser.add_argument('--patch_size', type=int, default=7)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--num_proj', type=int, default=64, help="Number of random projections used to approximate SWD")

# Pyramids parameters
parser.add_argument('--fine_dim', type=int, default=None, help="Height of the largest ptramid scale (can be used to get smaller output)."
                                                     "If None use the target_image height")
parser.add_argument('--coarse_dim', type=int, default=21, help="Height of the smallest pyramid scale, ")
parser.add_argument('--pyr_factor', type=float, default=0.85, help="Downscale factor of the pyramid")
parser.add_argument('--AR_height', type=float, default=1., help="Controls the aspect ratio of the result: factor of height")
parser.add_argument('--AR_width', type=float, default=1., help="Controls the aspect ratio of the result: factor of width")

# GPDM parameters
parser.add_argument('--init_from', default='zeros', help="Defines the intial guess for the first level. Can one of ('zeros', 'target', '<path-to-image>')")
parser.add_argument('--lr', type=float, default=0.01, help="Adam learning rate for the optimization")
parser.add_argument('--num_steps', type=int, default=300, help="Number of Adam steps")
parser.add_argument('--noise_sigma', type=float, default=1.5, help="Std of noise added to the first initial image")
parser.add_argument('--num_images', type=int, default=1, help="If > 1, batched inference is used (see paper) and multiple images are generated")

args = parser.parse_args()


if __name__ == '__main__':
    refernce_images = load_image(args.target_image).repeat(args.num_images, 1, 1, 1)

    criteria = PatchSWDLoss(patch_size=args.patch_size, stride=args.stride, num_proj=args.num_proj)

    fine_dim = args.fine_dim if args.fine_dim is not None else refernce_images.shape[-2]

    new_iamges = GPDM.generate(refernce_images, criteria,
                               pyramid_scales=get_pyramid_scales(fine_dim, args.coarse_dim, args.pyr_factor),
                               aspect_ratio=(args.AR_height, args.AR_width),
                               init_from=args.init_from,
                               lr=args.lr,
                               num_steps=args.num_steps,
                               additive_noise_sigma=args.noise_sigma
    )
    dump_images(new_iamges, join(args.output_dir, basename(args.target_image)))