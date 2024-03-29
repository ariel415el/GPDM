import argparse
import os

import torch
from torchvision.utils import save_image
from torchvision.transforms import Resize, CenterCrop, InterpolationMode

from sr_utils.debug_utils import dump_hists, plot_values
from models import DirectSWD, GMMSWD, GD_gradient_projector, predefinedDirectSWD, LapSWD, MSSWD, TwoScalesSWD, \
    back_projector, WindowDirectSWD
from utils import load_image



class DownsampleOperator:
    def __init__(self, down_factor, high_dim, **resize_kwargs):
        self.down = Resize(high_dim // down_factor, **resize_kwargs)
        self.up = Resize(high_dim, **resize_kwargs)

    def __call__(self, x):
        return self.down(x)

    def naive_reverse(self, x):
        return self.up(x)


class NoiseOperator:
    def __init__(self, noise_sigma):
        self.noise_sigma = noise_sigma
    def __call__(self, x):
        return x + torch.randn_like(x) * self.noise_sigma

    def naive_reverse(self, x):
        return x


def dump_image(img, path, normalize=True):
    save_image(img, path, normalize=normalize)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GPDM')
    parser.add_argument('reference', default='/cs/labs/yweiss/ariel1/data/afhq/train/wild/flickr_wild_000060.jpg')
    parser.add_argument('--gray', default=False, action='store_true')
    parser.add_argument('--high_dim', type=int, default=None)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--n_proj', type=int, default=64)
    parser.add_argument('--p', type=int, default=5)
    parser.add_argument('--s', type=int, default=5)
    parser.add_argument('--projected_gradient', default=False, action='store_true')
    args = parser.parse_args()
    device = torch.device(args.device)

    # Parse inputs
    refernce_image = load_image(args.reference, args.gray).to(device)
    if args.high_dim is not None:
        refernce_image = Resize(min(refernce_image.shape[-2], args.high_dim), antialias=True)(refernce_image)
    refernce_image = CenterCrop(min(refernce_image.shape[-2:]))(refernce_image)
    high_dim = refernce_image.shape[-2]

    operator, op_name = DownsampleOperator(down_factor=4, high_dim=high_dim, antialias=True, interpolation=InterpolationMode.BILINEAR), "SRx4"
    # operator, op_name = NoiseOperator(noise_sigma=0.25), "Denoise"

    output_dir = "outputs/" + os.path.splitext(os.path.basename(args.reference))[0] + f"{op_name}_{high_dim}_p-{args.p}_s-{args.s}"
    os.makedirs(output_dir, exist_ok=True)

    # Preprocess inputs
    corrupt_image = operator(refernce_image)
    initial_guess = operator.naive_reverse(corrupt_image)

    dump_image(refernce_image, f"{output_dir}/1-GT.png")
    dump_image(corrupt_image, f"{output_dir}/2-corrupt_image.png")
    dump_image(initial_guess, f"{output_dir}/2-initial_guess.png")

    # Define models
    models = [
        # Fixe vs resample:
        # DirectSWD(refernce_image, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj),
        # DirectSWD(refernce_image, p=args.p, s=args.s, mode="Fixed", num_steps=args.num_steps, n_proj=args.n_proj),

        # # Patch sizes
        # DirectSWD(refernce_image, p=8, s=1, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj, lr=100),
        # DirectSWD(refernce_image, p=16, s=2, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj),
        # DirectSWD(refernce_image, p=32, s=1, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj),
        # MSSWD(refernce_image, ps=(3, 5, 7, 9, 11), s=1, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj, lr=50),

        # # Lap vs RGB
        # DirectSWD(refernce_image, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj),
        # LapSWD(refernce_image, initial_guess, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj),
        # LapSWD(refernce_image, initial_guess, p=5, s=1, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj),
        # LapSWD(refernce_image, initial_guess, p=5, s=1, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj,
        #        gradient_projector=GD_gradient_projector(corrupt_image, operator), name="PGD")

        # Windows
        # DirectSWD(refernce_image, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj),
        # WindowDirectSWD(refernce_image, w=128, ws=64, p=args.p, ps=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj),

        # PGD + PGD-reg
        # DirectSWD(refernce_image, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj),
        # DirectSWD(refernce_image, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj,
        #           gradient_projector=GD_gradient_projector(corrupt_image, operator), name="PGD"),
        # DirectSWD(refernce_image, p=5, s=1, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj,
        #           gradient_projector=GD_gradient_projector(corrupt_image, operator, reg_weight=1), name="PGD-reg1"),
        # DirectSWD(refernce_image, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj,
        #           gradient_projector=back_projector(corrupt_image, operator, n_steps=100), name="BackProject"),

        # # Self
        TwoScalesSWD(corrupt_image, scale_factor=4, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj, name="self-rescaled"),
        DirectSWD(corrupt_image, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj, name="self"),
        # DirectSWD(corrupt_image, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj, name="self-PGD",
        #           gradient_projector=gradient_projector),
        # DirectSWD(corrupt_image, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj, name="self-PGDreg",
        #           gradient_projector=GD_gradient_projector(corrupt_image, operator, reg_weight=1)),

        # Gabor
        # DirectSWD(refernce_image, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj),
        # predefinedDirectSWD(refernce_image, p=args.p, s=args.s, num_steps=args.num_steps, n_proj=args.n_proj, lr=0.005),

        # # Optimizer
        # DirectSWD(refernce_image, p=args.p, s=args.s, mode="Resample", num_steps=100, n_proj=args.n_proj, name="100",lr=50),
        # DirectSWD(refernce_image, p=args.p, s=args.s, mode="Resample", num_steps=1000, n_proj=args.n_proj, name="1000",lr=50),
        # DirectSWD(refernce_image, p=args.p, s=args.s, mode="Resample", num_steps=10000, n_proj=args.n_proj, name="10000", lr=50),

        # GMM
        # DirectSWD(refernce_image, p=args.p, s=args.s, mode='Fixed', num_steps=args.num_steps, n_proj=args.n_proj, name='PGD',
        #           gradient_projector=gradient_projector),
        # GMMSWD(refernce_image, p=args.p, s=args.s, mode="Fixed", num_steps=args.num_steps, n_components=5, n_proj=args.n_proj, name="PGD",
        #        gradient_projector=None),
        # MSSWD(refernce_image, ps=(32,64,128), s=4, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj),

    ]

    # Run
    output_images = dict()
    loss_lists = dict()
    constraint_lists = dict()
    for model in models:
        torch.cuda.empty_cache()
        output, losses = model.run(initial_guess.clone())

        loss_lists[model.name] = losses
        output_images[model.name] = output
        dump_image(output, f"{output_dir}/3-{model.name}.png")

    dump_hists(refernce_image, corrupt_image, output_images, f"{output_dir}/4-hists.png")
    plot_values(loss_lists, f"{output_dir}/5-loss.png")


