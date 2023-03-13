import argparse
import os

import torch
import torchvision
from torchvision.utils import save_image

from debug_utils import dump_hists, plot_values
from models import DirectSWD, GMMSWD, GD_gradient_projector
from utils import load_image

from torchvision.transforms import Resize, CenterCrop, InterpolationMode


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



def dump_image(img, path):
    # from PIL import Image
    # img = to_256b_np(img)
    # Image.fromarray(img).save(path)
    save_image(img, path, normalize=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GPDM')
    parser.add_argument('reference', default='/cs/labs/yweiss/ariel1/data/afhq/train/wild/flickr_wild_000060.jpg')
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
    refernce_image = load_image(args.reference).to(device)
    if args.high_dim is not None:
        refernce_image = Resize(min(refernce_image.shape[-2], args.high_dim), antialias=True)(refernce_image)
    refernce_image = CenterCrop(min(refernce_image.shape[-2:]))(refernce_image)
    high_dim = refernce_image.shape[-2]

    operator = DownsampleOperator(down_factor=4, high_dim=high_dim, antialias=True, interpolation=InterpolationMode.BICUBIC)
    # operator = NoiseOperator(noise_sigma=0.25)


    output_dir = "outputs/" + os.path.splitext(os.path.basename(args.reference))[0] + f"_p-{args.p}_s-{args.s}"
    os.makedirs(output_dir, exist_ok=True)

    # Preprocess inputs
    corrupt_image = operator(refernce_image)
    initial_guess = operator.naive_reverse(corrupt_image)

    dump_image(refernce_image, f"{output_dir}/1-GT.png")
    dump_image(corrupt_image, f"{output_dir}/2-corrupt_image.png")
    dump_image(initial_guess, f"{output_dir}/2-initial_guess.png")


    gradient_projector = GD_gradient_projector(corrupt_image, operator)
    # Define models
    models = [
        DirectSWD(refernce_image, p=args.p, s=args.s, mode='Fixed', num_steps=args.num_steps, n_proj=args.n_proj),
        # DirectSWD(refernce_image, p=args.p, s=args.s, mode='Fixed', num_steps=args.num_steps, n_proj=args.n_proj, name='PGD',
        #           gradient_projector=gradient_projector),
        # GMMSWD(refernce_image, p=args.p, s=args.s, mode="Fixed", num_steps=args.num_steps, n_components=5, n_proj=args.n_proj, name="PGD",
        #        gradient_projector=None),
        #
        DirectSWD(refernce_image, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj),
        # DirectSWD(refernce_image, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj, name='PGD',
        #           gradient_projector=gradient_projector),
        #
        # DirectSWD(corrupt_image, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj, name="self-PGD",
        #           gradient_projector=gradient_projector),
        DirectSWD(corrupt_image, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj, name="self"),
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



