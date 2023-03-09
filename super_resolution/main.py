import argparse
import os

import torch
from torchvision.utils import save_image

from debug_utils import dump_hists, plot_values
from models import DirectSWD, GMMSWD, gradient_projector
from utils import load_image

from torchvision.transforms import Resize, CenterCrop, InterpolationMode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GPDM')
    parser.add_argument('reference', default='/cs/labs/yweiss/ariel1/data/afhq/train/wild/flickr_wild_000060.jpg')
    parser.add_argument('--high_dim', type=int, default=512)
    parser.add_argument('--down_factor', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--n_proj', type=int, default=64)
    parser.add_argument('--p', type=int, default=5)
    parser.add_argument('--s', type=int, default=5)
    parser.add_argument('--l2', default=False, action='store_true')
    parser.add_argument('--projected_gradient', default=False, action='store_true')
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # resize_kwargs = {"antialias": False, "interpolation": InterpolationMode.NEAREST}
    resize_kwargs = {"antialias": True, "interpolation": InterpolationMode.BILINEAR}

    # Parse inputs
    refernce_image = load_image(args.reference).to(device)
    high_dim = min(refernce_image.shape[-2], args.high_dim)
    low_dim = high_dim // args.down_factor

    output_dir = "outputs/" + os.path.splitext(os.path.basename(os.path.basename(args.reference)))[0] + f"_{low_dim}->{high_dim}_p-{args.p}_s-{args.s}"
    os.makedirs(output_dir, exist_ok=True)

    # Preprocess inputs
    refernce_image = CenterCrop(min(refernce_image.shape[-2:]))(refernce_image)
    refernce_image = Resize(high_dim, **resize_kwargs)(refernce_image)
    low_res = Resize(low_dim, **resize_kwargs)(refernce_image)
    low_res_big = Resize(refernce_image.shape[-2:], **resize_kwargs)(low_res)

    save_image(refernce_image, f"{output_dir}/1-GT.png", normalize=True)
    save_image(low_res, f"{output_dir}/2-low_res.png", normalize=True)
    save_image(low_res_big, f"{output_dir}/2-low_res_big.png", normalize=True)

    # Define models
    models = [
        DirectSWD(refernce_image, p=args.p, s=args.s, mode='1-Fixed-PGD', num_steps=args.num_steps, n_proj=args.n_proj, gradient_projector=gradient_projector(low_dim, high_dim, resize_kwargs)),
        GMMSWD(refernce_image, p=args.p, s=args.s, mode="2-Fixed-PGD", num_steps=args.num_steps, n_components=6, n_proj=args.n_proj, gradient_projector=gradient_projector(low_dim, high_dim, resize_kwargs)),
        DirectSWD(refernce_image, p=args.p, s=args.s, mode="3-Resample-PGD", num_steps=args.num_steps, n_proj=args.n_proj, gradient_projector=gradient_projector(low_dim, high_dim, resize_kwargs)),
        DirectSWD(low_res, p=args.p, s=args.s, mode="4-Resample", num_steps=args.num_steps, n_proj=args.n_proj, name="self"),

        # GMMSWD(refernce_image, p=args.p, s=args.s, mode="Fixed", num_steps=args.num_steps, n_components=4, n_proj=args.n_proj),
        # GMMSWD(refernce_image, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_components=6, n_proj=args.n_proj),
        # DirectSWD(refernce_image, p=args.p, s=args.s, mode='Fixed', num_steps=args.num_steps, n_proj=args.n_proj),
        # DirectSWD(refernce_image, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj),
        # DirectSWD(low_res, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj, name="self-PGD", gradient_projector=gradient_projector(low_dim, high_dim, resize_kwargs)),
        # DirectSWD(low_res, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj, name="self"),
    ]

    # Run
    output_images = dict()
    loss_lists = dict()
    constraint_lists = dict()
    for model in models:
        torch.cuda.empty_cache()
        output, losses = model.run(low_res_big.clone())

        loss_lists[model.name] = losses
        # constraint_lists[model.name] = constraints
        output_images[model.name] = output
        save_image(output, f"{output_dir}/3-{model.name}.png", normalize=True)

    dump_hists(refernce_image, low_res, output_images, f"{output_dir}/4-hists.png")
    plot_values(loss_lists, f"{output_dir}/5-loss.png")
    # plot_values(constraint_lists, f"{output_dir}/5-constraints.png")



