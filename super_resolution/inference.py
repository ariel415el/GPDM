import argparse
import os

import torch
from torchvision.transforms import Resize, InterpolationMode

from test import dump_image
from utils import load_image
from models import DirectSWD

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GPDM')
    parser.add_argument('input_image', default='/cs/labs/yweiss/ariel1/data/afhq/train/wild/flickr_wild_000060.jpg')
    parser.add_argument('--scale_factor', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--n_proj', type=int, default=64)
    parser.add_argument('--p', type=int, default=5)
    parser.add_argument('--s', type=int, default=5)
    parser.add_argument('--projected_gradient', default=False, action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)

    output_dir = "outputs/" + os.path.splitext(os.path.basename(args.input_image))[0] + f"x{args.scale_factor}_p-{args.p}_s-{args.s}"
    os.makedirs(output_dir, exist_ok=True)

    low_res = load_image(args.input_image).to(device)

    up_perator = Resize(low_res.shape[-1] * args.scale_factor, antialias=True, interpolation=InterpolationMode.BICUBIC)

    init_image = up_perator(low_res)
    print(init_image.shape, low_res.shape)

    model = DirectSWD(low_res, p=args.p, s=args.s, mode="Resample", num_steps=args.num_steps, n_proj=args.n_proj)

    torch.cuda.empty_cache()
    output, losses = model.run(init_image.clone())

    dump_image(init_image, f"{output_dir}/bilinear.png")
    dump_image(output, f"{output_dir}/3-{model.name}.png")
