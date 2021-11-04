import os
from collections import defaultdict

import imageio


def make_gif(directory, duration_per_image=0.7):
    images = []
    fnames = sorted(os.listdir(directory), key=lambda x: int(x.split('-')[1][:-4]))
    for filename in fnames:
        fpath = os.path.join(directory, filename)
        if (fpath.endswith('.jpg') or fpath.endswith('.png')) and "plot.png" not in fpath:
            images.append(imageio.imread(fpath))
    imageio.mimsave(os.path.join(os.path.dirname(directory), os.path.basename(directory) + ".gif"), images, duration=duration_per_image)


def make_diversity_gifs(dir_path):
    image_sets = defaultdict(lambda: [])
    for name in os.listdir(dir_path):
        img_idx = name.split('$')[0]
        image_sets[img_idx].append(imageio.imread(os.path.join(dir_path, name)))

    for name in image_sets:
        imageio.mimsave(os.path.join(os.path.dirname(dir_path), f"{name}.gif"), image_sets[name], duration=0.5)

if __name__ == '__main__':
    path = '/home/ariel/university/GPDM/GPDM/scripts/outputs/resampling_dataset/SIGD16_org_PatchSWD(p-5:1)_AR-(1.0, 1.0)_R-None_S-0.85x13_I-noise+I(0,1.5)/generated_images'
    make_diversity_gifs(path)