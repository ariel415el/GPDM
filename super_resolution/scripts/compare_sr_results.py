import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from super_resolution.sr_utils.metrics import calculate_psnr, calculate_ssim, LapSWD

if __name__ == '__main__':
    s = 3
    bbox_dict = defaultdict(lambda : ((0.5, 0.5, 0.125), (0.5, 0.125, 0.125), (0.125, 0.5, 0.125)))
    bbox_dict["fox"] = [(0.5, 0.5, 0.125), (0.5, 0.125, 0.125), (0.125, 0.5, 0.125)]
    bbox_dict["00130"] = [(0.5, 0.5, 0.125), (0.5, 0.7, 0.125), (0.75, 0.5, 0.125)]

    dirpath = "outputs"
    for dirname in os.listdir("outputs"):
        img_dir = os.path.join(dirpath, dirname)

        img_name = dirname.split("_")[0]
        bboxes = bbox_dict[img_name]

        gt_img = np.array(Image.open(os.path.join(img_dir, "1-GT.png")))

        paths = [os.path.join(img_dir, "1-GT.png"), os.path.join(img_dir, "2-initial_guess.png")]
        paths += sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.startswith("3")])
        paths = sorted(paths, key=lambda x: 1 if "gan" in x.lower() else 0)

        H = len(bboxes)
        W = len(paths)
        fig, axes = plt.subplots(nrows=H+1, ncols=W, figsize=(W*s, H*s))

        for i, path in enumerate(paths):
            print(path)
            img = np.array(Image.open(path))
            dim = img.shape[-2]
            for j, (x,y,d) in enumerate(bboxes):
                x,y,d = int(x*dim), int(y*dim), int(d * dim)
                # if i == 0:
                #     img_with_rects = cv2.rectangle(img, (x,y), (x+d, y+d), (255,0,0), s)
                # else:
                #     img_with_rects = img
                axes[j+1, i].imshow(img[y:y+d, x:x+d])
                axes[j+1, i].axis('off')
            axes[0, i].imshow(img)
            axes[0, i].axis('off')
            name=os.path.splitext(os.path.basename(path))[0]
            if name.startswith("3-"):
                name = name[2:]
            if name == "2-initial_guess":
                name = "bilinear"
            if name == "1-GT":
                name = "GT"
            name += f"\nPSNR: {calculate_psnr(gt_img, img):.2f} " \
                    f"\nSSIM: {calculate_ssim(img, gt_img): .2f} " \
                    f"\nLapSWD: {LapSWD()(img, gt_img):.2f}"
            axes[0, i].set_title(name, fontsize=5*s)

        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, "5-comparison.png"))

# from basicsr import calculate_ssim, calculate_psnr