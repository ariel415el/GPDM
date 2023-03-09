import os

import cv2
import np as np
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

img_dir, bboxes ='../outputs/fox_256->1024_p-5_s-5_64',  [(500, 500, 128), (550,108, 128),  (150,550, 128)]
img_dir, bboxes ='../outputs/00130_256->1024_p-5_s-5',  [(500, 500, 128), (550,50, 128),  (800, 500, 128)]
s = 5
paths = [os.path.join(img_dir, "1-GT.png"), os.path.join(img_dir, "2-low_res_big.png")]
paths += sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.startswith("3")])
paths = sorted(paths, key=lambda x: 1 if "gan" in x.lower() else 0)
H = len(bboxes)
W = len(paths)
fig, axes = plt.subplots(nrows=H+1, ncols=W, figsize=(W*s, H*s))


for i, path in enumerate(paths):
    img = np.array(Image.open(path))
    for j, (x,y,d) in enumerate(bboxes):
        if i == 0:
            gt_img = cv2.rectangle(img, (x,y), (x+d, y+d), (255,0,0), s)
        else:
            gt_img = img
        axes[j+1, i].imshow(img[y:y+d, x:x+d])
        axes[j+1, i].axis('off')
    axes[0, i].imshow(gt_img)
    axes[0, i].axis('off')
    name=os.path.splitext(os.path.basename(path))[0]
    if name.startswith("3-"):
        name = name.split("3-")[1]
    if name == "2-low_res_big":
        name = "bilinear"
    if name == "1-GT":
        name = "GT"
    print(name)
    axes[0, i].set_title(name, fontsize=5*s)



plt.tight_layout()
plt.savefig(os.path.join(img_dir, "5-comparison.png"))
