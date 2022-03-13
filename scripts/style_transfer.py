import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import distribution_metrics
from GPDM import GPDM
from utils import get_file_name, save_image, LossesList, GrayLevelLoss, aspect_ratio_resize, cv2pt, match_image_sizes

STYLE_DIR = 'images/style_transfer/style/'
CONTENT_DIR = 'images/style_transfer/content/'


def get_matching_content_img(target_img_path, content_img_path, resize):
    """Replicates the way a target image is matched to the size of the content like for"""
    import cv2
    target_img = cv2.imread(target_img_path)
    target_img = aspect_ratio_resize(target_img, max_dim=resize)
    target_img = cv2pt(target_img)
    content_img = cv2pt(cv2.imread(content_img_path))
    content_img = match_image_sizes(content_img.unsqueeze(0), target_img.unsqueeze(0))

    return content_img


def style_transfer(input_and_target_images, out_dir):
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    # resize = 256; p=7; lr=0.1;num_steps=500; content_weight=0.1
    resize=1024; p=11; lr=0.035;num_steps=500; content_weight=0
    for content_image_path, style_image_path in input_and_target_images:
        criteria = LossesList([
            distribution_metrics.PatchSWDLoss(patch_size=p, stride=1, num_proj=64),
            # GrayLevelLoss(get_matching_content_img(style_image_path, content_image_path, resize), resize=resize//2)
        ], weights=[1, content_weight])

        model = GPDM(coarse_dim=resize, lr=lr, num_steps=num_steps, init=content_image_path, noise_sigma=0, resize=resize)

        result = model.run(style_image_path, criteria, debug_dir=None)
        output_path = os.path.join(out_dir, model.name, f'{get_file_name(content_image_path)}-to-{get_file_name(style_image_path)}.png')
        save_image(result, output_path)


if __name__ == '__main__':
    input_and_target_images = [
        (f'{CONTENT_DIR}/bin2.jpg', f'{STYLE_DIR}brick.jpg'),
        (f'{CONTENT_DIR}/chicago.jpg', f'{STYLE_DIR}/starry_night.jpg'),
        (f'{CONTENT_DIR}/home_alone.jpg', f'{STYLE_DIR}/scream.jpg'),
        (f'{CONTENT_DIR}/hillary1.jpg', f'{STYLE_DIR}thick_oil.jpg'),
        (f'{CONTENT_DIR}/trump.jpg', f'{STYLE_DIR}mondrian.jpg'),
        (f'{CONTENT_DIR}/golden_gate.jpg', f'{STYLE_DIR}/scream.jpg'),
        (f'{CONTENT_DIR}/hotel_bedroom2.jpg', f'{STYLE_DIR}/Muse.jpg'),
        (f'{CONTENT_DIR}/cat1.jpg', f'{STYLE_DIR}/olive_Trees.jpg'),
        (f'{CONTENT_DIR}/cornell.jpg', f'{STYLE_DIR}/rug.jpeg'),
        (f'{CONTENT_DIR}/man1.jpg', f'{STYLE_DIR}/drawing.jpg'),
        (f'{CONTENT_DIR}/cornell.jpg', f'{STYLE_DIR}/rug.jpeg'),
        ('images/analogies/duck_mosaic.jpg', 'images/analogies/S_char.jpg'),
        ('images/analogies/S_char.jpg', 'images/analogies/duck_mosaic.jpg'),
        ('images/analogies/kanyon2.jpg', 'images/analogies/tower.jpg'),
        ('images/analogies/tower.jpg', 'images/analogies/kanyon2.jpg'),
    ]
    outputs_dir = f'outputs/style_transfer'
    style_transfer(input_and_target_images, outputs_dir)
