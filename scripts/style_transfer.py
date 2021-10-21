import os
import distribution_metrics
from synthesis import synthesize_image
from utils import SyntesisConfigurations, get_file_name, GrayLevelLoss

STYLE_DIR = '../images/style_transfer/style/'
CONTENT_DIR = '../images/style_transfer/content/'

input_and_target_images = [
    (f'{STYLE_DIR}brick.jpg', f'{CONTENT_DIR}/bin2.jpg'),
    (f'{STYLE_DIR}/starry_night.jpg', f'{CONTENT_DIR}/chicago.jpg'),
    (f'{STYLE_DIR}thick_oil.jpg', f'{CONTENT_DIR}/hillary1.jpg'),
    (f'{STYLE_DIR}mondrian.jpg', f'{CONTENT_DIR}/trump.jpg'),
    (f'{STYLE_DIR}/scream.jpg', f'{CONTENT_DIR}/golden_gate.jpg'),
    (f'{STYLE_DIR}/Muse.jpg', f'{CONTENT_DIR}/hotel_bedroom2.jpg'),
    (f'{STYLE_DIR}/olive_Trees.jpg', f'{CONTENT_DIR}/cat1.jpg'),
    (f'/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/textures/rug.jpeg', f'{CONTENT_DIR}/cornell.jpg'),

]


def main():
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    criteria = distribution_metrics.PatchSWDLoss(patch_size=11, stride=1, num_proj=256, normalize_patch='none')
    conf = SyntesisConfigurations(n_scales=0, lr=0.03, num_steps=500, resize=512)
    for style_image_path, content_image_path in input_and_target_images:
            conf.init = content_image_path

            outputs_dir = f'outputs/style_transfer/{get_file_name(content_image_path)}-to-{get_file_name(style_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

            content_loss = GrayLevelLoss(128)
            # content_loss = None
            synthesize_image(style_image_path, criteria, content_loss, conf, outputs_dir)


if __name__ == '__main__':
    main()