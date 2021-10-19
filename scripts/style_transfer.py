import os
import distribution_metrics
from retarget_image import retarget_image
from utils import SyntesisConfigurations, get_file_name, GrayLevelLoss

STYLE_DIR = '../images/style_transfer/style/'
CONTENT_DIR = '../images/style_transfer/content/'

input_and_target_images = [
    # (f'{STYLE_DIR}scream.jpg', f'{CONTENT_DIR}/home_alone.jpg'),
    # (f'{STYLE_DIR}mondrian.jpg', f'{CONTENT_DIR}/trump.jpg'),
    # (f'{STYLE_DIR}starry_night.jpg', f'{CONTENT_DIR}/cat1.jpg'),
    # (f'{STYLE_DIR}brick.jpg', f'{CONTENT_DIR}/bin2.jpg'),
    # (f'{STYLE_DIR}scream.jpg', f'{CONTENT_DIR}/golden_gate.jpg'),
    # (f'{STYLE_DIR}thick_oil.jpg', f'{CONTENT_DIR}/hillary1.jpg'),
    ('/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/style/boats_sunset.jpg', f'{CONTENT_DIR}/golden_gate.jpg'),

]


def main():
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    criteria = distribution_metrics.PatchSWDLoss(patch_size=11, stride=1, num_proj=256, normalize_patch='mean')
    conf = SyntesisConfigurations(n_scales=0, lr=0.05, num_steps=500, resize=512)
    # for content_image_path in [os.path.join(CONTENT_DIR, x) for x in os.listdir(CONTENT_DIR)]:
    #     for style_image_path in [os.path.join(STYLE_DIR, x) for x in os.listdir(STYLE_DIR)]:
    for style_image_path, content_image_path in input_and_target_images:
            conf.init = content_image_path

            outputs_dir = f'outputs/style_transfer/{get_file_name(content_image_path)}-to-{get_file_name(style_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

            # content_loss = GrayLevelLoss(128)
            content_loss = None
            retarget_image(style_image_path, criteria, content_loss, conf, outputs_dir)


if __name__ == '__main__':
    main()