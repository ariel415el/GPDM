import distribution_metrics
from retarget_image import retarget_image
from utils import get_file_name, SyntesisConfigurations, cv2pt

import torch



def generate_texture():
    for texture_image_path in [
        'images/textures/olives.png',
        'images/textures/tomatos.png',
        'images/textures/green_waves.jpg',
        'images/textures/cobbles.jpeg'
    ]:
        criteria = distribution_metrics.PatchMMD_RBF(patch_size=7, stride=1, normalize_patch='none')
        # criteria = distribution_metrics.PatchSWDLoss(patch_size=11, stride=1, num_proj=512, normalize_patch='none')

        conf = SyntesisConfigurations(pyr_factor=0.75, n_scales=8, aspect_ratio=(2, 2), lr=0.01, num_steps=500, init="mean", resize=256)

        outputs_dir = f'outputs/texture_synthesis/{get_file_name(texture_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

        retarget_image(texture_image_path, criteria, None, conf, outputs_dir)


def style_transfer():
    for style_image_path, content_image_path in [
        # ('images/analogies/duck_mosaic.jpg', 'images/analogies/S_char.jpg'),
        # ('images/analogies/S_char.jpg', 'images/analogies/duck_mosaic.jpg'),
        # ('images/analogies/kanyon2.jpg', 'images/analogies/tower.jpg'),
        # ('images/analogies/tower.jpg', 'images/analogies/kanyon2.jpg'),
        # ('images/style_transfer/trump.jpg', 'images/style_transfer/obama1.jpg'),
        # ('/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/style/scream.jpg', '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/content/home_alone.jpg'),
        # ('/home/ariel/university/imageTranslation/outputs/9-to-58-nn_smooth/reference_a.png', '/home/ariel/university/imageTranslation/outputs/9-to-58-nn_smooth/result_a-with-patches-of-b.png'),
        # ('/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/style/starry_night.jpg', '/home/ariel/university/repos/image-quilting1/images/tezan.jpg'),
        # ('/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/faces/00006.png', '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/faces/00001_bw.png'),
        ('/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/style/yellow_sunset.jpg', '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/content/bair.jpg')
    ]:
        # criteria = distribution_metrics.PatchCoherentLoss(patch_size=15, stride=9)
        criteria = distribution_metrics.PatchSWDLoss(patch_size=15, stride=1, num_proj=256, normalize_patch='none')
        # criteria = distribution_metrics.VGGPerceptualLoss(pretrained=True, features_metric_name='gram',
        #                                     layers_and_weights=[('relu1_2', 1.0), ('relu2_2', 1.0), ('relu3_3', 1.0),('relu4_3', 1.0), ('relu5_3', 1.0)])

        conf = SyntesisConfigurations(n_scales=0, pyr_factor=0.75, lr=0.05, num_steps=300, init=content_image_path, resize=256)

        outputs_dir = f'outputs/style_transfer/{get_file_name(content_image_path)}-to-{get_file_name(style_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

        # content_loss = GrayLevelLoss(content_image_path, 32)
        content_loss = None
        retarget_image(style_image_path, criteria, content_loss, conf, outputs_dir)


def edit_image():
    for style_image_path, content_image_path in [
        # ('images/resampling/balls.jpg', 'images/edit_inputs/balls_green_and_black.jpg'),
        # ('images/resampling/birds.png', 'images/edit_inputs/birds_edit_1.jpg'),
        # ('images/resampling/birds.png', 'images/edit_inputs/birds_edit_2.png'),
        # ('images/retargeting/birds2.jpg', 'images/edit_inputs/birds-on-tree_edit_1.jpg'),
        ('images/retargeting/house.jpg', 'images/edit_inputs/house_edit1.jpg'),
        # ('images/resampling/balloons.png', 'images/edit_inputs/balloons_edit.jpg'),
        # ('images/image_editing/stone.png', 'images/edit_inputs/stone_edit.png'),
        # ('images/image_editing/tree.png', 'images/edit_inputs/tree_edit.png'),
        # ('images/image_editing/swiming1.jpg', 'images/edit_inputs/swiming1_edit.jpg'),
        # ('images/retargeting/mountins2.jpg', 'images/edit_inputs/mountins2_edit.jpg'),
        # ('images/retargeting/mountins2.jpg', 'images/edit_inputs/mountins2_edit2.jpg'),
    ]:
        for (patch_size, n_scales) in [(11,0), (5,0)]:
            criteria = distribution_metrics.PatchSWDLoss(patch_size=patch_size, stride=1, num_proj=512, normalize_patch='mean')

            conf = SyntesisConfigurations(n_scales=n_scales, pyr_factor=0.75, lr=0.05, num_steps=500, init=content_image_path, resize=256, tv_loss=0)

            outputs_dir = f'outputs/image_editing/{get_file_name(content_image_path)}-to-{get_file_name(style_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

            # content_loss = GrayLevelLoss(content_image_path, 256)
            retarget_image(style_image_path, criteria, None, conf, outputs_dir)


def image_resampling():
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    for input_image_path in [
        # 'images/resampling/cows.png',
        # 'images/resampling/balloons.png',
        # 'images/resampling/people_on_the_beach.jpg',
        # 'images/resampling/balls.jpg',
        # 'images/SIGD16/boats.jpg',
        # 'images/SIGD16/birds_on_branch.jpg',
        # 'images/places50/16.jpg',
        # 'images/places50/30.jpg',
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/style/tarry_night.jpg',
        '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/style/Vincent_van_Gogh_Olive_Trees.jpg',
        '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/style/yellow_sunset.jpg'
        # 'images/places50/33.jpg',
        # 'images/resampling/pinguins.png',
        # 'images/resampling/birds.png',
        # 'images/resampling/jerusalem2.jpg',
    ]:
        for i in range(3):
            # criteria = distribution_metrics.MMDApproximate(patch_size=5, strides=1, sigma=0.03, pool_size=-1, r=512, normalize_patch='channel_mean')
            criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=256, normalize_patch='mean')
            # criteria = distribution_metrics.PatchCoherentLoss(patch_size=11, stride=4)
            # criteria = distribution_metrics.PatchMMD_RBF(patch_size=7, stride=3, sigma=0.06, normalize_patch='mean')
            # from distribution_metrics.patch_swd import PatchCoherentSWDLoss
            # criteria = PatchCoherentSWDLoss(patch_size=7, stride=1, num_proj=1, normalize_patch='mean')

            conf = SyntesisConfigurations(pyr_factor=0.75, n_scales=9, lr=0.05, num_steps=75, init='blur', resize=512)

            outputs_dir = f'outputs/image_resampling/{get_file_name(input_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

            # content_loss = GrayLevelLoss(content_image_path, 256)
            retarget_image(input_image_path, criteria, None, conf, outputs_dir)


def image_retargeting():
    for input_image_path in [
        # 'images/retargeting/mountains3.png',
        # 'images/retargeting/birds2.jpg',
        # 'images/retargeting/fish.png',
        # 'images/retargeting/fruit.png',
        # 'images/retargeting/mountins2.jpg',
        # 'images/retargeting/colusseum.png',
        # 'images/retargeting/mountains.jpg',
        # 'images/retargeting/SupremeCourt.jpeg',
        # 'images/retargeting/kanyon.jpg',
        # 'images/retargeting/corn.png',
        'images/retargeting/corn.png',
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/image_retargeting/test-outputs/image_retargeting/PatchSWD(p-7:1)_AR-(1, 0.9)_R-256_S-0.75x5_I-blur/PatchSWD(p-7:1)_AR-(1, 0.9)_R-256_S-0.75x5_I-blur.png'
        # 'images/resampling/balloons.png',
        # 'images/resampling/birds.png'
        # 'images/resampling/pinguins.png',
        # 'images/SIGD16/birds_on_branch.jpg',
        # 'images/resampling/birds.png',
    ]:
        for aspect_ratio in [(1,1.5), (1,0.4)]:
            # criteria = distribution_metrics.MMDApproximate(patch_size=11, strides=1, sigma=0.03, pool_size=-1, r=512, normalize_patch='channel_mean')
            # criteria = distribution_metrics.PatchSWDLoss(patch_size=11, stride=1, num_proj=256, normalize_patch='channel_mean')
            criteria = distribution_metrics.PatchCoherentLoss(patch_size=15, stride=5)

            conf = SyntesisConfigurations(pyr_factor=0.75, n_scales=0, aspect_ratio=aspect_ratio, lr=0.05, num_steps=200, init="blur", resize=128, blur_loss=0)

            outputs_dir = f'outputs/image_retargeting/{get_file_name(input_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

            # content_loss = GrayLevelLoss(input_image_path, 256)
            retarget_image(input_image_path, criteria, None, conf, outputs_dir)


if __name__ == '__main__':
    # image_resampling()
    # style_transfer()
    # edit_image()
    image_resampling()
    # image_retargeting()

