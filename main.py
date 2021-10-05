import distribution_metrics
from retarget_image import retarget_image
from utils import get_file_name, SyntesisConfigurations, cv2pt

import torch
class GrayLevelLoss(torch.nn.Module):
    def __init__(self, img_path, resize):
        super(GrayLevelLoss, self).__init__()
        import cv2
        from utils import aspect_ratio_resize
        self.img = cv2pt(aspect_ratio_resize(cv2.imread(img_path), max_dim=resize))

    def forward(self, x):
        from torchvision import transforms
        img = transforms.Resize((x.shape[-2], x.shape[-1]), antialias=True)(self.img.to(x.device))
        # return ((img.mean(0) - x[0].mean(0))**2).mean()
        return ((img - x[0])**2).mean()


def generate_texture():
    for texture_image_path in [
        # 'images/textures/olives.png',
        # 'images/textures/tomatos.png',
        # 'images/textures/green_waves.jpg',
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
        # ('/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/style/scream.jpg', '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/content/home_alone.jpg'),
        ('/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/style/starry_night.jpg', '/home/ariel/university/repos/image-quilting/images/tezan.jpg'),
        # ('/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/faces/00006.png', '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/faces/00001_bw.png'),
        # ('/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/style/yellow_sunset.jpg', '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/content/bair.jpg')
    ]:
        criteria = distribution_metrics.PatchSWDLoss(patch_size=13, stride=1, num_proj=512, normalize_patch='none')
        # criteria = distribution_metrics.VGGPerceptualLoss(pretrained=True, features_metric_name='gram',
        #                                     layers_and_weights=[('relu1_2', 1.0), ('relu2_2', 1.0), ('relu3_3', 1.0),('relu4_3', 1.0), ('relu5_3', 1.0)])

        conf = SyntesisConfigurations(n_scales=3, pyr_factor=0.75, lr=0.05, num_steps=10, init=content_image_path, resize=256)

        outputs_dir = f'test-outputs/style_transfer/{get_file_name(content_image_path)}-to-{get_file_name(style_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

        content_loss = GrayLevelLoss(content_image_path, 256)
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
        'images/resampling/balls.jpg',
        # 'images/resampling/birds.png'
        # 'images/resampling/pinguins.png'
        # 'images/resampling/green_view.jpg',
        # 'images/resampling/birds.png',
        # 'images/resampling/jerusalem2.jpg',
    ]:
        for i in range(3):
            # criteria = distribution_metrics.MMDApproximate(patch_size=5, strides=1, sigma=0.03, pool_size=-1, r=512, normalize_patch='channel_mean')
            criteria = distribution_metrics.PatchSWDLoss(patch_size=11, stride=1, num_proj=256, normalize_patch='none')

            conf = SyntesisConfigurations(pyr_factor=0.75, n_scales=5, lr=0.05, num_steps=1000, init="blur", resize=256)

            outputs_dir = f'outputs/image_resampling/{get_file_name(input_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

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
        # 'images/retargeting/corn.png',
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/image_retargeting/test-outputs/image_retargeting/PatchSWD(p-7:1)_AR-(1, 0.9)_R-256_S-0.75x5_I-blur/PatchSWD(p-7:1)_AR-(1, 0.9)_R-256_S-0.75x5_I-blur.png'
        # 'images/resampling/balloons.png',
        # 'images/resampling/birds.png'
        'images/resampling/pinguins.png',
        # 'images/resampling/birds.png',
    ]:
        for aspect_ratio in [(1,1.5)]:
            # criteria = distribution_metrics.MMDApproximate(patch_size=11, strides=1, sigma=0.03, pool_size=-1, r=512, normalize_patch='channel_mean')
            criteria = distribution_metrics.PatchSWDLoss(patch_size=11, stride=1, num_proj=1024, normalize_patch='channel_mean')

            conf = SyntesisConfigurations(pyr_factor=0.65, n_scales=5, aspect_ratio=aspect_ratio, lr=0.05, num_steps=500, init="blur", resize=256, blur_loss=0)

            outputs_dir = f'test-outputs/image_retargeting/{get_file_name(input_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

            # content_loss = GrayLevelLoss(input_image_path, 256)
            retarget_image(input_image_path, criteria, None, conf, outputs_dir)


if __name__ == '__main__':
    # image_resampling()
    # style_transfer()
    # edit_image()
    # image_resampling()
    image_retargeting()

