import distribution_metrics
from retarget_image import retarget_image
from utils import SyntesisConfigurations, get_file_name

images = [
        '../images/textures/olives.png',
        '../images/textures/tomatos.png',
        '../images/textures/green_waves.jpg',
        '../images/textures/cobbles.jpeg'
    ]

def main():
    # criteria = distribution_metrics.PatchMMD_RBF(patch_size=7, stride=1, normalize_patch='none')
    criteria = distribution_metrics.PatchSWDLoss(patch_size=11, stride=1, num_proj=512, normalize_patch='none')

    for texture_image_path in images:

        conf = SyntesisConfigurations(pyr_factor=0.75, n_scales=8, aspect_ratio=(2, 2), lr=0.005, num_steps=500, init="mean", resize=256)

        outputs_dir = f'outputs/periodic_textures/{get_file_name(texture_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

        retarget_image(texture_image_path, criteria, None, conf, outputs_dir)

if __name__ == '__main__':
    main()