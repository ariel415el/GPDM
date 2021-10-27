import os
import distribution_metrics
from synthesis import synthesize_image
from utils import SyntesisConfigurations, get_file_name
import torchvision.utils as vutils

image_paths = []
# dataset_dir = '/home/ariel/university/GPDM/GPDM/tests/downloaded_results/SIGD16_org'
# image_paths += [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]

dataset_dir = '/home/ariel/university/GPDM/GPDM/images/places50'
image_paths += [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]


def main():
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    for (p, l) in [(5, 8)]:
        dataset_name = os.path.basename(dataset_dir) + f'_p-{p}_l-{l}'
        criteria = distribution_metrics.PatchSWDLoss(patch_size=p, stride=1, num_proj=16, normalize_patch='none')
        os.makedirs(f'outputs/resampling_dataset/{dataset_name}/generated_images', exist_ok=True)
        for input_image_path in image_paths[:5]:
            for i in range(10):

                conf = SyntesisConfigurations(pyr_factor=0.75, n_scales=l, lr=0.05, num_steps=500, init='+noise', resize=None)

                outputs_dir = f'outputs/resampling_dataset/{dataset_name}/optimization/{get_file_name(input_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

                result = synthesize_image(input_image_path, criteria, None, conf, outputs_dir)
                fname, ext = os.path.splitext(os.path.basename(input_image_path))[:2]
                fname = f"{fname}${i}.{ext}"
                vutils.save_image(result, f'outputs/resampling_dataset/{dataset_name}/generated_images/{fname}', normalize=True)

if __name__ == '__main__':
    main()