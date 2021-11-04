import os
import distribution_metrics
from GPDM import GPDM
from utils import save_image, get_file_name, LossesList

image_paths = []
dataset_dir = '/home/ariel/university/GPDM/GPDM/images/HQ_16'
image_paths += [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]


def main():
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    # criteria = LossesList([
    #     distribution_metrics.PatchSWDLoss(patch_size=5, stride=1, num_proj=64, normalize_patch='mean'),
    #     distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=64, normalize_patch='mean'),
    #     distribution_metrics.PatchSWDLoss(patch_size=11, stride=1, num_proj=64, normalize_patch='mean')
    # ], weights=[1,1,1])
    criteria = distribution_metrics.PatchSWDLoss(patch_size=7, stride=1, num_proj=64)
    model = GPDM(pyr_factor=0.85, coarse_dim=28, lr=0.05, num_steps=300, init='target', noise_sigma=0.15, resize=None, scale_factor=(1,1.25))
    output_dir = f'outputs/resampling_dataset/{os.path.basename(dataset_dir)}_{criteria.name}_{model.name}'
    for input_image_path in image_paths:
        for i in range(1):
            fname, ext = os.path.splitext(os.path.basename(input_image_path))[:2]

            debug_dir = f'{output_dir}/optimization/{fname}-{i}'

            result = model.run(input_image_path, criteria, debug_dir)

            save_image(result, f'{output_dir}/generated_images/{fname}${i}{ext}')

if __name__ == '__main__':
    main()