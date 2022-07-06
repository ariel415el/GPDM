import os
import sys
from math import sqrt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os.path
import GPDM
from patch_swd import PatchSWDLoss
from utils import load_image, get_pyramid_scales
from torchvision.utils import save_image
from cog import BasePredictor, Path, Input


"""
This script is for A live demo available at "https://replicate.com/ariel415el/gpdm".
It allows performing the "Reshuffle", "Retarget" or "Style-transfer" tasks on any uploaded image online with a 
considerable amount of control over the GPDM algorithm hyper-parameters
"""


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
            self,
            task: str = Input(choices=["Reshuffle", "Retarget", "Style-transfer"], default="Reshuffle",
                              description="(Reshuffle): Start from a noisy and generates samples from the same scene "
                                          "in the reference image.\n"
                                          "(Retarget): Start from a stretched version of the reference and recreates "
                                          "the scene in another aspect ratio.\n"
                                          "(Style-transfer): Recreates a content-image with the style of the reference"
                                          " image"),
            num_outputs: int = Input(default=4, description="How many output images to generate. Generating multiple "
                                                            "images at once improves the quality and diversity of the "
                                                            "results"),
            num_projections: int = Input(default=64, description="Number of ramdom projections for SWD."
                                                                 " More is better results but is slower and memory "
                                                                 "inefficient"),
            patch_size: int = Input(default=8, description="Size of the extracted a patches"),
            height_factor: int = Input(default=1, description="Change the output's aspect ratio (Retargeting)"),
            width_factor: int = Input(default=1, description="Change the output's aspect ratio (Retargeting)"),
            reference_image: Path = Input(description="The main input image - Style image in style-transfer."),
            content_image: Path = Input(description="Only relevant for style-transfer", default=None),

    ) -> Path:
        refernce_images = load_image(str(reference_image)).repeat(num_outputs, 1, 1, 1)
        ref_height = refernce_images.shape[-2]

        kwargs = {
            'init_from':"mean",
            'lr':0.01,
            'num_steps': 300,
            'additive_noise_sigma': 0,
            'aspect_ratio': (height_factor, width_factor),
            'pyramid_scales': get_pyramid_scales(ref_height, 4 * patch_size, 2 * patch_size)
        }
        if task == "Style-transfer":
            assert os.path.exists(content_image), "style-transfer must be initialized with a content image"
            kwargs['init_from'] = str(content_image)
            kwargs['pyramid_scales'] = [ref_height]
            kwargs['lr'] = 0.035
            kwargs['num_steps'] = 500
        elif task == "Retarget":
            if height_factor == 1 and width_factor == 1:
                raise Warning("You should change the aspect ratio in order to get interesting retargeting outputs,"
                              " set height_factor/width_factor different from 1")
            kwargs['init_from'] = "target"
        else:  # Reshuffle
            kwargs['additive_noise_sigma'] = 1.5
            if height_factor != 1 or width_factor != 1:
                raise Warning("'Reshuffling' with non-default aspect ratio means 'retargeting'. Change task='Retarget' "
                              "for better results")

        criteria = PatchSWDLoss(patch_size=patch_size, stride=1, num_proj=num_projections)
        new_images = GPDM.generate(refernce_images, criteria, **kwargs)

        output_path = 'output.png'
        save_image(new_images, output_path, normalize=True, nrow=int(sqrt(num_outputs)))

        return Path(output_path)
