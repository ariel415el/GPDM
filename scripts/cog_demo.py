import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os.path
import GPDM
from patch_swd import PatchSWDLoss
from utils import load_image, get_scales
from torchvision.utils import save_image
from cog import BasePredictor, Path, Input


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        task: str = Input(choices=["Reshuffle", "retarget", "style-transfer"]),
        num_outputs: int = Input(default=1, description="How many output images to generate"),
        num_projections: int = Input(default=64, description="Number of ramdom projections for SWD."
                                                 "More is better results but is slower and more memory inefficient"),
        patch_size: int = Input(default=8, description="size of the extracted a patches, I found 11  to work better for style transfer"),
        reference_image: Path = Input(description="The main input image - Style image in style-transfer."),
        content_image: Path = Input(description="Only relevant for style-transfer"),

    ) -> Path:
        refernce_images = load_image(reference_image).repeat(num_outputs, 1, 1, 1)

        criteria = PatchSWDLoss(patch_size=patch_size, stride=1, num_proj=num_projections)

        scales = get_scales(refernce_images.shape[-2], 32, 16)
        if task == "style-transfer":
            assert os.path.exists(content_image), "style-transfer must be initialized with a content image"
            init_from = content_image
            scales = [refernce_images.shape[-2]]
        elif task == "retarget":
            init_from = "target"
        else:
            init_from = "mean"

        new_images = GPDM.generate(refernce_images, criteria,
                                   scales=scales,
                                   init_from=init_from
                                   )

        output_path = 'output.png'
        save_image(new_images, output_path, normalize=True)

        return Path(output_path)

