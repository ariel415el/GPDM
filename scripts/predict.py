import tempfile
import torch
import torchvision.utils as vutils
from cog import BasePredictor, Path, Input

import distribution_metrics
from GPDM import GPDM
from utils import LossesList


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        content: Path = Input(description="Content image."),
        style: Path = Input(description="Style image."),
    ) -> Path:

        resize = 1024
        p = 11
        lr = 0.035
        num_steps = 500
        content_weight = 0

        criteria = LossesList(
            [
                distribution_metrics.PatchSWDLoss(patch_size=p, stride=1, num_proj=64),
            ],
            weights=[1, content_weight],
        )

        model = GPDM(
            coarse_dim=resize,
            lr=lr,
            num_steps=num_steps,
            init=str(content),
            noise_sigma=0,
            resize=resize,
        )

        result = model.run(str(style), criteria, debug_dir=None)

        out_path = Path(tempfile.mkdtemp()) / "output.png"
        vutils.save_image(torch.clip(result, -1, 1), str(out_path), normalize=True)

        return out_path
