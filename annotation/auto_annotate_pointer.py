import cv2
import torch
import numpy as np

from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator


class PointerAutoAnnotator:

    def __init__(self, checkpoint):

        sam = sam_model_registry["vit_b"](
            checkpoint=checkpoint
        )

        sam.to("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = (
            SamAutomaticMaskGenerator(
                sam
            )
        )

    def process(self, image_path):

        image = cv2.imread(image_path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        masks = self.generator.generate(
            image
        )

        return masks