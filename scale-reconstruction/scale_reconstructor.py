from .ocr_scale_parser import (
    OCRScaleParser
)

from .scale_builder import (
    ScaleBuilder
)

from .interpolation import (
    ScaleInterpolator
)

from .confidence import (
    ScaleConfidence
)

from .models import ScaleModel


class ScaleReconstructor:

    def __init__(self):

        self.parser = OCRScaleParser()

        self.builder = ScaleBuilder()

        self.confidence = ScaleConfidence()

    def reconstruct(
        self,
        center_x,
        center_y,
        ocr_results,
        unit
    ):

        numeric_items = (
            self.parser.extract_numeric_items(
                ocr_results
            )
        )

        scale_points = self.builder.build(
            center_x,
            center_y,
            numeric_items
        )

        scale_points.sort(
            key=lambda x: x.angle
        )

        angles = [
            p.angle
            for p in scale_points
        ]

        values = [
            p.value
            for p in scale_points
        ]

        interpolator = ScaleInterpolator()

        interpolator.fit(
            angles,
            values
        )

        conf = (
            self.confidence.compute(
                values,
                angles
            )
        )

        return {
            "scale_model": ScaleModel(
                unit=unit,
                points=scale_points,
                confidence=conf
            ),
            "interpolator": interpolator
        }