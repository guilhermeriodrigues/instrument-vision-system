from typing import List

from .models import (
    OCRItem,
    ScalePoint
)

from .angle_mapper import AngleMapper


class ScaleBuilder:

    def build(
        self,
        center_x: float,
        center_y: float,
        ocr_items: List[OCRItem]
    ) -> List[ScalePoint]:

        points = []

        for item in ocr_items:

            angle = AngleMapper.calculate_angle(
                center_x,
                center_y,
                item.center_x,
                item.center_y
            )

            points.append(
                ScalePoint(
                    value=item.value,
                    angle=angle,
                    confidence=item.confidence
                )
            )

        return points