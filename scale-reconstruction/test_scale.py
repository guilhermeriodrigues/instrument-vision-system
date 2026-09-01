from dataclasses import dataclass
from typing import List
import math
import numpy as np
from scipy.interpolate import interp1d


@dataclass
class OCRItem:
    text: str
    value: float
    center_x: float
    center_y: float
    confidence: float


@dataclass
class ScalePoint:
    value: float
    angle: float
    confidence: float


class AngleMapper:

    @staticmethod
    def calculate_angle(
        center_x,
        center_y,
        point_x,
        point_y
    ):

        dx = point_x - center_x
        dy = center_y - point_y

        return math.degrees(
            math.atan2(dy, dx)
        )


class ScaleInterpolator:

    def __init__(self):
        self.model = None

    def fit(self, angles, values):

        order = np.argsort(angles)

        angles = np.array(angles)[order]
        values = np.array(values)[order]

        self.model = interp1d(
            angles,
            values,
            fill_value="extrapolate"
        )

    def predict(self, angle):

        return float(self.model(angle))


class ScaleReconstructor:

    def reconstruct(
        self,
        center_x,
        center_y,
        ocr_items
    ):

        points = []

        for item in ocr_items:

            angle = AngleMapper.calculate_angle(
                center_x,
                center_y,
                item.value["x"],
                item.value["y"]
            )

            points.append(
                ScalePoint(
                    value=item.key,
                    angle=angle,
                    confidence=1.0
                )
            )

        points.sort(key=lambda x: x.angle)

        angles = [p.angle for p in points]
        values = [p.value for p in points]

        interp = ScaleInterpolator()
        interp.fit(angles, values)

        return interp, points


if __name__ == "__main__":

    center_x = 360
    center_y = 360

    scale_numbers = {
        0: {"x": 120, "y": 560},
        4: {"x": 170, "y": 170},
        8: {"x": 360, "y": 80},
        12: {"x": 550, "y": 170},
        16: {"x": 600, "y": 560},
    }

    reconstructor = ScaleReconstructor()

    interpolator, points = reconstructor.reconstruct(
        center_x,
        center_y,
        scale_numbers
    )

    print("\nEscala Reconstruída:")
    print("-" * 40)

    for p in points:
        print(
            f"Valor={p.value:5.1f} "
            f"Ângulo={p.angle:8.2f}"
        )

    pointer_angle = -20

    reading = interpolator.predict(
        pointer_angle
    )

    print("\nLeitura:")
    print(
        f"Ângulo={pointer_angle:.2f}° "
        f"→ Valor={reading:.2f}"
    )