import math


class AngleMapper:

    @staticmethod
    def calculate_angle(
        center_x: float,
        center_y: float,
        point_x: float,
        point_y: float
    ) -> float:

        dx = point_x - center_x

        dy = center_y - point_y

        angle = math.degrees(
            math.atan2(dy, dx)
        )

        return angle