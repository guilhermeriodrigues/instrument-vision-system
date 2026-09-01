import numpy as np


class ScaleConfidence:

    def compute(
        self,
        values,
        angles
    ):

        if len(values) < 3:
            return 0.0

        value_steps = np.diff(values)

        angle_steps = np.diff(angles)

        value_std = np.std(value_steps)

        angle_std = np.std(angle_steps)

        score = 1.0

        score -= value_std * 0.05

        score -= angle_std * 0.01

        return max(
            0.0,
            min(score, 1.0)
        )