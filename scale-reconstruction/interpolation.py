import numpy as np

from scipy.interpolate import interp1d


class ScaleInterpolator:

    def __init__(self):

        self.model = None

    def fit(
        self,
        angles,
        values
    ):

        order = np.argsort(angles)

        angles = np.array(angles)[order]

        values = np.array(values)[order]

        self.model = interp1d(
            angles,
            values,
            fill_value="extrapolate",
            kind="linear"
        )

    def predict(
        self,
        angle
    ):

        return float(
            self.model(angle)
        )