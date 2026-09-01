import cv2
import numpy as np


class PointerDetector:

    def detect(
        self,
        image,
        cx,
        cy
    ):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        edges = cv2.Canny(
            gray,
            50,
            150
        )

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=50,
            minLineLength=40,
            maxLineGap=15
        )

        if lines is None:
            raise RuntimeError(
                "Ponteiro não encontrado"
            )

        best = None
        best_length = 0

        for line in lines:

            x1, y1, x2, y2 = line[0]

            d1 = np.hypot(
                x1 - cx,
                y1 - cy
            )

            d2 = np.hypot(
                x2 - cx,
                y2 - cy
            )

            if min(d1, d2) > 50:
                continue

            length = np.hypot(
                x2 - x1,
                y2 - y1
            )

            if length > best_length:

                best_length = length
                best = (
                    x1,
                    y1,
                    x2,
                    y2
                )

        if best is None:
            raise RuntimeError(
                "Ponteiro inválido"
            )

        x1, y1, x2, y2 = best

        if np.hypot(
            x1 - cx,
            y1 - cy
        ) < np.hypot(
            x2 - cx,
            y2 - cy
        ):
            tip_x = x2
            tip_y = y2
        else:
            tip_x = x1
            tip_y = y1

        angle = np.degrees(
            np.arctan2(
                cy - tip_y,
                tip_x - cx
            )
        )

        return {
            "tip_x": int(tip_x),
            "tip_y": int(tip_y),
            "angle": float(angle)
        }