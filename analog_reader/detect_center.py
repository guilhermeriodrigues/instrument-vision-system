import cv2
import numpy as np


class CenterDetector:

    def detect(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=100,
            param2=40,
            minRadius=50,
            maxRadius=0
        )

        if circles is None:
            raise RuntimeError(
                "Nenhum círculo encontrado"
            )

        circles = np.uint16(
            np.around(circles)
        )

        largest = max(
            circles[0],
            key=lambda c: c[2]
        )

        x, y, r = largest

        return {
            "cx": int(x),
            "cy": int(y),
            "radius": int(r)
        }