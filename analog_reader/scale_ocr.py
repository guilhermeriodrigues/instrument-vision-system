import easyocr
import re


class ScaleOCR:

    def __init__(self):

        self.reader = easyocr.Reader(
            ['en'],
            gpu=False
        )

    def read(self, image):

        results = self.reader.readtext(
            image
        )

        numbers = []

        units = []

        for bbox, text, conf in results:

            text = text.strip()

            if conf < 0.3:
                continue

            center_x = int(
                sum(
                    p[0]
                    for p in bbox
                ) / 4
            )

            center_y = int(
                sum(
                    p[1]
                    for p in bbox
                ) / 4
            )

            if re.match(
                r'^-?\d+(\.\d+)?$',
                text
            ):

                numbers.append(
                    {
                        "value": float(text),
                        "x": center_x,
                        "y": center_y,
                        "confidence": conf
                    }
                )

            else:

                units.append(text)

        return {
            "numbers": numbers,
            "units": units
        }