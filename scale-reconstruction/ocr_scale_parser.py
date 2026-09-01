import re

from typing import List

from .models import OCRItem


class OCRScaleParser:

    def extract_numeric_items(
        self,
        ocr_results
    ) -> List[OCRItem]:

        output = []

        for item in ocr_results:

            text = item["text"]

            text = text.replace(",", ".")

            match = re.match(
                r"^-?\d+(\.\d+)?$",
                text
            )

            if not match:
                continue

            value = float(text)

            x = item["center_x"]
            y = item["center_y"]

            output.append(
                OCRItem(
                    text=text,
                    value=value,
                    center_x=x,
                    center_y=y,
                    confidence=item["confidence"]
                )
            )

        return output