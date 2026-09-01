import cv2

from detect_center import CenterDetector
from scale_ocr import ScaleOCR
from pointer_detection import PointerDetector


IMAGE_PATH = (
    r"C:\Users\guilh\instrument-vision-system\images\Analog_Instruments\analog_instrument4.jpg"
)

image = cv2.imread(
    IMAGE_PATH
)

center_detector = CenterDetector()
ocr = ScaleOCR()
pointer_detector = PointerDetector()

center = center_detector.detect(
    image
)

ocr_result = ocr.read(
    image
)

pointer = pointer_detector.detect(
    image,
    center["cx"],
    center["cy"]
)

print("\n=== CENTRO ===")
print(center)

print("\n=== OCR ===")
print(ocr_result)

print("\n=== PONTEIRO ===")
print(pointer)

cv2.circle(
    image,
    (
        center["cx"],
        center["cy"]
    ),
    center["radius"],
    (0,255,0),
    2
)

cv2.circle(
    image,
    (
        center["cx"],
        center["cy"]
    ),
    5,
    (255,0,0),
    -1
)

cv2.line(
    image,
    (
        center["cx"],
        center["cy"]
    ),
    (
        pointer["tip_x"],
        pointer["tip_y"]
    ),
    (0,0,255),
    2
)

output_path = "resultado_debug.jpg"

cv2.imwrite(
    output_path,
    image
)

print(
    f"\nImagem salva em: {output_path}"
)