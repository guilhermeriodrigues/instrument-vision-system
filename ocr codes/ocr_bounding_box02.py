import cv2
import pytesseract
from pytesseract import Output

# Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load image
img = cv2.imread(r"C:\Users\guilh\Documents\instrument-vision-system\images\digital_instrument02.jpg")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# OCR
config_tesseract = r'--psm 6'
result = pytesseract.image_to_data(gray, config=config_tesseract, lang="eng", output_type=Output.DICT)

min_conf = 7

# Containers for metrics
confidences = []
areas = []
widths = []
heights = []
detected_words = []

# Draw bounding boxes + compute metrics
for i in range(len(result["text"])):
    text = result["text"][i].strip()
    conf = int(result["conf"][i])

    if conf > -1:   # evita valores inválidos
        confidences.append(conf)

    if text:  # só considera quando há texto
        detected_words.append(text)
        x = result["left"][i]
        y = result["top"][i]
        w = result["width"][i]
        h = result["height"][i]

        widths.append(w)
        heights.append(h)
        areas.append(w * h)

        if conf > min_conf:
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Put text above box
            cv2.putText(
                img,
                text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

# ====== METRICS ======
avg_conf = sum(confidences) / len(confidences) if confidences else 0
num_words = len(detected_words)
avg_w = sum(widths) / len(widths) if widths else 0
avg_h = sum(heights) / len(heights) if heights else 0
avg_area = sum(areas) / len(areas) if areas else 0
good_conf_ratio = len([c for c in confidences if c > min_conf]) / len(confidences) if confidences else 0

# ====== OUTPUT ======
print("\n===== OCR METRICS =====")
print(f"Total words detected: {num_words}")
print(f"Average confidence: {avg_conf:.2f}")
print(f"High-confidence ratio (> {min_conf}): {good_conf_ratio:.2f}")
print(f"Average bounding box width: {avg_w:.2f}")
print(f"Average bounding box height: {avg_h:.2f}")
print(f"Average bounding box area: {avg_area:.2f}")

print("\nExtracted text:")
print(" ".join(detected_words))

# Show image with annotations
cv2.imshow("OCR Result with Metrics", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
