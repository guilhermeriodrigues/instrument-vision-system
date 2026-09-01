import cv2
import easyocr

from ultralytics import YOLO

# =========================================
# LOAD CLASSIFIER
# =========================================

classifier = YOLO(
    r"C:\Users\guilh\Documents\instrument-vision-system\runs\classify\train\weights\best.pt"
)

# =========================================
# LOAD OCR
# =========================================

reader = easyocr.Reader(['en'])

# =========================================
# CAMERA
# =========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Erro ao abrir webcam")
    exit()

# =========================================
# FRAME COUNTER
# =========================================

frame_count = 0

ocr_text = ""

# =========================================
# LOOP
# =========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # =====================================
    # ROI
    # =====================================

    h, w, _ = frame.shape

    box_size = 300

    x1 = (w // 2) - (box_size // 2)
    y1 = (h // 2) - (box_size // 2)

    x2 = x1 + box_size
    y2 = y1 + box_size

    roi = frame[y1:y2, x1:x2]

    # =====================================
    # CLASSIFICATION
    # =====================================

    results = classifier.predict(
        roi,
        imgsz=224,
        verbose=False
    )

    result = results[0]

    predicted_class = result.names[
        result.probs.top1
    ]

    confidence = float(
        result.probs.top1conf
    )

    # =====================================
    # COLORS
    # =====================================

    if predicted_class == "Analog_Instruments":

        color = (0, 255, 0)

    else:

        color = (0, 0, 255)

    # =====================================
    # OCR ONLY EVERY 20 FRAMES
    # =====================================

    if (
        predicted_class == "Digital_Instruments"
        and confidence > 0.8
        and frame_count % 20 == 0
    ):

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        _, thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        ocr_results = reader.readtext(
            thresh
        )

        detected = ""

        for item in ocr_results:

            text = item[1]

            conf = item[2]

            if conf > 0.3:

                detected += text + " "

        ocr_text = detected

    # =====================================
    # DRAW ROI
    # =====================================

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    # =====================================
    # CLASS LABEL
    # =====================================

    label = (
        f"{predicted_class} "
        f"{confidence:.2f}"
    )

    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    # =====================================
    # OCR TEXT
    # =====================================

    cv2.putText(
        frame,
        f"OCR: {ocr_text}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # =====================================
    # SHOW
    # =====================================

    cv2.imshow(
        "Instrument Vision System",
        frame
    )

    # =====================================
    # EXIT
    # =====================================

    if cv2.waitKey(1) == ord("q"):
        break

# =========================================
# RELEASE
# =========================================

cap.release()

cv2.destroyAllWindows()