from ultralytics import YOLO

# =========================================
# LOAD MODEL
# =========================================

model = YOLO("yolov8n-cls.pt")

# =========================================
# TRAIN
# =========================================

model.train(
    data=r"C:\Users\guilh\instrument-vision-system\dataset",
    epochs=20,
    imgsz=224,
    batch=16,
    device="cpu",
    workers=4
)