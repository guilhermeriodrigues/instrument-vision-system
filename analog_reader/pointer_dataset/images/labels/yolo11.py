from ultralytics import YOLO

model = YOLO(
    "yolo11n-seg.pt"
)

model.train(
    data="pointer_dataset.yaml",
    epochs=100,
    imgsz=640
)