from ultralytics import YOLO

model = YOLO(
    "best_pointer.pt"
)

result = model.predict(
    image
)