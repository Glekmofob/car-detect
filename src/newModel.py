from ultralytics import YOLO

model = YOLO("yolov8n.pt")

resuts = model.train(data="data.yaml", epochs = 100, imgsz = 416)