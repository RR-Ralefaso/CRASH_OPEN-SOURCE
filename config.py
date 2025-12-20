# config.py

MODEL_PATH = 'yolov11x.pt'  # Top-accuracy YOLO11 extra-large model (SOTA mAP on COCO, excels in complex scenes like crashes)
# Download via: yolo export model=yolov11x.pt format=pt  (runs offline after)
CONFIDENCE = 0.7  # Filters low-confidence detections
WINDOW_NAME = 'CRASH'
CAMERA_ID = 2  # Your camera index
