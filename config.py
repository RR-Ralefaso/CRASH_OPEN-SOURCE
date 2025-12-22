import os

# Model Configuration
MODEL_PATH = 'yolo11n.pt'  # YOLOv11 nano model
CONFIDENCE = 0.7

# Camera Configuration
WINDOW_NAME = 'CRASH DETECTOR'
CAMERA_ID = 0 #change if camera sucks foo

# YOLO Inference Settings
YOLO_INPUT_SIZE = (640, 640)  # Standard YOLO input size

# Directory Configuration
LOG_DIR = "logs"
SCREENSHOT_DIR = "screenshots"
MODEL_DIR = "models"

# Create directories if they don't exist
for directory in [LOG_DIR, SCREENSHOT_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

print(f"[CONFIG] Using model: {MODEL_PATH}")
print(f"[CONFIG] Camera ID: {CAMERA_ID}")
print(f"[CONFIG] YOLO input size: {YOLO_INPUT_SIZE}")