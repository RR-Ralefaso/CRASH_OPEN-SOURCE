# config.py - Configuration settings
import os

# Model configuration
MODEL_PATH = 'yolo11n.pt'  # Will auto-download if not found locally
CONFIDENCE = 0.7  # Lower for maximum detections
WINDOW_NAME = 'CRASH DETECTOR'
CAMERA_ID = 0  # Changed to 0 as default (most common)

# Directories
LOG_DIR = "logs"
MODEL_DIR = "models"
SCREENSHOT_DIR = "screenshots"

# Create necessary directories
for directory in [LOG_DIR, MODEL_DIR, SCREENSHOT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Performance settings
YOLO_INPUT_SIZE = (640, 480)  # YOLO inference size
TARGET_FPS = 30