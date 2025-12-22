# config.py
import os

# Use the smaller, faster nano model - it's more reliable
MODEL_PATH = 'yolo11n.pt'  # Use yolo8n.pt if yolo11n doesn't work
CONFIDENCE = 0.7
WINDOW_NAME = 'CRASH DETECTOR'
CAMERA_ID = 0  # Your system is using camera 0

# Create directories if they don't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

print(f"[CONFIG] Using model: {MODEL_PATH}")
print(f"[CONFIG] Camera ID: {CAMERA_ID}")