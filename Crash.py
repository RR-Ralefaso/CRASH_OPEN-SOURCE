import torch
import numpy as np
from ultralytics import YOLO as yl
from config import MODEL_PATH, CONFIDENCE


class CrashDetector:
    def __init__(self):
        print("Loading YOLO model...")
        self.model = yl(MODEL_PATH)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Model loaded on {self.model.device}")
    
    def detect(self, frame):
        if frame is None or frame.size == 0:
            return frame
        # Real-time optimized inference
        results = self.model(frame, verbose=False, conf=CONFIDENCE, device=self.model.device)
        return results[0].plot()
