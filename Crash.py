
from ultralytics import YOLO as yl
from config import MODEL_PATH, CONFIDENCE


class CrashDetector:
    def __init__(self):
        self.model = yl(MODEL_PATH)
    
    def detect(self, frame):
        results = self.model(frame, verbose=False, conf=CONFIDENCE)
        return results[0].plot()
