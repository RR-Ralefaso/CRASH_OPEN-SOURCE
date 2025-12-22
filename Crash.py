import torch
import cv2
import numpy as np
from ultralytics import YOLO
from config import MODEL_PATH, CONFIDENCE
from datetime import datetime


class CrashDetector:
    def __init__(self):
        print(f"Loading YOLO model from {MODEL_PATH}...")
        self.model = YOLO(MODEL_PATH)
        # Auto-select device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Detection tracking
        self.last_detection_info = None
        self.detection_count = 0
        
        print(f"Model loaded on {self.device.upper()}")
        print(f"Using confidence threshold: {CONFIDENCE}")
    
    def detect(self, frame):
        """Detect objects in frame and return annotated image."""
        if frame is None or frame.size == 0:
            return frame
        
        try:
            # Run inference with optimized settings
            results = self.model(frame, 
                                 verbose=False, 
                                 conf=CONFIDENCE,
                                 device=self.device,
                                 half=True if self.device == 'cuda' else False,  # FP16 on GPU
                                 max_det=10,  # Limit detections for speed
                                 classes=[0, 1, 2, 3, 5, 7])  # Common vehicle/person classes
            
            # Check for potential crash scenarios
            self._analyze_detections(results[0])
            
            # Return annotated frame
            annotated = results[0].plot()
            return annotated
            
        except Exception as e:
            print(f"Detection error: {e}")
            return frame
    
    def _analyze_detections(self, result):
        """Analyze detections for potential crash scenarios."""
        if not result.boxes:
            self.last_detection_info = None
            return
        
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy.is_cuda else result.boxes.xyxy.numpy()
        confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf.is_cuda else result.boxes.conf.numpy()
        classes = result.boxes.cls.cpu().numpy() if result.boxes.cls.is_cuda else result.boxes.cls.numpy()
        
        # Count objects
        car_count = np.sum(classes == 2)
        truck_count = np.sum(classes == 7)
        person_count = np.sum(classes == 0)
        
        # Check for proximity between objects (simplified crash detection)
        potential_crash = self._check_proximity(boxes, classes)
        
        # Store detection info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_detection_info = {
            'timestamp': timestamp,
            'objects_detected': len(boxes),
            'cars': int(car_count),
            'trucks': int(truck_count),
            'persons': int(person_count),
            'potential_crash': potential_crash,
            'confidence_avg': float(np.mean(confidences)) if len(confidences) > 0 else 0.0
        }
        
        self.detection_count += 1
    
    def _check_proximity(self, boxes, classes, threshold=0.3):
        """Check if objects are too close to each other."""
        if len(boxes) < 2:
            return False
        
        # Calculate IoU between all pairs of boxes
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = self._calculate_iou(boxes[i], boxes[j])
                if iou > threshold:
                    # Check if these are different types of objects
                    if classes[i] != classes[j]:
                        return True
        return False
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0