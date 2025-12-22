import torch
import cv2
import numpy as np
from ultralytics import YOLO
from config import MODEL_PATH, CONFIDENCE, MODEL_DIR
from datetime import datetime
import os


class CrashDetector:
    def __init__(self):
        print("Initializing Crash Detector...")
        
        # Resolve model path
        model_path = self._resolve_model_path(MODEL_PATH)
        print(f"Model path: {model_path}")
        
        try:
            # Load YOLO model with error handling
            print("Loading YOLO model...")
            self.model = YOLO(model_path)
            
            # Auto-select device
            if torch.cuda.is_available():
                self.device = 'cuda'
                print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                print("Using CPU")
            
            self.model.to(self.device)
            
            # Warm up the model with a dummy inference
            self._warm_up_model()
            
            # Detection tracking
            self.last_detection_info = None
            self.detection_count = 0
            self.last_crash_time = None
            
            print(f"✓ Model loaded successfully on {self.device.upper()}")
            print(f"✓ Confidence threshold: {CONFIDENCE}")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("Attempting to download model...")
            self._download_model()
    
    def _resolve_model_path(self, model_name):
        """Resolve the model file path."""
        # Check if it's a local file
        if os.path.exists(model_name):
            return model_name
        
        # Check in models directory
        model_in_dir = os.path.join(MODEL_DIR, model_name)
        if os.path.exists(model_in_dir):
            return model_in_dir
        
        # Return the model name (YOLO will download it if needed)
        return model_name
    
    def _download_model(self):
        """Attempt to download the model."""
        try:
            print("Downloading YOLOv11n model...")
            # YOLO will automatically download the model
            self.model = YOLO('yolo11n.pt')
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            print("✓ Model downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download model: {e}")
            raise
    
    def _warm_up_model(self):
        """Warm up the model with a dummy inference."""
        try:
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_frame, verbose=False, device=self.device)
            print("✓ Model warmed up successfully")
        except Exception as e:
            print(f"⚠️ Model warmup failed: {e}")
    
    def detect(self, frame):
        """Detect objects in frame and return annotated image."""
        if frame is None or frame.size == 0:
            print("⚠️ Empty frame received")
            return frame
        
        try:
            # Run inference
            results = self.model(frame, 
                                 verbose=False, 
                                 conf=CONFIDENCE,
                                 device=self.device,
                                 max_det=15,  # Reasonable limit for traffic scenes
                                 classes=[0, 1, 2, 3, 5, 7, 9])  # Relevant classes
            
            # Check for potential crash scenarios
            self._analyze_detections(results[0])
            
            # Return annotated frame
            annotated = results[0].plot()
            return annotated
            
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            return frame
    
    def _analyze_detections(self, result):
        """Analyze detections for potential crash scenarios."""
        if not hasattr(result, 'boxes') or result.boxes is None:
            self.last_detection_info = None
            return
        
        try:
            # Extract detection data
            boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, 'is_cuda') and result.boxes.xyxy.is_cuda else result.boxes.xyxy.numpy()
            confidences = result.boxes.conf.cpu().numpy() if hasattr(result.boxes.conf, 'is_cuda') and result.boxes.conf.is_cuda else result.boxes.conf.numpy()
            classes = result.boxes.cls.cpu().numpy() if hasattr(result.boxes.cls, 'is_cuda') and result.boxes.cls.is_cuda else result.boxes.cls.numpy()
            
            # Count objects by type
            car_count = np.sum(classes == 2)
            truck_count = np.sum(classes == 7)
            bus_count = np.sum(classes == 5)
            person_count = np.sum(classes == 0)
            motorcycle_count = np.sum(classes == 3)
            
            # Check for potential crashes
            potential_crash, crash_info = self._check_crash_scenarios(boxes, classes, confidences)
            
            # Store detection info
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.last_detection_info = {
                'timestamp': timestamp,
                'objects_detected': len(boxes),
                'cars': int(car_count),
                'trucks': int(truck_count),
                'buses': int(bus_count),
                'persons': int(person_count),
                'motorcycles': int(motorcycle_count),
                'potential_crash': potential_crash,
                'crash_info': crash_info,
                'confidence_avg': float(np.mean(confidences)) if len(confidences) > 0 else 0.0
            }
            
            self.detection_count += 1
            
            if potential_crash:
                self.last_crash_time = timestamp
        
        except Exception as e:
            print(f"⚠️ Error analyzing detections: {e}")
            self.last_detection_info = None
    
    def _check_crash_scenarios(self, boxes, classes, confidences, proximity_threshold=0.4):
        """Check for various crash scenarios."""
        if len(boxes) < 2:
            return False, []
        
        crash_events = []
        
        # Check proximity between all pairs of objects
        for i in range(len(boxes)):
            box_i = boxes[i]
            class_i = classes[i]
            
            for j in range(i + 1, len(boxes)):
                box_j = boxes[j]
                class_j = classes[j]
                
                # Calculate overlap
                iou = self._calculate_iou(box_i, box_j)
                
                if iou > proximity_threshold:
                    # Objects are overlapping significantly
                    crash_type = self._determine_crash_type(class_i, class_j)
                    confidence = min(confidences[i], confidences[j])
                    
                    crash_events.append({
                        'type': crash_type,
                        'iou': float(iou),
                        'confidence': float(confidence),
                        'object1': int(class_i),
                        'object2': int(class_j)
                    })
        
        return len(crash_events) > 0, crash_events
    
    def _determine_crash_type(self, class1, class2):
        """Determine the type of crash based on object classes."""
        # Map class IDs to names
        class_names = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        name1 = class_names.get(int(class1), f'class_{class1}')
        name2 = class_names.get(int(class2), f'class_{class2}')
        
        return f"{name1}-{name2}_collision"
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def get_status(self):
        """Get current detector status."""
        return {
            'device': self.device,
            'detection_count': self.detection_count,
            'last_crash_time': self.last_crash_time,
            'model_loaded': hasattr(self, 'model') and self.model is not None
        }