import cv2
import time
import numpy as np
from crash import CrashDetector  # Fixed import
from config import CAMERA_ID, WINDOW_NAME


def main():
    detector = CrashDetector()
    cap = cv2.VideoCapture(CAMERA_ID)
    
    if not cap.isOpened():
        print("Camera error! Try CAMERA_ID = 0, 1, or 2 in config.py")
        return
    
    # Optimize camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("CRASH started! Press 'q' to quit.")
    fps_counter = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost camera feed!")
            break
        
        # Resize for speed
        frame = cv2.resize(frame, (640, 480))
        
        # Detect crashes
        annotated = detector.detect(frame)
        
        # FPS overlay
        fps_counter += 1
        if fps_counter % 30 == 0:
            fps = 30 / (time.time() - start_time)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            start_time = time.time()
        
        cv2.imshow(WINDOW_NAME, annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(" Crash stopped!")

if __name__ == "__main__":
    main()
