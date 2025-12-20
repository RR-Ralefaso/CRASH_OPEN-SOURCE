import cv2
import time
from crash import CrashDetector
from config import CAMERA_ID, WINDOW_NAME

def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    
    if not cap.isOpened():
        print("Camera error! Try CAMERA_ID = 0, 1, 2 in config.py")
        return
    
    # FIX BLURRY CAMERA - Set these properties FIRST
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Higher native res
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # Faster shutter
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)      # Brighter
    cap.set(cv2.CAP_PROP_CONTRAST, 120)        # Sharper
    cap.set(cv2.CAP_PROP_SHARPNESS, 200)       # CRISP edges
    
    # Get actual properties after setting
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera: {width}x{height} @ {fps:.1f}FPS")
    
    detector = CrashDetector()
    
    print("CRASH DETECTOR - Crystal Clear! 'q' to quit, 'd' toggle detection")
    detect_on = True
    fps_counter = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # RAW CAMERA (NO RESIZE) = CLEAR IMAGE
        display_frame = frame.copy()
        
        if detect_on:
            # Resize ONLY for YOLO (not display)
            small_frame = cv2.resize(frame, (640, 480))
            annotated_small = detector.detect(small_frame)
            # Scale back up for display
            display_frame = cv2.resize(annotated_small, (width, height))
        else:
            # Pure raw camera feed
            cv2.putText(display_frame, "RAW CAMERA (Press 'd' for detection)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # FPS overlay
        fps_counter += 1
        if fps_counter % 30 == 0:
            fps = 30 / (time.time() - start_time)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, height-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            start_time = time.time()
        
        cv2.imshow(WINDOW_NAME, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            detect_on = not detect_on
            print("Detection:", "ON" if detect_on else "OFF")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
