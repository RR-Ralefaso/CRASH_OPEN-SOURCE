import cv2
import time
from Crash import CrashDetector
from config import CAMERA_ID, WINDOW_NAME
from outputting import DetectionLogger

def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    
    if not cap.isOpened():
        print(f"Camera error! Could not open camera with ID: {CAMERA_ID}")
        print("Try CAMERA_ID = 0, 1, 2 in config.py")
        return
    
    # Initialize detection logger
    logger = DetectionLogger()
    
    # Set camera to maximum resolution
    max_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    max_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
    cap.set(cv2.CAP_PROP_CONTRAST, 120)
    cap.set(cv2.CAP_PROP_SHARPNESS, 200)
    
    # Get actual properties after setting
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera initialized: {width}x{height} @ {fps:.1f} FPS")
    
    # Initialize detector
    detector = CrashDetector()
    logger.log_start_session()
    
    print("Crystal Clear! 'q' to quit, 'd' toggle detection, 's' to save screenshot")
    
    # Performance tracking
    detect_on = True
    fps_timer = time.time()
    frame_count = 0
    fps_value = 0
    
    # Resize target for YOLO (optimized size)
    YOLO_WIDTH, YOLO_HEIGHT = 640, 480
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        frame_count += 1
        display_frame = frame.copy()
        
        if detect_on:
            # Optimized: Only resize for YOLO inference
            small_frame = cv2.resize(frame, (YOLO_WIDTH, YOLO_HEIGHT), 
                                     interpolation=cv2.INTER_LINEAR)
            
            # Perform detection
            annotated_small = detector.detect(small_frame)
            
            # Scale back for display
            display_frame = cv2.resize(annotated_small, (width, height), 
                                      interpolation=cv2.INTER_LINEAR)
            
            # Check for crashes (assuming detector provides crash info)
            # You'll need to modify Crash.py to return detection info
            if hasattr(detector, 'last_detection_info'):
                logger.log_detection(detector.last_detection_info)
        else:
            cv2.putText(display_frame, "RAW CAMERA (Press 'd' for detection)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update FPS every second
        current_time = time.time()
        if current_time - fps_timer >= 1.0:
            fps_value = frame_count / (current_time - fps_timer)
            frame_count = 0
            fps_timer = current_time
        
        # Display FPS
        cv2.putText(display_frame, f"FPS: {fps_value:.1f}", (10, height-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display mode
        cv2.putText(display_frame, f"Mode: {'DETECT' if detect_on else 'RAW'}", 
                   (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if detect_on else (255, 255, 0), 2)
        
        cv2.imshow(WINDOW_NAME, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            detect_on = not detect_on
            mode = "Detection: ON" if detect_on else "Detection: OFF"
            print(mode)
            logger.log_event(mode)
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"Screenshot saved: {filename}")
            logger.log_event(f"Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    logger.log_end_session()
    print("Application closed")


if __name__ == "__main__":
    main()