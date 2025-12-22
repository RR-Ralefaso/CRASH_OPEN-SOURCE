import cv2
import time
import traceback
from Crash import CrashDetector
from config import CAMERA_ID, WINDOW_NAME, LOG_DIR, SCREENSHOT_DIR, YOLO_INPUT_SIZE
from outputting import DetectionLogger


def initialize_camera(camera_id):
    """Initialize camera with error handling."""
    print(f"Initializing camera {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"✗ Failed to open camera {camera_id}")
        return None
    
    # Get default properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✓ Camera initialized: {width}x{height} @ {fps:.1f} FPS")
    
    # Try to set better properties (optional)
    try:
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce buffer for lower latency
    except:
        pass  # Some cameras don't support these settings
    
    return cap


def main():
    print("=" * 50)
    print("CRASH DETECTION SYSTEM")
    print("=" * 50)
    
    # Initialize logger
    logger = DetectionLogger()
    
    # Initialize camera
    cap = initialize_camera(CAMERA_ID)
    if cap is None:
        print("Trying camera 0...")
        cap = initialize_camera(0)
        if cap is None:
            print("No camera found. Exiting.")
            return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize detector
    try:
        detector = CrashDetector()
    except Exception as e:
        print(f"✗ Failed to initialize detector: {e}")
        print("Exiting...")
        cap.release()
        return
    
    # Start session
    session_id = logger.log_start_session()
    print(f"Session ID: {session_id}")
    
    # Performance tracking
    detect_on = True
    fps_timer = time.time()
    frame_count = 0
    fps_value = 0
    frame_number = 0
    
    print("\nControls:")
    print("  'q' - Quit")
    print("  'd' - Toggle detection")
    print("  's' - Save screenshot")
    print("  'p' - Pause/Resume")
    print("=" * 50)
    print("Starting detection...\n")
    
    paused = False
    
    try:
        while True:
            if not paused:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("✗ Failed to capture frame")
                    break
                
                frame_count += 1
                frame_number += 1
                display_frame = frame.copy()
                
                if detect_on:
                    try:
                        # Resize for YOLO inference
                        small_frame = cv2.resize(frame, YOLO_INPUT_SIZE)
                        
                        # Perform detection
                        annotated_small = detector.detect(small_frame)
                        
                        # Scale back for display
                        display_frame = cv2.resize(annotated_small, (width, height))
                        
                        # Log detection if available
                        if detector.last_detection_info:
                            logger.log_detection(detector.last_detection_info, frame_number)
                            
                            # Display crash warning if detected
                            if detector.last_detection_info.get('potential_crash', False):
                                cv2.putText(display_frame, "⚠️ POTENTIAL CRASH!", 
                                           (width//2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                           1.5, (0, 0, 255), 3)
                    
                    except Exception as e:
                        print(f"⚠️ Detection error: {e}")
                        # Fallback to raw frame
                        cv2.putText(display_frame, "DETECTION ERROR", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "RAW CAMERA", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update FPS every second
                current_time = time.time()
                if current_time - fps_timer >= 1.0:
                    fps_value = frame_count / (current_time - fps_timer)
                    frame_count = 0
                    fps_timer = current_time
                
                # Display info overlay
                cv2.putText(display_frame, f"FPS: {fps_value:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                mode_text = "DETECT" if detect_on else "RAW"
                mode_color = (0, 255, 0) if detect_on else (255, 255, 0)
                cv2.putText(display_frame, f"Mode: {mode_text}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
                
                # Display detection count
                cv2.putText(display_frame, f"Detections: {detector.detection_count}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if paused:
                    cv2.putText(display_frame, "PAUSED", 
                               (width//2 - 50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 0, 255), 3)
            
            # Display the frame
            cv2.imshow(WINDOW_NAME, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('d'):
                detect_on = not detect_on
                mode = "Detection: ON" if detect_on else "Detection: OFF"
                print(mode)
                logger.log_event(mode)
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{SCREENSHOT_DIR}/screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Screenshot saved: {filename}")
                logger.log_event(f"Screenshot saved: {filename}")
            elif key == ord('p'):
                paused = not paused
                state = "PAUSED" if paused else "RESUMED"
                print(f"Video {state}")
                logger.log_event(f"Video {state}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n" + "=" * 50)
        print("Cleaning up...")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Log session end
        logger.log_end_session()
        
        # Show session summary
        stats = logger.get_detection_stats()
        print(f"\nSession Summary:")
        print(f"  Total Detections: {stats['total_detections']}")
        print(f"  Potential Crashes: {stats['potential_crashes']}")
        print(f"  Log Directory: {LOG_DIR}")
        
        print("\n✓ Application closed")


if __name__ == "__main__":
    main()