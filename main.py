import cv2
from Crash import CrashDetector
from config import CAMERA_ID, WINDOW_NAME


def main():
    detector = CrashDetector()
    cap = cv2.VideoCapture(CAMERA_ID)
    
    if not cap.isOpened():
        print("Camera error!")
        return
    
    print("YOLOv11x multi-file project started. 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated = detector.detect(frame)
        cv2.imshow(WINDOW_NAME, annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
