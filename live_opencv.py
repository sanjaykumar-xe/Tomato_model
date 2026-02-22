import cv2
from ultralytics import YOLO
import argparse
import os

# Path to the trained model
MODEL_PATH = r'd:/train/runs/tomato_yolov8n/weights/best.pt'

def run_live_inference(source='tomato.mp4'):
    """
    Run real-time inference with live display using OpenCV.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please ensure training is complete.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Override class names
    model.model.names[1] = 'red'

    print(f"Opening video source {source}...")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Press 'q' to quit, 'p' to pause/resume")
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video. Restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Run inference
            results = model(frame, verbose=False)
            
            # Get annotated frame
            annotated_frame = results[0].plot()
        
        # Display the frame
        cv2.imshow('Tomato Detection (Green/Red/Ripened) - Press Q to quit', annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Live Tomato Detection with OpenCV')
    parser.add_argument('--source', type=str, default='tomato.mp4', help='Video source: path to video file or 0 for webcam')
    args = parser.parse_args()
    
    # Handle webcam
    src = int(args.source) if args.source.isdigit() else args.source
    
    run_live_inference(src)
