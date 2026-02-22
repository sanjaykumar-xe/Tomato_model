import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import os

# Path to the trained model
MODEL_PATH = r'd:/train/runs/tomato_yolov8n/weights/best.pt'

def run_live_display(source='tomato.mp4'):
    """
    Display video with detections in real-time using matplotlib.
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

    # Setup matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to read first frame")
        return
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, verbose=False)
    annotated_frame = results[0].plot()
    
    im = ax.imshow(annotated_frame)
    plt.tight_layout()
    
    def update_frame(frame_num):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            ret, frame = cap.read()
            if not ret:
                return im,
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = model(frame_rgb, verbose=False)
        annotated_frame = results[0].plot()
        
        im.set_array(annotated_frame)
        ax.set_title(f'Tomato Detection - Frame {frame_num}', fontsize=14)
        return im,
    
    print("Starting live display... Close the window to stop.")
    anim = FuncAnimation(fig, update_frame, interval=33, blit=True, cache_frame_data=False)
    plt.show()
    
    cap.release()
    print("Display closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Live Tomato Detection Display')
    parser.add_argument('--source', type=str, default='tomato.mp4', help='Path to video file or 0 for webcam')
    args = parser.parse_args()
    
    # Handle webcam
    src = int(args.source) if args.source.isdigit() else args.source
    
    run_live_display(src)
