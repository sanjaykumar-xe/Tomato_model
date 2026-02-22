import cv2
from ultralytics import YOLO
import argparse
import os

# Path to the trained model
# Adjust this if the specific run folder name changes (e.g., tomato_yolov8n2)
MODEL_PATH = r'd:/train/runs/tomato_yolov8n/weights/best.pt'

def run_inference(source='d:/train/tomato.mp4', output='output_detection.mp4'):
    """
    Run inference on a video source and save the output.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please ensure training is complete.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Override class names as per user request
    # 0: green, 1: half-ripened -> red, 2: ripened
    model.model.names[1] = 'red'

    print(f"Opening video source {source}...")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))
    
    print(f"Processing video: {total_frames} frames at {fps} FPS")
    print(f"Output will be saved to: {output}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame, verbose=False) 

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Write the frame
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:  # Progress update every 30 frames
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")

    cap.release()
    out.release()
    print(f"\nDone! Output saved to: {output}")
    print(f"Processed {frame_count} frames total.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tomato Detection on Video')
    parser.add_argument('--source', type=str, default='tomato.mp4', help='Path to input video file')
    parser.add_argument('--output', type=str, default='output_detection.mp4', help='Path to output video file')
    args = parser.parse_args()
    
    run_inference(args.source, args.output)
