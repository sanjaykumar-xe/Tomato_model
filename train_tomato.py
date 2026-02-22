from ultralytics import YOLO
import os

# Paths
DATA_YAML = "tomato_data.yaml"
PROJECT_DIR = "d:/train/runs"

def train():
    # Load model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=DATA_YAML,
        epochs=15,
        imgsz=640,
        project=PROJECT_DIR,
        name='tomato_yolov8n',
        batch=16,
        cache=True, # Use CPU if no GPU available, safer default for automation
        save=True
    )
    
    print("Training Completed.")
    print(f"Metrics: {results.box.map}")

if __name__ == '__main__':
    train()
