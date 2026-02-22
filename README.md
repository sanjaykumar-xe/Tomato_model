# Tomato Ripeness Detection using YOLOv8

A computer vision project for detecting and classifying tomato ripeness stages using YOLOv8 object detection model.

## Features

- Detects tomatoes in images and videos
- Classifies ripeness into 3 categories:
  - Green
  - Half-ripened (Red)
  - Ripened
- Real-time inference on webcam or video files
- Trained on custom COCO format dataset

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd train
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Place your dataset in the following structure:
```
C:\Users\user\Desktop\train\
├── _annotations.coco.json
├── IMG_*.jpg (training images)
└── tomato.mp4 (optional test video)
```

## Usage

### 1. Prepare Dataset
Convert COCO annotations to YOLO format and split into train/val sets:
```bash
python prepare_dataset.py
```

### 2. Train Model
Train YOLOv8 on the tomato dataset:
```bash
python train_tomato.py
```

Training parameters:
- Model: YOLOv8n (nano)
- Epochs: 15
- Image size: 640x640
- Batch size: 16

### 3. Run Inference

#### Live Display (OpenCV)
```bash
python live_opencv.py --source tomato.mp4
```

For webcam:
```bash
python live_opencv.py --source 0
```

Controls:
- Press 'q' to quit
- Press 'p' to pause/resume

#### Save Output Video
```bash
python live_inference.py --source tomato.mp4 --output result.mp4
```

Auto-play after processing:
```bash
python live_inference.py --source tomato.mp4 --play
```

#### Live Display (Matplotlib - Alternative)
```bash
python live_display.py --source tomato.mp4
```

### 4. Analyze Dataset
View dataset statistics:
```bash
python analyze_coco.py
```

Simple analysis:
```bash
python analyze_simple.py
```

## Project Structure

```
train/
├── prepare_dataset.py      # Dataset preparation script
├── train_tomato.py          # Model training script
├── live_opencv.py           # Live inference with OpenCV
├── live_inference.py        # Save inference to video
├── live_display.py          # Live inference with matplotlib
├── analyze_coco.py          # Dataset analysis
├── analyze_simple.py        # Simple dataset stats
├── tomato_data.yaml         # YOLO dataset config
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Model Output

Trained models are saved in:
```
d:/train/runs/tomato_yolov8n/weights/
├── best.pt    # Best model checkpoint
└── last.pt    # Last epoch checkpoint
```

## Classes

- 0: Green
- 1: Half-ripened (displayed as "Red")
- 2: Ripened

## Troubleshooting

### OpenCV GUI not working
If you get OpenCV GUI errors, reinstall opencv-python:
```bash
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python
```

Test the GUI:
```bash
python test_opencv_gui.py
```

### Video file not found
Make sure your video path is correct or use webcam with `--source 0`

## License

MIT License

## Acknowledgments

- YOLOv8 by Ultralytics
- COCO dataset format
