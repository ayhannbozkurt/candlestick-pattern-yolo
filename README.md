# Stock Market Pattern Detection with YOLOv8

This project uses YOLOv8 for detecting patterns in stock market candlestick charts. The model can identify various technical analysis patterns including 'Head and shoulders bottom,' 'Head and shoulders top,' 'M_Head,' 'StockLine,' 'Triangle,' and 'W_Bottom.'

https://github.com/user-attachments/assets/9c4ace10-a8c8-4c67-b5d4-2675a6eac50e

## Features

- Real-time pattern detection in both images and videos
- Easy integration with existing applications
- Configurable confidence thresholds
- Automatic model downloading from Hugging Face
- Visualization of detected patterns

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ayhannbozkurt/candlestick-pattern-yolo
cd candlestick-pattern-yolo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Hugging Face token:

Create a `.env` file in the project root directory with the following content:
```
HUGGINGFACE_TOKEN=your_huggingface_token
```

You can obtain your Hugging Face token from [Hugging Face account settings](https://huggingface.co/settings/tokens).

## Project Structure

```
stock-pattern-detection/
├── app.py                  # Main application entry point
├── src/
│   ├── model/
│   │   ├── model_loader.py        # Handles model downloading and loading
│   │   └── pattern_detector.py    # Detects patterns in images
│   ├── inference/
│   │   ├── video_processor.py     # Processes videos
│   │   ├── video_inference.py     # Video inference implementation
│   │   └── image_inference.py     # Image inference implementation
│   └── utils/
│       └── logger.py              # Logging utilities
└── requirements.txt        # Project dependencies
```

## Usage

### Quick Start

Run the main application to process a video file:

```bash
python app.py
```

### Image Processing

```python
from src.model.pattern_detector import PatternDetector
import cv2

# Initialize the pattern detector
detector = PatternDetector()

# Read and process an image
image = cv2.imread("path/to/your/image.jpg")
detections = detector.detect_patterns(image, conf_threshold=0.3)

# Draw detections on the image
result_image = detector.draw_detections(image, detections)

# Save or display the result
cv2.imwrite("output.jpg", result_image)
cv2.imshow("Detected Patterns", result_image)
cv2.waitKey(0)

# Print detected patterns
for det in detections:
    print(f"Pattern: {det['pattern']}, Confidence: {det['confidence']:.2f}")
```

### Video Processing

```python
from src.inference.video_processor import VideoProcessor
import cv2

# Initialize video processor
processor = VideoProcessor()

# Process video file
video_path = "input.mp4"
output_path = "output.mp4"

print(f"Processing video: {video_path}")
print("Press 'q' to quit")

# Process and display video frames
for frame in processor.process_video(video_path, output_path, conf_threshold=0.3):
    # Display the frame
    cv2.imshow('Stock Pattern Detection', frame)
    
    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print(f"Output saved to: {output_path}")
```

### Advanced Usage - Custom Model Configuration

You can customize the model configuration by passing parameters:

```python
from src.model.model_loader import ModelLoader
from src.model.pattern_detector import PatternDetector
from src.inference.video_processor import VideoProcessor

# Initialize with custom model ID and filename
model_loader = ModelLoader(
    model_id="foduucom/stockmarket-pattern-detection-yolov8",
    model_filename="best.pt"
)

# Create detector with custom model loader
detector = PatternDetector(model_loader=model_loader)

# Create video processor with custom detector
processor = VideoProcessor(pattern_detector=detector)

# Process with custom confidence threshold
detections = detector.detect_patterns(image, conf_threshold=0.4)
```

### Advanced Video Processing Options

The `VideoInference` class provides additional options for video processing:

```python
from src.inference.video_inference import VideoInference

# Initialize video inference
video_inf = VideoInference()

# Process video with custom parameters
output_path = video_inf.process_video(
    video_path="input.mp4",
    output_path="output.mp4",
    frame_skip=5,              # Process every 5th frame
    conf_threshold=0.3,        # Confidence threshold
    iou_threshold=0.45,        # IOU threshold for NMS
    detection_duration=75      # How long to display detections (in frames)
)
```

## Detection Output Format

The `detect_patterns` method returns a list of detections with the following format:

```python
[
    {
        "pattern": "Head and shoulders top",
        "confidence": 0.95,         # Confidence score between 0 and 1
        "bbox": [x1, y1, x2, y2]    # Bounding box coordinates (top-left, bottom-right)
    },
    # ... more detections
]
```

## Supported Patterns

The model can detect the following candlestick patterns:

- Head and shoulders bottom (ID: 0)
- Head and shoulders top (ID: 1)
- M_Head (ID: 2)
- StockLine (ID: 3)
- Triangle (ID: 4)
- W_Bottom (ID: 5)

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- Hugging Face Hub
- Other dependencies listed in `requirements.txt`

## Troubleshooting

### Model Loading Issues

If you encounter issues loading the model, check:

1. Your Hugging Face token is valid and correctly set
2. You have internet connectivity for the initial model download
3. You have sufficient disk space for the model

### Memory Issues

For processing high-resolution videos, you may need to adjust the frame size:

```python
# Resize frame to reduce memory usage
frame = cv2.resize(frame, (640, 480))
```

## Acknowledgments

- Model trained on [foduucom/stockmarket-pattern-detection-yolov8](https://huggingface.co/foduucom/stockmarket-pattern-detection-yolov8)
- Built with YOLOv8 by Ultralytics
