# YOLO Stock Market Pattern Detection

This project uses YOLOv8 for detecting patterns in stock market candlestick charts. The model is trained to recognize various patterns including 'Head and shoulders bottom,' 'Head and shoulders top,' 'M_Head,' 'StockLine,' 'Triangle,' and 'W_Bottom.'

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd yolo-pattern
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Linux/Mac
export HUGGINGFACE_TOKEN="your_huggingface_token"

# Windows
set HUGGINGFACE_TOKEN=your_huggingface_token
```

You can get your Hugging Face token from your [Hugging Face account settings](https://huggingface.co/settings/tokens).

## Usage

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

# Method 1: Process and save video without display
for frame in processor.process_video(video_path, output_path):
    pass  # Video is automatically saved to output_path

# Method 2: Process, display and save video
print("Processing video... Press 'q' to quit")
for frame in processor.process_video(video_path, output_path):
    # Display the frame
    cv2.imshow('Stock Pattern Detection', frame)
    
    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

### Custom Model Configuration

You can customize the model configuration by passing parameters:

```python
from src.model.model_loader import ModelLoader
from src.model.pattern_detector import PatternDetector
from src.inference.video_processor import VideoProcessor

# Initialize with custom model ID and filename
model_loader = ModelLoader(
    model_id="foduucom/stockmarket-pattern-detection-yolov8",
    model_filename="model.pt"
)

# For image processing
detector = PatternDetector(model_loader=model_loader)

# For video processing
processor = VideoProcessor(pattern_detector=PatternDetector(model_loader=model_loader))

# Process with custom confidence threshold
detections = detector.detect_patterns(image, conf_threshold=0.3)
```

## Model Information

The model is hosted on Hugging Face Hub and will be automatically downloaded when you first use it. Make sure you have an internet connection for the initial download.

## Supported Patterns

The model can detect the following patterns:
- Head and shoulders bottom
- Head and shoulders top
- M_Head
- StockLine
- Triangle
- W_Bottom

## Requirements

See `requirements.txt` for a full list of dependencies.

## Output Format

### Image Detection Output
```python
[
    {
        "pattern": "Pattern Name",
        "confidence": 0.95,  # Confidence score between 0 and 1
        "bbox": [x1, y1, x2, y2]  # Bounding box coordinates
    },
    # ... more detections
]
```

## Notes
- For video processing, the output video will be saved in MP4 format
- Press 'q' to quit video display
- Default confidence threshold is 0.3
- The model automatically downloads from Hugging Face Hub on first use # candlestick-pattern-yolo
