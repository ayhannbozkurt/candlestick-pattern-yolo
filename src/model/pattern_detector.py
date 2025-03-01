from typing import List, Dict, Any
import cv2
import numpy as np
from .model_loader import ModelLoader

class PatternDetector:
    def __init__(self, model_loader: ModelLoader = None):
        self.model_loader = model_loader or ModelLoader()
        # Load the model immediately
        self.model = self.model_loader.get_model()
        self.patterns = {
            0: "Head and shoulders bottom",
            1: "Head and shoulders top",
            2: "M_Head",
            3: "StockLine",
            4: "Triangle",
            5: "W_Bottom"
        }
    
    def detect_patterns(self, image: np.ndarray, conf_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Detect patterns in the given image
        Args:
            image: Input image as numpy array
            conf_threshold: Confidence threshold for detection
        Returns:
            List of detected patterns with their details
        """
        results = self.model(image)[0]
        detections = []

        for box in results.boxes:
            confidence = float(box.conf)
            if confidence < conf_threshold:
                continue

            class_id = int(box.cls)
            pattern_name = self.patterns.get(class_id, "Unknown")
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "pattern": pattern_name,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })

        return detections

    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detected patterns on the image
        Args:
            image: Input image
            detections: List of detected patterns
        Returns:
            Image with drawn detections
        """
        img_copy = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            pattern = det["pattern"]
            conf = det["confidence"]
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{pattern}: {conf:.2f}"
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.putText(img_copy, label, (x1, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img_copy 