import cv2
from collections import deque
from ..model.model_loader import ModelLoader

class VideoInference:
    def __init__(self, model_loader=None):
        """
        Initialize the VideoInference class.
        
        Args:
            model_loader (ModelLoader, optional): Instance of ModelLoader class
        """
        self.model_loader = model_loader if model_loader else ModelLoader()
        self.model = None
        self.active_detections = deque(maxlen=100)  # Store active detections
        
    def load_model(self):
        """Load the model if not already loaded."""
        if self.model is None:
            self.model = self.model_loader.get_model()
    
    def process_video(self, video_path, output_path, frame_skip=5, conf_threshold=0.25, iou_threshold=0.45, detection_duration=75):
        """
        Process a video and save the annotated output.
        
        Args:
            video_path (str): Path to the input video
            output_path (str): Path to save the output video
            frame_skip (int): Number of frames to skip between processing
            conf_threshold (float): Confidence threshold for predictions
            iou_threshold (float): IOU threshold for NMS
            detection_duration (int): Number of frames to keep detections visible (75 frames ≈ 2.5 seconds at 30fps)
            
        Returns:
            str: Path to the output video
        """
        # Load model if not loaded
        self.load_model()
        
        # Set model parameters
        self.model.overrides['conf'] = conf_threshold
        self.model.overrides['iou'] = iou_threshold
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['max_det'] = 1000
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video at path: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        self.active_detections.clear()  # Clear any existing detections
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # Create a copy of the frame for visualization
                annotated_frame = frame.copy()
                
                # Only run detection on specified frames (to save processing time)
                if frame_count % frame_skip == 0:
                    # Perform inference
                    results = self.model.predict(frame)
                    
                    # Process new detections
                    if len(results) > 0 and hasattr(results[0], 'boxes'):
                        boxes = results[0].boxes
                        for box in boxes:
                            # Get detection info
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            
                            # Get class name
                            cls_name = results[0].names[cls_id]
                            
                            # Add to active detections with lifespan
                            self.active_detections.append({
                                'box': (int(x1), int(y1), int(x2), int(y2)),
                                'conf': conf,
                                'cls_id': cls_id,
                                'cls_name': cls_name,
                                'end_frame': frame_count + detection_duration
                            })
                
                # Draw all active detections that haven't expired
                active_detections_updated = deque(maxlen=100)
                for det in self.active_detections:
                    if det['end_frame'] >= frame_count:
                        # Detection is still active, draw it
                        x1, y1, x2, y2 = det['box']
                        conf = det['conf']
                        cls_name = det['cls_name']
                        
                        # Draw box
                        color = self.get_color(det['cls_id'])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Calculate text position and size
                        text = f"{cls_name} {conf:.2f}"
                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        
                        # Draw text background
                        cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + text_width, y1), color, -1)
                        
                        # Draw text
                        cv2.putText(annotated_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        # Keep this detection
                        active_detections_updated.append(det)
                
                # Update active detections (remove expired ones)
                self.active_detections = active_detections_updated
                
                # Write the annotated frame to the output video
                out.write(annotated_frame)
                
                frame_count += 1
                
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        return output_path
    
    def get_color(self, class_id):
        """
        Get color for visualization based on class id.
        
        Args:
            class_id: Class ID
            
        Returns:
            Color as BGR tuple
        """
        # Define a list of colors for different classes
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (128, 128, 0),  # Teal
            (0, 128, 128),  # Brown
            (128, 0, 128),  # Purple
            (255, 165, 0)   # Orange
        ]
        
        return colors[class_id % len(colors)]