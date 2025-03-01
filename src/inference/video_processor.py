import cv2
import numpy as np
from typing import Optional, Tuple, Generator
from pathlib import Path
from ..model.pattern_detector import PatternDetector

class VideoProcessor:
    def __init__(self, pattern_detector: Optional[PatternDetector] = None):
        """
        Initialize video processor
        Args:
            pattern_detector: Pattern detector instance
        """
        self.pattern_detector = pattern_detector or PatternDetector()

    def get_video_info(self, video_path: str) -> Tuple[int, int, int, float]:
        """
        Get video information
        Args:
            video_path: Path to video file
        Returns:
            Tuple of (width, height, total_frames, fps)
        """
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return width, height, total_frames, fps

    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     conf_threshold: float = 0.3) -> Generator[np.ndarray, None, None]:
        """
        Process video and detect patterns
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            conf_threshold: Confidence threshold for detection
        Yields:
            Processed frames with detections
        """
        cap = cv2.VideoCapture(video_path)
        width, height, _, fps = self.get_video_info(video_path)

        # Setup video writer if output path is provided
        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect patterns
            detections = self.pattern_detector.detect_patterns(frame, conf_threshold)
            
            # Draw detections
            processed_frame = self.pattern_detector.draw_detections(frame, detections)

            if writer:
                writer.write(processed_frame)
            
            yield processed_frame

        cap.release()
        if writer:
            writer.release()

    def process_frame(self, frame: np.ndarray, conf_threshold: float = 0.3) -> Tuple[np.ndarray, list]:
        """
        Process a single frame
        Args:
            frame: Input frame
            conf_threshold: Confidence threshold for detection
        Returns:
            Tuple of (processed frame, detections)
        """
        detections = self.pattern_detector.detect_patterns(frame, conf_threshold)
        processed_frame = self.pattern_detector.draw_detections(frame, detections)
        return processed_frame, detections 