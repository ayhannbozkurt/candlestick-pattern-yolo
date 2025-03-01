import cv2
from src.inference.video_processor import VideoProcessor

def main():
    # Initialize video processor
    processor = VideoProcessor()
    
    # Process video file
    video_path = "thy-5x.mp4"
    output_path = "output.mp4"
    
    print(f"Processing video: {video_path}")
    print("Press 'q' to quit")
    
    # Process and display video frames
    for frame in processor.process_video(video_path, output_path):
        # Display the frame
        cv2.imshow('Stock Pattern Detection', frame)
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main() 