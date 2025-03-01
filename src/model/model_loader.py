from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os
import torch

class ModelLoader:
    def __init__(self, model_id="foduucom/stockmarket-pattern-detection-yolov8", model_filename="best.pt"):
        """
        Initialize the ModelLoader.
        
        Args:
            model_id (str): Hugging Face model ID
            model_filename (str): Name of the model file
        """
        self.model_id = model_id
        self.model_filename = model_filename
        self.model = None
        
    def download_model(self):
        """
        Download the model from Hugging Face Hub.
        
        Returns:
            str: Path to the downloaded model
        """
        try:
            # Get token from environment variable
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            if not hf_token:
                raise Exception("HUGGINGFACE_TOKEN environment variable is not set")

            model_path = hf_hub_download(
                repo_id=self.model_id,
                filename=self.model_filename,
                token=hf_token
            )
            return model_path
        except Exception as e:
            raise Exception(f"Error downloading model: {str(e)}")
    
    def load_model(self, conf_threshold=0.25, iou_threshold=0.45):
        """
        Load the YOLO model with specified parameters.
        
        Args:
            conf_threshold (float): Confidence threshold for predictions
            iou_threshold (float): IOU threshold for NMS
            
        Returns:
            YOLO: Loaded model instance
        """
        try:
            model_path = self.download_model()
            
            # Method 1: Set environment variable to use pickle safely (backward compatibility)
            # This is more reliable than other methods for ultralytics models
            os.environ["PYTORCH_ENABLE_UNSAFE_LOAD"] = "1"
            self.model = YOLO(model_path)
            
            # Optional: Unset the environment variable after loading
            del os.environ["PYTORCH_ENABLE_UNSAFE_LOAD"]
            
            # Set model parameters
            self.model.overrides['conf'] = conf_threshold
            self.model.overrides['iou'] = iou_threshold
            self.model.overrides['agnostic_nms'] = False
            self.model.overrides['max_det'] = 1000
            
            return self.model
        except Exception as e:
            # If first method fails, try with monkey-patching torch.load
            try:
                print(f"Loading with environment variable failed: {str(e)}")
                print("Attempting to load with monkey-patched torch.load...")
                
                # Save the original torch.load
                original_load = torch.load
                
                # Define a monkey-patched version that forces weights_only=False
                def patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                # Apply the monkey patch
                torch.load = patched_load
                
                # Try loading the model with the patched load function
                self.model = YOLO(model_path)
                
                # Restore the original torch.load
                torch.load = original_load
                
                # Set model parameters
                self.model.overrides['conf'] = conf_threshold
                self.model.overrides['iou'] = iou_threshold
                self.model.overrides['agnostic_nms'] = False
                self.model.overrides['max_det'] = 1000
                
                return self.model
            except Exception as e2:
                raise Exception(f"Error loading model: {str(e2)}")
    
    def get_model(self):
        """
        Get the loaded model instance.
        
        Returns:
            YOLO: Loaded model instance
        """
        if self.model is None:
            return self.load_model()
        return self.model