import cv2
from ultralyticsplus import render_result
from ..model.model_loader import ModelLoader

class ImageInference:
    def __init__(self, model_loader=None):
        """
        Initialize the ImageInference class.
        
        Args:
            model_loader (ModelLoader, optional): Instance of ModelLoader class
        """
        self.model_loader = model_loader if model_loader else ModelLoader()
        self.model = None
        
    def load_model(self):
        """Load the model if not already loaded."""
        if self.model is None:
            self.model = self.model_loader.get_model()
    
    def process_image(self, image_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Process an image and return the predictions.
        
        Args:
            image_path (str): Path to the input image
            conf_threshold (float): Confidence threshold for predictions
            iou_threshold (float): IOU threshold for NMS
            
        Returns:
            numpy.ndarray: Rendered image with predictions
        """
        # Load model if not loaded
        self.load_model()
        
        # Set model parameters
        self.model.overrides['conf'] = conf_threshold
        self.model.overrides['iou'] = iou_threshold
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['max_det'] = 1000
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at path: {image_path}")
        
        # Perform inference
        results = self.model.predict(image)
        
        # Render results
        rendered_image = render_result(
            model=self.model,
            image=image,
            result=results[0]
        )
        
        return rendered_image 