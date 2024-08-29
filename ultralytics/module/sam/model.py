from pathlib import Path
from ultralytics.engine.model import Model
from ultralytics.utils.torch_utils import model_info
from .build import build_sam
from .predict import Predictor

class SAM(Model):
    """
    SAM (Segment Anything Model) Interface Class.

    The SAM class provides an interface for prompt-based image segmentation. It supports various prompt types 
    including bounding boxes, points, and labels. SAM is designed for zero-shot performance and utilizes the SA-1B 
    dataset for training.
    """

    def __init__(self, model_path="sam_b.pt") -> None:
        """
        Initialize the SAM model with a pre-trained file.

        Args:
            model_path (str): The path to the pre-trained model file. Expected to be in .pt or .pth format.

        Raises:
            NotImplementedError: If the model file does not have a .pt or .pth extension.
        """
        if model_path and Path(model_path).suffix not in (".pt", ".pth"):
            raise NotImplementedError("The SAM model must be in *.pt or *.pth format.")
        super().__init__(model=model_path, task="segment")

    def _load(self, weights_path: str, task=None):
        """
        Load the model weights from a specified file.

        Args:
            weights_path (str): Path to the weights file.
            task (str, optional): The task for which the model is loaded. Defaults to None.
        """
        self.model = build_sam(weights_path)

    def predict(self, input_source, real_time=False, bounding_boxes=None, points=None, labels=None, **additional_args):
        """
        Perform segmentation prediction on the provided input.

        Args:
            input_source (str or PIL.Image or numpy.ndarray): Path to an image or video file, or an image object.
            real_time (bool, optional): If True, enables real-time processing. Defaults to False.
            bounding_boxes (list, optional): Coordinates for bounding box prompts. Defaults to None.
            points (list, optional): Points for segmentation prompts. Defaults to None.
            labels (list, optional): Labels for segmentation prompts. Defaults to None.

        Returns:
            list: The model's predictions.
        """
        default_settings = dict(confidence_threshold=0.25, task="segment", mode="predict", image_size=1024)
        additional_args.update(default_settings)
        prompts = dict(bboxes=bounding_boxes, points=points, labels=labels)
        return super().predict(input_source, real_time, prompts=prompts, **additional_args)

    def __call__(self, input_source=None, real_time=False, bounding_boxes=None, points=None, labels=None, **additional_args):
        """
        Shortcut method for calling the 'predict' method.

        Args:
            input_source (str or PIL.Image or numpy.ndarray): Path to an image or video file, or an image object.
            real_time (bool, optional): If True, enables real-time processing. Defaults to False.
            bounding_boxes (list, optional): Coordinates for bounding box prompts. Defaults to None.
            points (list, optional): Points for segmentation prompts. Defaults to None.
            labels (list, optional): Labels for segmentation prompts. Defaults to None.

        Returns:
            list: The model's predictions.
        """
        return self.predict(input_source, real_time, bounding_boxes, points, labels, **additional_args)

    def info(self, detailed=False, verbose=True):
        """
        Retrieve and log information about the SAM model.

        Args:
            detailed (bool, optional): If True, provides detailed model information. Defaults to False.
            verbose (bool, optional): If True, prints information to the console. Defaults to True.

        Returns:
            tuple: A tuple with information about the model.
        """
        return model_info(self.model, detailed=detailed, verbose=verbose)

    @property
    def task_map(self):
        """
        Provides a mapping for tasks to their corresponding Predictor.

        Returns:
            dict: A dictionary mapping the segmentation task to the Predictor class.
        """
        return {"segment": {"predictor": Predictor}}

"""
MODULE Segment Anything Model (SAM) Interface

DESCRIPTION
    This module provides an interface for interacting with the Segment Anything Model (SAM).
    SAM is used for prompt-based image segmentation and supports real-time processing.

IMPORTS
    - Path for handling file paths
    - Model class from the Ultralytics engine
    - model_info function for retrieving model information
    - build_sam function for building the SAM model
    - Predictor class for making predictions

CLASS SAM
    DESCRIPTION
        SAM interface class for promptable image segmentation.

    METHODS
        METHOD __init__(model_path)
            DESCRIPTION
                Initializes the SAM model with a pre-trained file.
            PARAMETERS
                - model_path: Path to the pre-trained model file (must be .pt or .pth).
            EXCEPTION
                - NotImplementedError: Raised if the file extension is not .pt or .pth.

        METHOD _load(weights_path, task)
            DESCRIPTION
                Loads weights into the SAM model from the specified file.
            PARAMETERS
                - weights_path: Path to the weights file.
                - task: Optional task name.
        
        METHOD predict(input_source, real_time, bounding_boxes, points, labels, **additional_args)
            DESCRIPTION
                Performs segmentation prediction on the provided input source.
            PARAMETERS
                - input_source: Path to the image/video file or an image object.
                - real_time: Boolean to enable real-time processing.
                - bounding_boxes: Optional list of bounding box coordinates for prompts.
                - points: Optional list of points for prompts.
                - labels: Optional list of labels for prompts.
                - **additional_args: Additional arguments for the prediction.
            RETURNS
                - List of model predictions.

        METHOD __call__(input_source, real_time, bounding_boxes, points, labels, **additional_args)
            DESCRIPTION
                Shortcut method to call the 'predict' method.
            PARAMETERS
                - Same as predict method.
            RETURNS
                - List of model predictions.

        METHOD info(detailed, verbose)
            DESCRIPTION
                Retrieves and logs information about the SAM model.
            PARAMETERS
                - detailed: Boolean for detailed model information.
                - verbose: Boolean for printing information to the console.
            RETURNS
                - Tuple containing model information.

        PROPERTY task_map
            DESCRIPTION
                Provides a mapping from task names to their corresponding Predictor classes.
            RETURNS
                - Dictionary mapping task names to Predictor classes.

"""
