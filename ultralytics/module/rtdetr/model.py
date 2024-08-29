from ultralytics.engine.model import BaseModel
from ultralytics.nn.tasks import VisionTransformerDetection

from .predictor import RTDETRPredict
from .trainer import RTDETRTrain
from .validator import RTDETREvaluate


class RTDETR(BaseModel):
    """
    RT-DETR Model Interface: A Vision Transformer-based object detector designed for high-speed and accurate
    real-time object detection. This class handles initialization and provides access to model functionalities.

    Attributes:
        model_path (str): File path to the pre-trained RT-DETR model. Default is 'rtdetr-large.pt'.
    """

    def __init__(self, model_path="rtdetr-large.pt") -> None:
        """
        Initializes the RT-DETR model with the specified pre-trained model file. Accepts model files in
        '.pt', '.yaml', or '.yml' formats.

        Args:
            model_path (str): The file path to the pre-trained model. Defaults to 'rtdetr-large.pt'.

        Raises:
            ValueError: If the provided model file extension is not supported (must be 'pt', 'yaml', or 'yml').
        """
        # Check if the model file extension is valid
        if model_path.split(".")[-1] not in ("pt", "yaml", "yml"):
            raise ValueError("Supported model file extensions are *.pt, *.yaml, and *.yml only.")
        
        # Initialize the base model with detection task
        super().__init__(model_path=model_path, task_type="object_detection")

    @property
    def task_mappings(self) -> dict:
        """
        Provides a dictionary mapping specific tasks to their corresponding classes within the Ultralytics framework.

        Returns:
            dict: A dictionary linking task names to their associated classes for RT-DETR.
        """
        return {
            "object_detection": {
                "predictor": RTDETRPredict,
                "trainer": RTDETRTrain,
                "validator": RTDETREvaluate,
                "model_class": VisionTransformerDetection,
            }
        }
"""
Define RT-DETR Model Interface:
    Description:
        A Vision Transformer-based model for real-time object detection.
        Optimized for high speed and accuracy, supports CUDA and TensorRT acceleration.
        Features include hybrid encoder and IoU-aware query selection.

    Initialize RTDETR Model:
        Input:
            model_path (string) - Path to the pre-trained model file.
            Default: 'rtdetr-large.pt'
        Check if model_path ends with valid extensions ('pt', 'yaml', 'yml'):
            If not valid, raise a ValueError with appropriate message.
        Call the base model initializer with:
            model_path
            task_type as "object_detection"

    Define task_mappings property:
        Return a dictionary mapping task names to classes:
            Key: "object_detection"
            Value:
                - predictor: RTDETRPredict
                - trainer: RTDETRTrain
                - validator: RTDETREvaluate
                - model_class: VisionTransformerDetection

"""
