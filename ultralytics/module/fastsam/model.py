from pathlib import Path
from ultralytics.engine.model import Model
from .predict import FastSAMPredictor
from .val import FastSAMValidator

class QuickSAM(Model):
    """
    QuickSAM model interface designed for fast and efficient image segmentation.

    Example Usage:
        ```python
        from ultralytics import QuickSAM

        # Initialize the QuickSAM model with pre-trained weights
        model = QuickSAM('best_weights.pt')

        # Use the model to predict segmentation results on an input image
        results = model.predict('ultralytics/assets/bus.jpg')
        ```
    """

    def __init__(self, model="QuickSAM-x.pt"):
        """
        Initialize the QuickSAM model with a pre-trained model file.
        
        Args:
            model (str): Path to the pre-trained model file. Defaults to 'QuickSAM-x.pt'.
        """
        # Update the model path if the old default is used
        if str(model) == "QuickSAM.pt":
            model = "QuickSAM-x.pt"
        
        # Ensure that the provided model file is a pre-trained model and not a configuration file
        assert Path(model).suffix not in (".yaml", ".yml"), "QuickSAM requires a pre-trained model, not a configuration file."
        
        # Initialize the parent class (Model) with the selected model and set the task to "segment"
        super().__init__(model=model, task="segment")

    @property
    def task_mapping(self):
        """
        Provides a mapping of segmentation tasks to their respective predictor and validator classes.
        
        Returns:
            dict: A dictionary mapping task names to corresponding classes.
        """
        return {"segment": {"predictor": FastSAMPredictor, "validator": FastSAMValidator}}

"""
IMPORT Path from pathlib
IMPORT Model from ultralytics.engine.model
IMPORT FastSAMPredictor and FastSAMValidator from local modules

DECLARE CLASS QuickSAM INHERITS Model:
    """
    Interface for the QuickSAM model, designed for image segmentation tasks.
    """

    METHOD __init__(self, model="QuickSAM-x.pt"):
        """
        Initialize the QuickSAM model with the specified pre-trained model file.

        Args:
            model: The path to the pre-trained model file (default is "QuickSAM-x.pt").
        """
        IF model is equal to "QuickSAM.pt":
            SET model to "QuickSAM-x.pt"
        
        ASSERT the model file is not a configuration file (with .yaml or .yml extension)
        
        CALL the parent class's __init__ method with model and task="segment"

    PROPERTY METHOD task_mapping:
        """
        Provide a mapping of segmentation tasks to the corresponding predictor and validator classes.

        Returns:
            A dictionary with task names as keys and corresponding classes as values.
        """
        RETURN a dictionary mapping "segment" to FastSAMPredictor and FastSAMValidator

"""
