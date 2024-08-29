
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils.metrics import SegmentMetrics

class QuickSegValidator(SegmentationValidator):
    def __init__(self, dataloader=None, save_directory=None, progress_bar=None, arguments=None, callbacks=None):
        """
        Initialize the QuickSegValidator, setting up the task, metrics, and disabling plotting features.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): The data loader to be used for fetching validation data.
            save_directory (Path, optional): Directory to save the validation results and metrics.
            progress_bar (tqdm.tqdm, optional): Progress bar object to visualize the validation progress.
            arguments (SimpleNamespace, optional): Configuration object containing various validation settings.
            callbacks (dict, optional): A dictionary of callback functions for extended functionality.

        Notes:
            Plotting features like ConfusionMatrix are turned off by default to prevent errors during the validation process.
        """
        # Call the parent class constructor to initialize base functionality
        super().__init__(dataloader, save_directory, progress_bar, arguments, callbacks)

        # Set the task to 'segment' to ensure proper handling of segmentation tasks
        self.args.task = "segment"

        # Disable plots such as ConfusionMatrix to streamline the validation process and avoid errors
        self.args.plots = False

        # Initialize segmentation metrics for evaluating the performance of the model
        self.metrics = SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)


"""
IMPORT necessary modules

DECLARE CLASS QuickSegValidator INHERITS SegmentationValidator:
    METHOD __init__(self, dataloader=None, save_directory=None, progress_bar=None, arguments=None, callbacks=None):
    
        CALL parent class constructor to initialize the base functionality

        SET task to 'segment' for proper handling of segmentation validation

        DISABLE plotting features such as ConfusionMatrix to avoid errors

        INITIALIZE segmentation metrics using SegmentMetrics with the save directory and plot settings


"""
