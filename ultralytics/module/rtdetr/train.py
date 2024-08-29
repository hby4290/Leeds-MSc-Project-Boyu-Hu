import torch
from copy import deepcopy

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, colorstr
from .val import RTDETRDataset, RTDETRValidator


class RTDETRTrainer(DetectionTrainer):
    """
    This class is designed to train the RT-DETR model, developed by Baidu for real-time object detection tasks.
    It extends the existing YOLO DetectionTrainer, adapting it to work with RT-DETR's specific features, 
    such as Vision Transformer technology, IoU-aware query selection, and configurable inference speeds.

    Important Notes:
        - The `deterministic=True` argument is not supported by the `F.grid_sample` function used in RT-DETR.
        - Automatic Mixed Precision (AMP) training may result in NaN values and cause issues during bipartite graph matching.
    
    Usage Example:
        ```python
        from ultralytics.models.rtdetr.train import RTDETRTrainer

        config = {'model': 'rtdetr-l.yaml', 'data': 'coco8.yaml', 'imgsz': 640, 'epochs': 3}
        trainer = RTDETRTrainer(overrides=config)
        trainer.train()
        ```
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Initializes and returns the RT-DETR model, set up for object detection.

        Args:
            cfg (dict, optional): Model configuration dictionary. Defaults to None.
            weights (str, optional): File path to the pre-trained model weights. Defaults to None.
            verbose (bool): If True, provides detailed logs. Defaults to True.

        Returns:
            RTDETRDetectionModel: A configured RT-DETR model ready for training or inference.
        """
        # Initialize the RTDETRDetectionModel with the provided configuration and number of classes
        model = RTDETRDetectionModel(cfg, nc=self.data["nc"], verbose=(verbose and RANK == -1))
        
        # If weights are specified, load them into the model
        if weights:
            model.load(weights)
        
        return model

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Constructs and returns a dataset tailored for RT-DETR, suited for either training or validation.

        Args:
            img_path (str): The path to the directory containing the images.
            mode (str): The operational mode, either 'train' or 'val'. Defaults to "val".
            batch (int, optional): Batch size for the dataset. Defaults to None.

        Returns:
            RTDETRDataset: A dataset object corresponding to the selected mode.
        """
        # Create the dataset object with parameters tailored for RT-DETR
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=(mode == "train"),
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def get_validator(self):
        """
        Provides an appropriate validator specifically designed for the RT-DETR model.

        Returns:
            RTDETRValidator: A validation object configured for the RT-DETR model.
        """
        # Define the loss components that will be tracked during validation
        self.loss_names = ("giou_loss", "cls_loss", "l1_loss")
        
        # Return an RTDETRValidator instance, initialized with the test loader and save directory
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=deepcopy(self.args))

    def preprocess_batch(self, batch):
        """
        Prepares a batch of images for training or validation by scaling and converting them to the appropriate format.

        Args:
            batch (dict): A dictionary containing images, bounding boxes, and class labels.

        Returns:
            dict: The processed batch, ready for model input.
        """
        # Call the base class method to preprocess the batch initially
        batch = super().preprocess_batch(batch)
        
        # Determine the batch size
        bs = len(batch["img"])
        
        # Retrieve the batch indices
        batch_idx = batch["batch_idx"]
        
        # Initialize lists to hold ground truth bounding boxes and class labels
        gt_bbox, gt_class = [], []
        
        # Iterate through each image in the batch
        for i in range(bs):
            # Append the bounding boxes and class labels corresponding to the current image
            gt_bbox.append(batch["bboxes"][batch_idx == i].to(batch_idx.device))
            gt_class.append(batch["cls"][batch_idx == i].to(device=batch_idx.device, dtype=torch.long))
        
        # Return the processed batch
        return batch

"""
CLASS RTDETRTrainer EXTENDS DetectionTrainer:
    """
    Trainer class for the RT-DETR model. This class adapts the YOLO DetectionTrainer to work with RT-DETR's
    specific features, such as Vision Transformer technology and configurable inference speeds.
    """

    METHOD get_model(cfg=None, weights=None, verbose=True):
        """
        Initializes and returns the RT-DETR model for object detection.

        Args:
            cfg: Configuration settings for the model (optional).
            weights: Path to pre-trained model weights (optional).
            verbose: Boolean flag for detailed logging (default: True).

        Returns:
            RTDETRDetectionModel: An instance of the RT-DETR model.
        """
        CREATE model instance with RTDETRDetectionModel using cfg and number of classes (nc)
        
        IF weights are provided:
            LOAD the weights into the model
        
        RETURN the initialized model

    METHOD build_dataset(img_path, mode="val", batch=None):
        """
        Constructs and returns a dataset for training or validation.

        Args:
            img_path: The path to the directory containing images.
            mode: The operational mode, either 'train' or 'val' (default: "val").
            batch: The batch size for the dataset (optional).

        Returns:
            RTDETRDataset: A dataset object for the specified mode.
        """
        RETURN RTDETRDataset with specified image path, image size, and other parameters based on mode

    METHOD get_validator():
        """
        Returns a validator specific to the RT-DETR model.

        Returns:
            RTDETRValidator: A validation object configured for RT-DETR.
        """
        SET loss_names to include giou_loss, cls_loss, and l1_loss
        
        RETURN RTDETRValidator initialized with the test loader, save directory, and copied arguments

    METHOD preprocess_batch(batch):
        """
        Prepares a batch of images for processing by scaling and converting them.

        Args:
            batch: A dictionary containing images, bounding boxes, and class labels.

        Returns:
            dict: The processed batch, ready for input to the model.
        """
        CALL the parent class's preprocess_batch method on the input batch
        
        DETERMINE batch size (bs)
        
        EXTRACT batch indices (batch_idx)
        
        INITIALIZE empty lists for ground truth bounding boxes (gt_bbox) and class labels (gt_class)
        
        FOR each image in the batch:
            APPEND corresponding bounding boxes and class labels to gt_bbox and gt_class respectively
        
        RETURN the processed batch

"""
