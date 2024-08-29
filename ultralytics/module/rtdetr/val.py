import torch
from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops

__all__ = ("RTDETRValidator",)  # specifying public API of the module

class RTDETRDataset(YOLODataset):
    """
    Custom dataset class for the RT-DETR model, extending the base YOLODataset class. 
    This class is specifically designed for handling datasets in real-time object detection and tracking tasks.
    """

    def __init__(self, *args, data=None, **kwargs):
        """
        Initializes the RTDETRDataset by inheriting the properties and methods of the YOLODataset class.
        """
        # Call the parent constructor to initialize the base dataset
        super().__init__(*args, data=data, **kwargs)

    def load_image(self, i, rect_mode=False):
        """
        Loads an image from the dataset at the given index `i`.

        Args:
            i (int): Index of the image in the dataset.
            rect_mode (bool): Whether to load the image in rectangular mode.

        Returns:
            tuple: Loaded image and its resized height and width.
        """
        # Utilize the parent class's method to load the image
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        """
        Constructs and returns the image transformation pipeline based on the dataset's mode (training or evaluation).

        Args:
            hyp (dict): Hyperparameters dictionary.

        Returns:
            Compose: A composition of image transformation functions.
        """
        # Handle transformations based on whether augmentation is enabled
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            # For evaluation mode, use an empty transformation pipeline
            transforms = Compose([])

        # Add formatting transformations for bounding boxes and other annotations
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms


class RTDETRValidator(DetectionValidator):
    """
    Custom validator class for the RT-DETR model, extending the DetectionValidator class.
    This class is tailored for validating RT-DETR model outputs, including the application of 
    Non-Maximum Suppression (NMS) and evaluation metrics updates.
    """

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Constructs and returns an RTDETRDataset for the specified mode (training or validation).

        Args:
            img_path (str): Path to the directory containing images.
            mode (str): Mode of operation, either 'train' or 'val'.
            batch (int, optional): Batch size for the dataset. Defaults to None.

        Returns:
            RTDETRDataset: A dataset object configured for the given mode.
        """
        # Create and return an RTDETRDataset object
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # Disable augmentation for validation
            hyp=self.args,
            rect=False,  # Disable rectangular training
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def postprocess(self, preds):
        """
        Post-processes the prediction outputs using Non-Maximum Suppression (NMS) to filter out redundant bounding boxes.

        Args:
            preds (torch.Tensor): The raw model predictions.

        Returns:
            list[torch.Tensor]: A list of filtered predictions after applying NMS.
        """
        # Extract bounding boxes and confidence scores from predictions
        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        bboxes *= self.args.imgsz  # Scale bounding boxes to image size

        # Initialize output list to hold processed predictions for each image in the batch
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs

        # Iterate over each image's bounding boxes
        for i, bbox in enumerate(bboxes):
            bbox = ops.xywh2xyxy(bbox)  # Convert bbox format from xywh to xyxy
            score, cls = scores[i].max(-1)  # Get the maximum confidence score and corresponding class
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)  # Concatenate bbox, score, and class

            # Sort predictions by confidence score for proper evaluation
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred  # Store the filtered predictions

        return outputs

    def _prepare_batch(self, si, batch):
        """
        Prepares and scales a batch of images for training or inference.

        Args:
            si (int): Index of the image in the batch.
            batch (dict): Dictionary containing images, bounding boxes, and class labels.

        Returns:
            dict: Dictionary containing the prepared bounding boxes and class labels.
        """
        idx = batch["batch_idx"] == si  # Find indices for the current image in the batch
        cls = batch["cls"][idx].squeeze(-1)  # Extract class labels
        bbox = batch["bboxes"][idx]  # Extract bounding boxes
        ori_shape = batch["ori_shape"][si]  # Original shape of the image
        imgsz = batch["img"].shape[2:]  # Target image size
        ratio_pad = batch["ratio_pad"][si]  # Padding ratio for scaling
        
        if len(cls):
            bbox = ops.xywh2xyxy(bbox)  # Convert bbox format to xyxy
            bbox[..., [0, 2]] *= ori_shape[1]  # Scale bounding boxes to original width
            bbox[..., [1, 3]] *= ori_shape[0]  # Scale bounding boxes to original height
        
        # Return the prepared data
        return dict(cls=cls, bbox=bbox, ori_shape=ori_shape, imgsz=imgsz, ratio_pad=ratio_pad)

    def _prepare_pred(self, pred, pbatch):
        """
        Adjusts and scales the predicted bounding boxes to match the original image dimensions.

        Args:
            pred (torch.Tensor): The raw model predictions.
            pbatch (dict): Dictionary containing the original shapes and scaling information.

        Returns:
            torch.Tensor: The adjusted predictions.
        """
        predn = pred.clone()  # Clone the predictions to avoid modifying the original tensor
        predn[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz  # Adjust x-coordinates
        predn[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz  # Adjust y-coordinates
        
        return predn.float()  # Return the adjusted predictions as floating-point numbers

"""
IMPORT necessary libraries and modules

DECLARE RTDETRDataset CLASS EXTENDING YOLODataset:
    """
    Dataset class for handling RT-DETR object detection tasks.
    """

    METHOD __init__(self, *args, data=None, **kwargs):
        """
        Initialize the RTDETRDataset by calling the parent class constructor.
        """
        CALL parent constructor with arguments
    
    METHOD load_image(self, i, rect_mode=False):
        """
        Load an image from the dataset at index 'i'.
        
        Args:
            i: Index of the image.
            rect_mode: Boolean to determine if image should be loaded in rectangular mode.

        Returns:
            The loaded image and its resized dimensions.
        """
        CALL parent class's load_image method with the index and rect_mode
        RETURN the result

    METHOD build_transforms(self, hyp=None):
        """
        Build and return the transformation pipeline for image preprocessing.
        
        Args:
            hyp: Hyperparameters for the transformation.

        Returns:
            A composed transformation function.
        """
        IF augmentation is enabled:
            SET mosaic and mixup parameters based on augmentation and rect mode
            CREATE transformation pipeline with v8_transforms and stretch enabled
        ELSE:
            CREATE an empty transformation pipeline
            
        ADD formatting transformations for bounding boxes and other annotations
        RETURN the composed transformation pipeline


DECLARE RTDETRValidator CLASS EXTENDING DetectionValidator:
    """
    Validator class tailored for validating RT-DETR model outputs.
    """

    METHOD build_dataset(self, img_path, mode="val", batch=None):
        """
        Build and return an RTDETRDataset for the specified mode.
        
        Args:
            img_path: Path to the images directory.
            mode: Mode of operation ('train' or 'val').
            batch: Batch size for the dataset (optional).

        Returns:
            An RTDETRDataset instance.
        """
        CREATE RTDETRDataset instance with specified parameters
        RETURN the dataset

    METHOD postprocess(self, preds):
        """
        Apply Non-Maximum Suppression (NMS) to filter prediction outputs.
        
        Args:
            preds: Raw model predictions.

        Returns:
            A list of filtered predictions after applying NMS.
        """
        EXTRACT bounding boxes and confidence scores from predictions
        SCALE bounding boxes to match the image size
        
        INITIALIZE an empty list to store filtered predictions for each image

        FOR each image's bounding boxes:
            CONVERT bounding box format from xywh to xyxy
            FIND the maximum confidence score and corresponding class
            CONCATENATE bounding boxes, scores, and class labels
            SORT predictions by confidence score
            STORE the sorted predictions in the output list

        RETURN the list of filtered predictions

    METHOD _prepare_batch(self, si, batch):
        """
        Prepare and scale a batch of images for training or inference.
        
        Args:
            si: Index of the image in the batch.
            batch: Dictionary containing images, bounding boxes, and class labels.

        Returns:
            A dictionary containing prepared bounding boxes and class labels.
        """
        FIND the indices for the current image in the batch
        EXTRACT class labels and bounding boxes for the current image
        GET the original shape of the image and target image size
        APPLY necessary scaling to the bounding boxes based on the original shape
        
        RETURN the prepared data as a dictionary

    METHOD _prepare_pred(self, pred, pbatch):
        """
        Adjust and scale predictions to match the original image dimensions.
        
        Args:
            pred: Raw predictions.
            pbatch: Dictionary containing original shapes and scaling information.

        Returns:
            The adjusted predictions.
        """
        CLONE the predictions to avoid modifying the original data
        SCALE the predictions to match the original image dimensions
        
        RETURN the scaled predictions as floating-point numbers

"""
