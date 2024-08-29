import torch
from ultralytics.engine.results import Results
from ultralytics.models.fastsam.utils import calculate_bbox_iou
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class QuickSegmentationPredictor(DetectionPredictor):
    """
    QuickSegmentationPredictor is a specialized class for handling segmentation tasks with the QuickSAM model within
    the Ultralytics YOLO framework.

    This class inherits from DetectionPredictor and customizes the prediction pipeline to focus on single-class
    segmentation, incorporating specific post-processing steps such as mask prediction and non-max suppression.

    Attributes:
        cfg (dict): Configuration settings for the prediction process.
        overrides (dict, optional): Optional overrides to customize behavior.
        callbacks (dict, optional): Optional list of callback functions for additional processing.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, callbacks=None):
        """
        Initializes the QuickSegmentationPredictor, setting up the segmentation task.

        Args:
            cfg (dict): Configuration settings for the prediction process.
            overrides (dict, optional): Optional overrides for the prediction process.
            callbacks (dict, optional): Optional callback functions for the prediction process.
        """
        # Initialize the base class with configuration and overrides
        super().__init__(cfg, overrides, callbacks)
        # Set the task to 'segment' to ensure proper handling in the YOLO framework
        self.args.task = "segment"

    def postprocess(self, predictions, processed_img, original_imgs):
        """
        Post-process the raw model predictions by applying non-max suppression, adjusting bounding boxes,
        and generating masks.

        Args:
            predictions (list): The raw predictions from the model.
            processed_img (torch.Tensor): The image tensor that has been pre-processed for prediction.
            original_imgs (list | torch.Tensor): The original images before any processing.

        Returns:
            list: A list of Results objects containing bounding boxes, masks, and other relevant data.
        """
        # Apply non-max suppression to filter and refine the predictions
        filtered_preds = ops.non_max_suppression(
            predictions[0],
            self.args.conf,  # Confidence threshold
            self.args.iou,  # IoU threshold
            agnostic=self.args.agnostic_nms,  # Class-agnostic NMS
            max_det=self.args.max_det,  # Maximum number of detections
            nc=1,  # Set number of classes to 1 as SAM is single-class
            classes=self.args.classes,  # Classes to filter
        )

        # Prepare a full image box for IoU comparison
        img_shape = processed_img.shape
        full_box = torch.zeros(filtered_preds[0].shape[1], device=filtered_preds[0].device)
        full_box[2], full_box[3], full_box[4], full_box[6:] = img_shape[3], img_shape[2], 1.0, 1.0
        full_box = full_box.view(1, -1)

        # Calculate IoU between full box and predicted boxes
        high_iou_indices = calculate_bbox_iou(full_box[0][:4], filtered_preds[0][:, :4], iou_thres=0.9, image_shape=img_shape[2:])
        
        # If high IoU is found, update the predictions
        if high_iou_indices.numel() > 0:
            full_box[0][4] = filtered_preds[0][high_iou_indices][:, 4]
            full_box[0][6:] = filtered_preds[0][high_iou_indices][:, 6:]
            filtered_preds[0][high_iou_indices] = full_box

        # Convert original images from torch.Tensor to numpy if needed
        if not isinstance(original_imgs, list):
            original_imgs = ops.convert_torch2numpy_batch(original_imgs)

        # Initialize a list to store the results
        results = []
        # Extract the segmentation prototype (masks)
        proto = predictions[1][-1] if len(predictions[1]) == 3 else predictions[1]
        
        # Iterate through each filtered prediction and process accordingly
        for i, pred in enumerate(filtered_preds):
            orig_img = original_imgs[i]
            img_path = self.batch[0][i]
            if len(pred) == 0:  # No predictions, set masks to None
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img_shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img_shape[2:], upsample=True)
                pred[:, :4] = ops.scale_boxes(img_shape[2:], pred[:, :4], orig_img.shape)
            
            # Append the result for this image
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))

        return results


"""
IMPORT necessary libraries and modules

DECLARE CLASS QuickSegmentationPredictor INHERITS DetectionPredictor:
    """
    A specialized class for handling segmentation tasks with QuickSAM model within the Ultralytics YOLO framework.
    """

    METHOD __init__(self, cfg=DEFAULT_CFG, overrides=None, callbacks=None):
        """
        Initialize the QuickSegmentationPredictor with configuration settings.

        Args:
            cfg: Configuration settings for prediction.
            overrides: Optional overrides for prediction behavior.
            callbacks: Optional callback functions for additional processing.
        """
        CALL the parent class's __init__ method with cfg, overrides, and callbacks
        SET task to "segment" to ensure proper handling in the YOLO framework

    METHOD postprocess(self, predictions, processed_img, original_imgs):
        """
        Post-process the raw predictions by applying non-max suppression and generating masks.

        Args:
            predictions: Raw predictions from the model.
            processed_img: Pre-processed image tensor.
            original_imgs: Original images before processing.

        Returns:
            A list of Results objects containing processed boxes, masks, and other metadata.
        """
        APPLY non-max suppression to refine predictions with the following parameters:
            - Confidence threshold
            - IoU threshold
            - Class-agnostic NMS
            - Maximum number of detections
            - Single-class setting (SAM is single-class)

        PREPARE a full image box for IoU comparison

        CALCULATE IoU between the full image box and predicted boxes

        IF high IoU indices are found:
            UPDATE predictions with the full box data

        IF original_imgs is a torch.Tensor:
            CONVERT original_imgs to numpy format

        INITIALIZE an empty list to store results

        EXTRACT the segmentation prototype (masks) from predictions

        FOR each prediction in the filtered predictions:
            GET the corresponding original image
            GET the image path from batch
            IF no predictions:
                SET masks to None
            ELSE IF retina masks are enabled:
                SCALE bounding boxes to original image size
                PROCESS masks using the native method
            ELSE:
                PROCESS masks using the standard method with upsampling
                SCALE bounding boxes to original image size
            
            APPEND the processed result to the results list

        RETURN the list of results

"""
