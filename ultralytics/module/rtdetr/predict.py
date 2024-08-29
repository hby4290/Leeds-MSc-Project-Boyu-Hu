import torch
from ultralytics.data.augment import ResizeToSquare
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import DetectionResults
from ultralytics.utils import tensor_utils


class RTDETRPredictor(BasePredictor):
    """
    RT-DETR Predictor class for handling predictions with Baidu's RT-DETR model. This class extends the
    BasePredictor and implements methods for processing model outputs and preparing images for inference.

    This class leverages Vision Transformer (ViT) technology to deliver real-time object detection with high accuracy.
    Key features include hybrid encoding and IoU-aware query selection for enhanced performance.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.rtdetr import RTDETRPredictor

        config = {'model': 'rtdetr-large.pt', 'source': ASSETS}
        predictor = RTDETRPredictor(config)
        predictor.run_prediction()
        ```

    Attributes:
        image_size (int): The size to which images are resized for inference. Must be square (e.g., 640).
        configuration (dict): Configuration settings for the predictor.
    """

    def post_process(self, raw_predictions, processed_images, original_images):
        """
        Convert raw model outputs into bounding boxes and confidence scores, and filter results based on confidence thresholds
        and class labels if specified.

        Args:
            raw_predictions (torch.Tensor): Model outputs containing bounding boxes and scores.
            processed_images (torch.Tensor): Images after pre-processing steps.
            original_images (list or torch.Tensor): The original images before any processing.

        Returns:
            (list[DetectionResults]): A list of DetectionResults objects, each containing bounding boxes, confidence scores,
                and class labels.
        """
        num_classes = raw_predictions[0].shape[-1]
        boxes, confidences = raw_predictions[0].split((4, num_classes - 4), dim=-1)

        if not isinstance(original_images, list):  # Convert Tensor to list if needed
            original_images = tensor_utils.tensor_to_numpy_list(original_images)

        detection_results = []
        for i, box in enumerate(boxes):
            box = tensor_utils.convert_xywh_to_xyxy(box)
            confidence, class_labels = confidences[i].max(dim=-1, keepdim=True)
            valid_indices = confidence.squeeze(dim=-1) > self.configuration['confidence_threshold']
            if 'classes' in self.configuration:
                valid_indices &= (class_labels == torch.tensor(self.configuration['classes'], device=class_labels.device)).any(dim=1)
            filtered_predictions = torch.cat([box, confidence, class_labels], dim=-1)[valid_indices]
            original_img = original_images[i]
            height, width = original_img.shape[:2]
            filtered_predictions[..., [0, 2]] *= width
            filtered_predictions[..., [1, 3]] *= height
            img_path = self.batch[0][i]
            detection_results.append(DetectionResults(original_img, path=img_path, labels=self.model.labels, boxes=filtered_predictions))
        return detection_results

    def prepare_images(self, images):
        """
        Pre-process images by resizing them to a square aspect ratio and ensuring the scale is preserved.
        The output size should be square (e.g., 640) and maintain aspect ratio.

        Args:
            images (list[np.ndarray] | torch.Tensor): Input images with shape (N, 3, H, W) for tensors or [(H, W, 3) x N] for lists.

        Returns:
            (list): List of images prepared for model inference.
        """
        resize_to_square = ResizeToSquare(self.image_size, scale_fill=True)
        return [resize_to_square(image=x) for x in images]

"""
CLASS RTDETRPredictor EXTENDS BasePredictor:
    """
    Class for handling predictions with Baidu's RT-DETR model.
        Convert raw model outputs to bounding boxes, confidence scores, and filter results.
        """

        num_classes = GET number of classes from raw_predictions
        
        SPLIT raw_predictions into boxes and confidences

        IF original_images is not a list THEN
            CONVERT original_images from Tensor to list
        
        INITIALIZE detection_results as an empty list

        FOR each box in boxes:
            CONVERT box format from xywh to xyxy
            EXTRACT confidence and class_labels from confidences
            FIND valid_indices based on confidence threshold
            
            IF 'classes' in configuration:
                FILTER valid_indices to include only specified classes
            
            CONCATENATE filtered_predictions from box, confidence, and class_labels
            SCALE filtered_predictions to original image dimensions
            GET img_path from batch
            APPEND DetectionResults object to detection_results list

        RETURN detection_results

    METHOD prepare_images(images):
        """
        Pre-process images by resizing to square aspect ratio.
        """

        INITIALIZE ResizeToSquare with image_size and scale_fill set to True
        RETURN list of images resized to square using ResizeToSquare

"""
