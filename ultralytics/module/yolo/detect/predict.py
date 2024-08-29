# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class ObjectDetectionPredictor(BasePredictor):
    """
    ObjectDetectionPredictor class inherits from BasePredictor to perform detection-based predictions.

    Example usage:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import ObjectDetectionPredictor

        settings = dict(model='yolov8n.pt', source=ASSETS)
        predictor = ObjectDetectionPredictor(overrides=settings)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, predictions, image, original_images):
        """
        Post-process the predictions by applying non-max suppression and scaling the bounding boxes.

        Args:
            predictions (torch.Tensor): The raw predictions from the model.
            image (torch.Tensor): The preprocessed input image.
            original_images (Union[torch.Tensor, List]): The original input images before preprocessing.

        Returns:
            List[Results]: A list of Results objects containing the final processed predictions.
        """

        # Apply Non-Maximum Suppression (NMS) to filter out redundant boxes
        filtered_preds = ops.non_max_suppression(
            predictions,
            self.args.conf,  # confidence threshold
            self.args.iou,  # IoU threshold
            agnostic=self.args.agnostic_nms,  # class-agnostic NMS
            max_det=self.args.max_det,  # maximum number of detections
            classes=self.args.classes,  # filter by class
        )

        # Convert original images to numpy if they are in torch.Tensor format
        if not isinstance(original_images, list):
            original_images = ops.convert_torch2numpy_batch(original_images)

        # Initialize an empty list to store the results
        processed_results = []

        # Iterate over each prediction and corresponding original image
        for idx, filtered_pred in enumerate(filtered_preds):
            orig_img = original_images[idx]  # get the original image

            # Scale the bounding boxes back to the original image dimensions
            filtered_pred[:, :4] = ops.scale_boxes(image.shape[2:], filtered_pred[:, :4], orig_img.shape)

            # Get the image path from the batch (assuming batch is stored in self.batch)
            image_path = self.batch[0][idx]

            # Create a Results object for the current image and append it to the results list
            result = Results(orig_img, path=image_path, names=self.model.names, boxes=filtered_pred)
            processed_results.append(result)

        # Return the list of processed results
        return processed_results

"""
Class ObjectDetectionPredictor extends BasePredictor:
    """
    ObjectDetectionPredictor is responsible for handling the prediction process
    of object detection models, including post-processing tasks like NMS and 
    box scaling.
    """

    Method postprocess(predictions, image, original_images):
        """
        Post-processes the raw predictions from the model.
        """

        # Step 1: Apply Non-Maximum Suppression (NMS) to filter redundant boxes
        filtered_preds = Apply NMS on predictions with given confidence, IoU thresholds, and other settings

        # Step 2: Convert original images to numpy format if they are in tensor format
        If original_images is not a list:
            Convert original_images to numpy format

        # Step 3: Initialize an empty list to store the processed results
        Initialize empty list processed_results

        # Step 4: Loop over each prediction and corresponding original image
        For each index, filtered_pred in filtered_preds:
            orig_img = original_images[index]  # Get the corresponding original image

            # Step 5: Scale bounding boxes back to the dimensions of the original image
            Scale filtered_pred boxes according to orig_img shape

            # Step 6: Retrieve the image path from the batch
            image_path = Get image path from the batch using the current index

            # Step 7: Create a Results object for the current prediction and append it to the results list
            Create Results object with orig_img, image_path, model names, and filtered_pred
            Append this Results object to processed_results list

        # Step 8: Return the list of processed results
        Return processed_results

"""
