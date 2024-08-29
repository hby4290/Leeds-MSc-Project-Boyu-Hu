from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import verify_requirements
from ultralytics.utils.metrics import SegmentMetrics, calculate_box_iou, calculate_mask_iou
from ultralytics.utils.plotting import convert_output_to_target, visualize_images


class SegmentationEvaluator(DetectionValidator):
    """
    SegmentationEvaluator class extends DetectionValidator for segmentation model evaluation.

    Usage:
        ```python
        from ultralytics.models.yolo.segment import SegmentationEvaluator

        args = dict(model='yolov8n-seg.pt', data='coco8-seg.yaml')
        evaluator = SegmentationEvaluator(args=args)
        evaluator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SegmentationEvaluator with specific settings for segmentation tasks."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.mask_plot = None
        self.mask_process = None
        self.args.task = "segment"
        self.metrics = SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)

    def preprocess_batch(self, batch):
        """Preprocess the input batch by converting masks to floating point format and moving to device."""
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()
        return batch

    def initialize_metrics(self, model):
        """Set up metrics and choose the appropriate mask processing function."""
        super().init_metrics(model)
        self.mask_plot = []
        if self.args.save_json:
            verify_requirements("pycocotools>=2.0.6")
            self.mask_process = ops.process_mask_upsample  # Use higher accuracy method
        else:
            self.mask_process = ops.process_mask  # Use faster method
        self.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[])

    def get_metrics_description(self):
        """Return a formatted string that describes the evaluation metrics."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def process_predictions(self, predictions):
        """Apply Non-Maximum Suppression (NMS) to the predictions and extract the proto tensor."""
        detections = ops.non_max_suppression(
            predictions[0],
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
            nc=self.nc,
        )
        proto = predictions[1][-1] if len(predictions[1]) == 3 else predictions[1]
        return detections, proto

    def _batch_preparation(self, si, batch):
        """Prepare the batch by extracting and processing images and targets."""
        prepared_batch = super()._prepare_batch(si, batch)
        mask_idx = [si] if self.args.overlap_mask else batch["batch_idx"] == si
        prepared_batch["masks"] = batch["masks"][mask_idx]
        return prepared_batch

    def _prediction_preparation(self, pred, prepared_batch, proto):
        """Prepare predictions by processing the images and targets."""
        pred_normalized = super()._prepare_pred(pred, prepared_batch)
        pred_masks = self.mask_process(proto, pred[:, 6:], pred[:, :4], shape=prepared_batch["imgsz"])
        return pred_normalized, pred_masks

    def update_evaluation_metrics(self, preds, batch):
        """Update the evaluation metrics with the current batch of predictions."""
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
            self.seen += 1
            num_preds = len(pred)
            stats = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(num_preds, self.niou, dtype=torch.bool, device=self.device),
                tp_m=torch.zeros(num_preds, self.niou, dtype=torch.bool, device=self.device),
            )
            prepared_batch = self._batch_preparation(si, batch)
            classes, bboxes = prepared_batch.pop("cls"), prepared_batch.pop("bbox")
            num_labels = len(classes)
            stats["target_cls"] = classes
            if num_preds == 0:
                if num_labels:
                    for key in self.stats.keys():
                        self.stats[key].append(stats[key])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bboxes, gt_cls=classes)
                continue

            # Handle Masks
            gt_masks = prepared_batch.pop("masks")
            if self.args.single_cls:
                pred[:, 5] = 0
            pred_normalized, pred_masks = self._prediction_preparation(pred, prepared_batch, proto)
            stats["conf"] = pred_normalized[:, 4]
            stats["pred_cls"] = pred_normalized[:, 5]

            # Evaluate Predictions
            if num_labels:
                stats["tp"] = self._process_batch(pred_normalized, bboxes, classes)
                stats["tp_m"] = self._process_batch(
                    pred_normalized, bboxes, classes, pred_masks, gt_masks, self.args.overlap_mask, masks=True
                )
                if self.args.plots:
                    self.confusion_matrix.process_batch(pred_normalized, bboxes, classes)

            for key in self.stats.keys():
                self.stats[key].append(stats[key])

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if self.args.plots and self.batch_i < 3:
                self.mask_plot.append(pred_masks[:15].cpu())  # Only plot top 15 masks

            # Save Predictions
            if self.args.save_json:
                pred_masks = ops.scale_image(
                    pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                    prepared_batch["ori_shape"],
                    ratio_pad=batch["ratio_pad"][si],
                )
                self.save_predictions_as_json(pred_normalized, batch["im_file"][si], pred_masks)

    def finalize_evaluation_metrics(self, *args, **kwargs):
        """Finalize the evaluation metrics by setting the speed and confusion matrix."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_masks=None, gt_masks=None, overlap=False, masks=False):
        """
        Compute and return the correct prediction matrix based on IoU.

        Args:
            detections (array[N, 6]): Detected boxes, confidence, and class.
            labels (array[M, 5]): Ground truth boxes and classes.

        Returns:
            correct (array[N, 10]): Boolean array for correct predictions at different IoU levels.
        """
        if masks:
            if overlap:
                num_labels = len(gt_cls)
                index = torch.arange(num_labels, device=gt_masks.device).view(num_labels, 1, 1) + 1
                gt_masks = gt_masks.repeat(num_labels, 1, 1)
                gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
            if gt_masks.shape[1:] != pred_masks.shape[1:]:
                gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
                gt_masks = gt_masks.gt_(0.5)
            iou = calculate_mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
        else:
            iou = calculate_box_iou(gt_bboxes, detections[:, :4])

        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def visualize_validation_samples(self, batch, ni):
        """Visualize and save a batch of validation samples."""
        visualize_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def visualize_predictions(self, batch, preds, ni):
        """Visualize and save a batch of predictions."""
        visualize_images(
            batch["img"],
            *convert_output_to_target(preds[0], max_det=15),  # Limit to 15 detections for visualization
            torch.cat(self.mask_plot, dim=0) if len(self.mask_plot) else self.mask_plot,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
        self.mask_plot.clear()

    def save_predictions_as_json(self, pred_normalized, filename, pred_masks):
        """Save predictions in COCO JSON format."""
        image_id = int(Path(filename).stem.split("_")[-1]) if self.args.save_json > 1 else int(Path(filename).stem)
        for box, mask in zip(pred_normalized.tolist(), ops.encode_binary_mask(pred_masks)):
            self.json["annotations"].append(
                {
                    "image_id": image_id,
                    "category_id": self.coco91class[int(box[5])] if self.args.data.endswith("coco.yaml") else int(box[5]),
                    "bbox": ops.xyxy_to_xywh(box[:4]),
                    "score": round(box[4], 5),
                    "segmentation": mask,
                }
            )
"""
# Initialize the SegmentationEvaluator class, inheriting from DetectionValidator
Class SegmentationEvaluator extends DetectionValidator:
    
    # Constructor to initialize the evaluator with specific settings
    Function __init__(dataloader, save_dir, pbar, args, _callbacks):
        Call the parent class constructor
        Initialize task as "segment"
        Initialize metrics for segmentation tasks

    # Preprocess the input batch data
    Function preprocess_batch(batch):
        Preprocess batch data using parent method
        Convert masks to float and move to device
        Return preprocessed batch

    # Initialize metrics and choose mask processing method
    Function initialize_metrics(model):
        Initialize metrics using parent method
        Set mask processing function based on JSON saving requirement
        Initialize stats dictionary

    # Provide a formatted string for evaluation metrics
    Function get_metrics_description():
        Return formatted string describing evaluation metrics

    # Process predictions to apply Non-Maximum Suppression (NMS)
    Function process_predictions(predictions):
        Apply NMS on predictions
        Extract the proto tensor from predictions
        Return detections and proto tensor

    # Prepare batch data for processing
    Function _batch_preparation(si, batch):
        Prepare the batch using parent method
        Extract and process masks based on configuration
        Return prepared batch

    # Prepare predictions for evaluation
    Function _prediction_preparation(pred, prepared_batch, proto):
        Normalize predictions using parent method
        Process predicted masks
        Return normalized predictions and processed masks

    # Update evaluation metrics with the current batch
    Function update_evaluation_metrics(preds, batch):
        For each sample in the batch:
            Increment seen counter
            Initialize stats dictionary for current predictions
            Prepare the batch for processing
            Extract classes and bounding boxes
            If no predictions:
                Update stats and confusion matrix if necessary
                Continue to next sample
            Handle masks and prepare predictions
            Update stats with true positives
            Append stats to the global statistics dictionary
            Save mask plots if needed
            Save predictions to JSON if required

    # Finalize the evaluation metrics
    Function finalize_evaluation_metrics():
        Set speed and confusion matrix in metrics

    # Process the batch to compute the correct prediction matrix
    Function _process_batch(detections, gt_bboxes, gt_cls, pred_masks, gt_masks, overlap, masks):
        If evaluating masks:
            Handle overlap and upsample masks if needed
            Compute IoU for masks
        Else:
            Compute IoU for bounding boxes
        Return the matched predictions

    # Visualize and save validation samples
    Function visualize_validation_samples(batch, ni):
        Visualize and save images with ground truth labels

    # Visualize and save predictions
    Function visualize_predictions(batch, preds, ni):
        Visualize and save images with predicted labels
        Clear the mask plot list

    # Save predictions in COCO JSON format
    Function save_predictions_as_json(pred_normalized, filename, pred_masks):
        Extract image ID from filename
        For each box and mask:
            Add prediction details to the JSON annotations list

"""
