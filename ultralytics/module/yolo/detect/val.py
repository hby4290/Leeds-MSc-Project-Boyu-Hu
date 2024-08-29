import os
from pathlib import Path
import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images

class DetectionValidator(BaseValidator):
    """
    A class to validate YOLO object detection models, inheriting from BaseValidator.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, callbacks=None):
        """
        Initialize the DetectionValidator with necessary parameters.

        Args:
            dataloader (DataLoader, optional): DataLoader instance for validation.
            save_dir (Path, optional): Directory to save validation results.
            pbar (ProgressBar, optional): Progress bar instance.
            args (dict, optional): Configuration arguments.
            callbacks (list, optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, pbar, args, callbacks)
        self._initialize_attributes()

    def _initialize_attributes(self):
        """Setup initial attributes for validation."""
        self.num_targets_per_class = None
        self.is_coco_format = False
        self.class_mapping = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iou_values = torch.linspace(0.5, 0.95, 10)
        self.num_iou_values = self.iou_values.numel()
        self.label_boxes = []

    def preprocess_batch(self, batch):
        """
        Prepare and preprocess the batch for validation.

        Args:
            batch (dict): Batch containing images and annotations.

        Returns:
            dict: Preprocessed batch.
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for key in ["batch_idx", "cls", "bboxes"]:
            batch[key] = batch[key].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            num_batches = len(batch["img"])
            boxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.label_boxes = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], boxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(num_batches)
            ] if self.args.save_hybrid else []
        
        return batch

    def setup_metrics(self, model):
        """
        Initialize metrics for evaluation.

        Args:
            model (nn.Module): YOLO model to evaluate.
        """
        validation_path = self.data.get(self.args.split, "")
        self.is_coco_format = isinstance(validation_path, str) and "coco" in validation_path and validation_path.endswith(f"{os.sep}val2017.txt")
        self.class_mapping = converter.coco80_to_coco91_class() if self.is_coco_format else list(range(1000))
        self.args.save_json |= self.is_coco_format and not self.training
        self.class_names = model.names
        self.num_classes = len(model.names)
        self.metrics.names = self.class_names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(num_classes=self.num_classes, confidence_threshold=self.args.conf)
        self.processed_samples = 0
        self.json_results = []
        self.statistics = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}

    def format_description(self):
        """
        Format the description for metrics display.

        Returns:
            str: Description string.
        """
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def apply_nms(self, predictions):
        """
        Apply Non-Maximum Suppression (NMS) to model predictions.

        Args:
            predictions (Tensor): Model predictions.

        Returns:
            Tensor: NMS-applied predictions.
        """
        return ops.non_max_suppression(
            predictions,
            self.args.conf,
            self.args.iou,
            labels=self.label_boxes,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )

    def prepare_batch_data(self, sample_index, batch):
        """
        Prepare batch data for evaluation.

        Args:
            sample_index (int): Index of the sample in the batch.
            batch (dict): Batch data containing images and annotations.

        Returns:
            dict: Prepared batch data.
        """
        idx = batch["batch_idx"] == sample_index
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        original_shape = batch["ori_shape"][sample_index]
        image_size = batch["img"].shape[2:]
        ratio_padding = batch["ratio_pad"][sample_index]

        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(image_size, device=self.device)[[1, 0, 1, 0]]
            ops.scale_boxes(image_size, bbox, original_shape, ratio_pad=ratio_padding)
        
        return {
            "cls": cls,
            "bbox": bbox,
            "original_shape": original_shape,
            "image_size": image_size,
            "ratio_padding": ratio_padding
        }

    def prepare_predictions(self, predictions, batch_data):
        """
        Prepare predictions for evaluation.

        Args:
            predictions (Tensor): Model predictions.
            batch_data (dict): Prepared batch data.

        Returns:
            Tensor: Prepared predictions.
        """
        predictions = predictions.clone()
        ops.scale_boxes(
            batch_data["image_size"], predictions[:, :4], batch_data["original_shape"], ratio_pad=batch_data["ratio_padding"]
        )
        return predictions

    def update_validation_metrics(self, predictions, batch):
        """
        Update metrics based on predictions and ground truth data.

        Args:
            predictions (Tensor): Model predictions.
            batch (dict): Batch data with ground truth annotations.
        """
        for sample_index, pred in enumerate(predictions):
            self.processed_samples += 1
            num_predictions = len(pred)
            stats = {
                "conf": torch.zeros(0, device=self.device),
                "pred_cls": torch.zeros(0, device=self.device),
                "tp": torch.zeros(num_predictions, self.num_iou_values, dtype=torch.bool, device=self.device),
            }
            batch_data = self.prepare_batch_data(sample_index, batch)
            cls, bbox = batch_data.pop("cls"), batch_data.pop("bbox")
            num_targets = len(cls)
            stats["target_cls"] = cls
            
            if num_predictions == 0:
                if num_targets:
                    for key in self.statistics.keys():
                        self.statistics[key].append(stats[key])
                    if self.args.plots and self.args.task != "obb":
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            if self.args.single_cls:
                pred[:, 5] = 0
            processed_preds = self.prepare_predictions(pred, batch_data)
            stats["conf"] = processed_preds[:, 4]
            stats["pred_cls"] = processed_preds[:, 5]

            if num_targets:
                stats["tp"] = self._evaluate_predictions(processed_preds, bbox, cls)
                if self.args.plots and self.args.task != "obb":
                    self.confusion_matrix.process_batch(processed_preds, bbox, cls)
            
            for key in self.statistics.keys():
                self.statistics[key].append(stats[key])

            if self.args.save_json:
                self.save_predictions_to_json(processed_preds, batch["im_file"][sample_index])
            if self.args.save_txt:
                output_file = self.save_dir / "labels" / f'{Path(batch["im_file"][sample_index]).stem}.txt'
                self.save_predictions_to_txt(processed_preds, self.args.save_conf, batch_data["original_shape"], output_file)

    def finalize_validation_metrics(self):
        """
        Finalize metrics by updating speed and confusion matrix.
        """
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def compute_statistics(self):
        """
        Compute and process metrics statistics.

        Returns:
            dict: Processed statistics.
        """
        stats = {key: torch.cat(value, 0).cpu().numpy() for key, value in self.statistics.items()}
        stats["results"] = self.metrics.result(stats["tp"], stats["conf"], stats["pred_cls"], stats["target_cls"])
        return stats

    def __call__(self):
        """
        Run the validation process and compute metrics.

        Returns:
            dict: Final validation statistics.
        """
        for batch in self.dataloader:
            batch = self.preprocess_batch(batch)
            predictions = self.model(batch["img"])
            predictions = self.apply_nms(predictions)
            self.update_validation_metrics(predictions, batch)
        self.finalize_validation_metrics()
        return self.compute_statistics()

"""
class DetectionValidator(BaseValidator):
    method __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, callbacks=None):
        Call parent class initialization method
        Initialize basic attributes

    method _initialize_attributes(self):
        Set initialization attributes, such as task type, metrics, etc.

    method preprocess_batch(self, batch):
        Preprocess batch data (e.g., normalize images, transfer to device)
        Return preprocessed batch data

    method setup_metrics(self, model):
        Initialize metrics, set class names, and confusion matrix

    method format_description(self):
        Format the metrics description string
        Return description string

    method apply_nms(self, predictions):
        Apply Non-Maximum Suppression (NMS) to predictions
        Return NMS-applied predictions

    method prepare_batch_data(self, sample_index, batch):
        Prepare batch data (e.g., adjust box sizes, process image dimensions)
        Return prepared data

    method prepare_predictions(self, predictions, batch_data):
        Prepare prediction results (e.g., adjust box sizes)
        Return prepared prediction results

    method update_validation_metrics(self, predictions, batch):
        Update validation metrics (e.g., calculate true positives, false positives)
        Save predictions to JSON or TXT files (based on configuration)

    method finalize_validation_metrics(self):
        Finalize metrics calculations, such as speed and confusion matrix

    method compute_statistics(self):
        Compute and process metric statistics
        Return processed statistics

    method __call__(self):
        Execute the validation process
        Iterate through batches in the dataloader
        Preprocess batch data
        Obtain model predictions
        Apply NMS
        Update validation metrics
        Finalize metrics calculations
        Return final statistics

"""
