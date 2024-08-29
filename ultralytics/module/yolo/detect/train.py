# Import necessary libraries and modules
import math
import random
from copy import deepcopy

import numpy as np
import torch.nn as nn

from ultralytics.data import create_dataloader, create_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import display_images, display_labels, display_results
from ultralytics.utils.torch_utils import remove_parallel, initialize_torch_distributed

class ObjectDetectionTrainer(BaseTrainer):
    """
    This class extends the BaseTrainer to handle training tasks for object detection models.
    
    Example usage:
        ```python
        from ultralytics.models.yolo.detect import ObjectDetectionTrainer
        
        parameters = {'model': 'yolov8n.pt', 'data': 'coco8.yaml', 'epochs': 3}
        trainer = ObjectDetectionTrainer(overrides=parameters)
        trainer.train()
        ```
    """

    def prepare_dataset(self, image_directory, mode="train", batch_size=None):
        """
        Creates a YOLO dataset for training or validation.
        
        Args:
            image_directory (str): Directory path containing the images.
            mode (str): Training or validation mode, allows for different augmentations.
            batch_size (int, optional): Size of the batches, particularly for rectangular modes.
        """
        grid_size = max(int(remove_parallel(self.model).stride.max() if self.model else 0), 32)
        return create_yolo_dataset(
            self.args, image_directory, batch_size, self.data, mode=mode, rect=False, stride=grid_size
        )

    def create_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Build and return a dataloader."""
        assert mode in ["train", "val"], "Mode must be 'train' or 'val'"
        
        with initialize_torch_distributed(rank):
            dataset = self.prepare_dataset(dataset_path, mode, batch_size)
        
        shuffle_data = mode == "train"
        if getattr(dataset, "rect", False) and shuffle_data:
            LOGGER.warning("WARNING  'rect=True' conflicts with DataLoader shuffle. Disabling shuffle.")
            shuffle_data = False
        
        num_workers = self.args.workers if mode == "train" else self.args.workers * 2
        return create_dataloader(dataset, batch_size, num_workers, shuffle_data, rank)

    def process_batch(self, batch):
        """Process a batch by normalizing and resizing images."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            images = batch["img"]
            new_size = (
                random.randrange(self.args.imgsz * 0.5, self.args.imgsz * 1.5 + self.stride) // self.stride
                * self.stride
            )  # Calculate new size
            scale_factor = new_size / max(images.shape[2:])  # Compute scale factor
            if scale_factor != 1:
                new_shape = [math.ceil(x * scale_factor / self.stride) * self.stride for x in images.shape[2:]]
                images = nn.functional.interpolate(images, size=new_shape, mode="bilinear", align_corners=False)
            batch["img"] = images
        return batch

    def configure_model(self):
        """Set model attributes including number of classes and class names."""
        self.model.nc = self.data["nc"]  # Number of classes
        self.model.names = self.data["names"]  # Class names
        self.model.args = self.args  # Hyperparameters

    def load_model(self, config=None, weights_path=None, verbose=True):
        """Load and return a YOLO detection model."""
        model = DetectionModel(config, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights_path:
            model.load(weights_path)
        return model

    def create_validator(self):
        """Create and return a validator for the YOLO detection model."""
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=deepcopy(self.args), _callbacks=self.callbacks
        )

    def format_loss_labels(self, loss_values=None, prefix="train"):
        """
        Formats loss values into a dictionary with labeled training loss items.
        
        Args:
            loss_values (list, optional): List of loss values.
            prefix (str): Prefix for loss keys.
        
        Returns:
            dict: Dictionary of formatted loss items.
        """
        keys = [f"{prefix}/{name}" for name in self.loss_names]
        if loss_values:
            loss_values = [round(float(value), 5) for value in loss_values]
            return dict(zip(keys, loss_values))
        return keys

    def progress_update(self):
        """Returns a formatted string showing training progress."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def visualize_training_samples(self, batch, index):
        """Visualize training samples with their annotations."""
        display_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            class_labels=batch["cls"].squeeze(-1),
            bounding_boxes=batch["bboxes"],
            image_paths=batch["im_file"],
            filename=self.save_dir / f"train_batch{index}.jpg",
            on_plot=self.on_plot,
        )

    def visualize_metrics(self):
        """Plot metrics from a CSV file."""
        display_results(file=self.csv, on_plot=self.on_plot)

    def visualize_training_labels(self):
        """Generate and save labeled training plots for the YOLO model."""
        boxes = np.concatenate([label["bboxes"] for label in self.train_loader.dataset.labels], axis=0)
        class_labels = np.concatenate([label["cls"] for label in self.train_loader.dataset.labels], axis=0)
        display_labels(boxes, class_labels.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)


"""
CLASS ObjectDetectionTrainer EXTENDS BaseTrainer

    FUNCTION prepare_dataset(image_directory, mode="train", batch_size=None)
        SET grid_size TO maximum of (model stride or 32)
        RETURN build_yolo_dataset with arguments (image_directory, batch_size, data, mode, rect=False, stride=grid_size)

    FUNCTION create_dataloader(dataset_path, batch_size=16, rank=0, mode="train")
        ASSERT mode IS "train" OR "val"
        CALL initialize_torch_distributed with rank
            SET dataset TO prepare_dataset with arguments (dataset_path, mode, batch_size)
        SET shuffle_data TO (mode IS "train")
        IF dataset is rectangular AND shuffle_data IS TRUE
            LOG warning about 'rect=True' and shuffle
            SET shuffle_data TO FALSE
        SET num_workers TO (workers if mode IS "train" ELSE workers * 2)
        RETURN build_dataloader with arguments (dataset, batch_size, num_workers, shuffle_data, rank)

    FUNCTION process_batch(batch)
        SET batch["img"] TO (transfer to device, convert to float, normalize)
        IF multi_scale IS TRUE
            SET images TO batch["img"]
            CALCULATE new_size BASED ON random size within a range
            CALCULATE scale_factor BASED ON new_size and images dimensions
            IF scale_factor IS NOT 1
                CALCULATE new_shape BASED ON scale_factor
                INTERPOLATE images TO new_shape
            SET batch["img"] TO processed images
        RETURN batch

    FUNCTION configure_model()
        SET model.nc TO number of classes
        SET model.names TO class names
        SET model.args TO hyperparameters

    FUNCTION load_model(config=None, weights_path=None, verbose=True)
        CREATE model OF DetectionModel with config and number of classes
        IF weights_path IS NOT NULL
            LOAD weights INTO model
        RETURN model

    FUNCTION create_validator()
        SET loss_names TO ("box_loss", "cls_loss", "dfl_loss")
        RETURN DetectionValidator with arguments (test_loader, save_dir, args, callbacks)

    FUNCTION format_loss_labels(loss_values=None, prefix="train")
        CREATE keys BASED ON loss_names with prefix
        IF loss_values IS NOT NULL
            ROUND loss_values TO 5 decimal places
            RETURN dictionary of keys and rounded loss_values
        RETURN keys

    FUNCTION progress_update()
        RETURN formatted string showing epoch, GPU memory, losses, instances, and size

    FUNCTION visualize_training_samples(batch, index)
        CALL display_images with arguments (images, batch_idx, class_labels, bounding_boxes, image_paths, filename, on_plot)

    FUNCTION visualize_metrics()
        CALL display_results with arguments (file, on_plot)

    FUNCTION visualize_training_labels()
        CONCatenate bounding boxes and class labels from dataset
        CALL display_labels with arguments (boxes, class_labels, names, save_dir, on_plot)

END CLASS

"""
