import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import FocalLoss, VarifocalLoss
from ultralytics.utils.metrics import bbox_iou
from .ops import HungarianMatcher

class ObjectDetectionLoss(nn.Module):
    """
    A loss function class for object detection models, such as DETR. Computes various loss components including 
    classification loss, bounding box regression loss, and GIoU loss.

    Attributes:
        num_classes (int): Number of object classes.
        loss_weights (dict): Weights for different loss components.
        compute_auxiliary (bool): Flag to determine if auxiliary losses should be computed.
        focal_loss (FocalLoss or None): Focal loss object if enabled.
        varifocal_loss (VarifocalLoss or None): Varifocal loss object if enabled.
        use_fixed_layer (bool): Flag to use a fixed layer for auxiliary losses.
        fixed_layer_index (int): Index of the fixed layer for label assignment if `use_fixed_layer` is True.
        matcher (HungarianMatcher): Object for matching predictions to ground truth.
        device (torch.device): Device where tensors are located.
    """

    def __init__(
        self, num_classes=80, loss_weights=None, compute_auxiliary=True, use_focal_loss=True, use_varifocal_loss=False,
        use_fixed_layer=False, fixed_layer_index=0
    ):
        """
        Initializes the loss function with various parameters.

        Args:
            num_classes (int): Number of object classes.
            loss_weights (dict): Weights for different loss components (default: classification=1, bbox_regression=5, giou=2, no_object=0.1).
            compute_auxiliary (bool): If True, computes losses for each decoder layer.
            use_focal_loss (bool): If True, uses FocalLoss for classification.
            use_varifocal_loss (bool): If True, uses VarifocalLoss for classification.
            use_fixed_layer (bool): If True, uses a fixed layer for auxiliary losses.
            fixed_layer_index (int): Index of the fixed layer for label assignment.
        """
        super().__init__()
        if loss_weights is None:
            loss_weights = {"classification": 1, "bbox_regression": 5, "giou": 2, "no_object": 0.1}
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(cost_weights={"classification": 2, "bbox_regression": 5, "giou": 2})
        self.loss_weights = loss_weights
        self.compute_auxiliary = compute_auxiliary
        self.focal_loss = FocalLoss() if use_focal_loss else None
        self.varifocal_loss = VarifocalLoss() if use_varifocal_loss else None
        self.use_fixed_layer = use_fixed_layer
        self.fixed_layer_index = fixed_layer_index
        self.device = None

    def compute_classification_loss(self, predictions, targets, ground_truth_scores, num_ground_truths, suffix=""):
        """
        Computes the classification loss between predictions and ground truth.

        Args:
            predictions (Tensor): Predicted class scores.
            targets (Tensor): Ground truth class labels.
            ground_truth_scores (Tensor): Ground truth scores for each target.
            num_ground_truths (int): Number of ground truth objects.
            suffix (str): Suffix to append to the loss name.

        Returns:
            dict: Dictionary containing the classification loss.
        """
        class_loss_name = f"classification_loss{suffix}"
        batch_size, num_queries = predictions.shape[:2]
        # One-hot encode the ground truth labels
        one_hot_encoded = torch.zeros((batch_size, num_queries, self.num_classes + 1), dtype=torch.int64, device=targets.device)
        one_hot_encoded.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot_encoded = one_hot_encoded[..., :-1]  # Remove the last class (background)
        ground_truth_scores = ground_truth_scores.view(batch_size, num_queries, 1) * one_hot_encoded

        if self.focal_loss:
            if num_ground_truths and self.varifocal_loss:
                classification_loss = self.varifocal_loss(predictions, ground_truth_scores, one_hot_encoded)
            else:
                classification_loss = self.focal_loss(predictions, one_hot_encoded.float())
            classification_loss /= max(num_ground_truths, 1) / num_queries
        else:
            classification_loss = nn.BCEWithLogitsLoss(reduction="none")(predictions, ground_truth_scores).mean(1).sum()

        return {class_loss_name: classification_loss.squeeze() * self.loss_weights["classification"]}

    def compute_bbox_loss(self, predicted_boxes, ground_truth_boxes, suffix=""):
        """
        Computes the bounding box regression loss and GIoU loss.

        Args:
            predicted_boxes (Tensor): Predicted bounding boxes.
            ground_truth_boxes (Tensor): Ground truth bounding boxes.
            suffix (str): Suffix to append to the loss name.

        Returns:
            dict: Dictionary containing bounding box regression loss and GIoU loss.
        """
        bbox_loss_name = f"bbox_loss{suffix}"
        giou_loss_name = f"giou_loss{suffix}"

        losses = {}
        if len(ground_truth_boxes) == 0:
            # If no ground truth boxes, loss is zero
            losses[bbox_loss_name] = torch.tensor(0.0, device=self.device)
            losses[giou_loss_name] = torch.tensor(0.0, device=self.device)
            return losses

        # Compute L1 loss for bounding box regression
        losses[bbox_loss_name] = self.loss_weights["bbox_regression"] * F.l1_loss(predicted_boxes, ground_truth_boxes, reduction="sum") / len(ground_truth_boxes)
        # Compute GIoU loss
        giou_loss = 1.0 - bbox_iou(predicted_boxes, ground_truth_boxes, xywh=True, GIoU=True)
        giou_loss = giou_loss.sum() / len(ground_truth_boxes)
        losses[giou_loss_name] = self.loss_weights["giou"] * giou_loss
        return {k: v.squeeze() for k, v in losses.items()}

    def compute_auxiliary_losses(
        self,
        predicted_boxes,
        predicted_scores,
        ground_truth_boxes,
        ground_truth_classes,
        ground_truth_groups,
        match_indices=None,
        suffix="",
        masks=None,
        ground_truth_masks=None,
    ):
        """
        Computes auxiliary losses for each decoder layer.

        Args:
            predicted_boxes (list of Tensor): List of predicted bounding boxes from each layer.
            predicted_scores (list of Tensor): List of predicted scores from each layer.
            ground_truth_boxes (Tensor): Ground truth bounding boxes.
            ground_truth_classes (Tensor): Ground truth classes.
            ground_truth_groups (Tensor): Ground truth groups.
            match_indices (list of tuples): List of match indices for predictions and ground truth.
            suffix (str): Suffix to append to the loss names.
            masks (list of Tensor): List of predicted masks from each layer.
            ground_truth_masks (Tensor): Ground truth masks.

        Returns:
            dict: Dictionary containing auxiliary classification loss, bbox regression loss, and GIoU loss.
        """
        loss_values = torch.zeros(5 if masks is not None else 3, device=predicted_boxes.device)
        if match_indices is None and self.use_fixed_layer:
            # Use a fixed layer to assign labels if no match indices are provided
            match_indices = self.matcher(
                predicted_boxes[self.fixed_layer_index],
                predicted_scores[self.fixed_layer_index],
                ground_truth_boxes,
                ground_truth_classes,
                ground_truth_groups,
                masks=masks[self.fixed_layer_index] if masks is not None else None,
                ground_truth_masks=ground_truth_masks,
            )
        for i, (aux_boxes, aux_scores) in enumerate(zip(predicted_boxes, predicted_scores)):
            aux_masks = masks[i] if masks is not None else None
            aux_loss = self.calculate_loss(
                aux_boxes,
                aux_scores,
                ground_truth_boxes,
                ground_truth_classes,
                ground_truth_groups,
                masks=aux_masks,
                ground_truth_masks=ground_truth_masks,
                suffix=suffix,
                match_indices=match_indices,
            )
            loss_values[0] += aux_loss[f"classification_loss{suffix}"]
            loss_values[1] += aux_loss[f"bbox_loss{suffix}"]
            loss_values[2] += aux_loss[f"giou_loss{suffix}"]
            # Uncomment if mask loss computation is needed:
            # if masks is not None and ground_truth_masks is not None:
            #     mask_loss = self.compute_mask_loss(aux_masks, ground_truth_masks, match_indices, suffix)
            #     loss_values[3] += mask_loss[f'mask_loss{suffix}']
            #     loss_values[4] += mask_loss[f'dice_loss{suffix}']

        loss_dict = {
            f"classification_loss_aux{suffix}": loss_values[0],
            f"bbox_loss_aux{suffix}": loss_values[1],
            f"giou_loss_aux{suffix}": loss_values[2],
        }
        # Uncomment if mask loss computation is needed:
        # if masks is not None and ground_truth_masks is not None:
        #     loss_dict[f'mask_loss_aux{suffix}'] = loss_values[3]
        #     loss_dict[f'dice_loss_aux{suffix}'] = loss_values[4]
        return loss_dict

    @staticmethod
    def extract_indices(match_indices):
        """
        Extracts batch, source, and destination indices from the match indices.

        Args:
            match_indices (list of tuples): List of match indices for predictions and ground truth.

        Returns:
            tuple: Batch indices, source indices, and destination indices.
        """
        batch_indices = torch.cat([index[0] for index in match_indices])
        src_indices = torch.cat([index[1] for index in match_indices])
        tgt_indices = torch.cat([index[2] for index in match_indices])
        return batch_indices, src_indices, tgt_indices

    def calculate_loss(
        self,
        predicted_boxes,
        predicted_scores,
        ground_truth_boxes,
        ground_truth_classes,
        ground_truth_groups,
        match_indices=None,
        masks=None,
        ground_truth_masks=None,
        suffix="",
    ):
        """
        Calculates the total loss for object detection.

        Args:
            predicted_boxes (Tensor): Predicted bounding boxes.
            predicted_scores (Tensor): Predicted class scores.
            ground_truth_boxes (Tensor): Ground truth bounding boxes.
            ground_truth_classes (Tensor): Ground truth classes.
            ground_truth_groups (Tensor): Ground truth groups.
            match_indices (list of tuples): List of match indices for predictions and ground truth.
            masks (Tensor or None): Predicted masks if available.
            ground_truth_masks (Tensor): Ground truth masks.
            suffix (str): Suffix to append to the loss names.

        Returns:
            dict: Dictionary containing the total loss values.
        """
        if match_indices is not None:
            batch_indices, src_indices, tgt_indices = self.extract_indices(match_indices)
            # Gather the relevant ground truth data
            ground_truth_boxes = torch.cat([ground_truth_boxes[i][idx] for i, idx in enumerate(tgt_indices)])
            ground_truth_classes = torch.cat([ground_truth_classes[i][idx] for i, idx in enumerate(tgt_indices)])
            ground_truth_boxes = ground_truth_boxes[src_indices]
            ground_truth_classes = ground_truth_classes[src_indices]

        # Compute the classification loss
        loss_dict = self.compute_classification_loss(predicted_scores, ground_truth_classes, None, len(ground_truth_boxes), suffix=suffix)
        # Compute the bounding box regression loss
        bbox_loss_dict = self.compute_bbox_loss(predicted_boxes, ground_truth_boxes, suffix=suffix)
        loss_dict.update(bbox_loss_dict)
        return loss_dict

    def forward(
        self,
        predicted_boxes,
        predicted_scores,
        ground_truth_boxes,
        ground_truth_classes,
        ground_truth_groups,
        match_indices=None,
        masks=None,
        ground_truth_masks=None,
    ):
        """
        Computes the total loss including auxiliary losses if enabled.

        Args:
            predicted_boxes (list of Tensor): List of predicted bounding boxes from each layer.
            predicted_scores (list of Tensor): List of predicted scores from each layer.
            ground_truth_boxes (Tensor): Ground truth bounding boxes.
            ground_truth_classes (Tensor): Ground truth classes.
            ground_truth_groups (Tensor): Ground truth groups.
            match_indices (list of tuples or None): List of match indices for predictions and ground truth.
            masks (list of Tensor or None): List of predicted masks from each layer.
            ground_truth_masks (Tensor or None): Ground truth masks.

        Returns:
            dict: Dictionary containing the total loss values including auxiliary losses.
        """
        # Compute the primary loss
        loss_dict = self.calculate_loss(
            predicted_boxes[0],
            predicted_scores[0],
            ground_truth_boxes,
            ground_truth_classes,
            ground_truth_groups,
            match_indices=match_indices,
            masks=masks[0] if masks is not None else None,
            ground_truth_masks=ground_truth_masks
        )

        if self.compute_auxiliary:
            # Compute auxiliary losses for additional decoder layers
            aux_loss_dict = self.compute_auxiliary_losses(
                predicted_boxes,
                predicted_scores,
                ground_truth_boxes,
                ground_truth_classes,
                ground_truth_groups,
                match_indices=match_indices,
                masks=masks,
                ground_truth_masks=ground_truth_masks
            )
            loss_dict.update(aux_loss_dict)
        
        return loss_dict

"""
Class ObjectDetectionLoss:
    Attributes:
        num_classes: Integer representing the number of object classes.
        loss_weights: Dictionary containing weights for different loss components (e.g., classification, bbox_regression).
        compute_auxiliary: Boolean flag to indicate if auxiliary losses should be computed.
        focal_loss: FocalLoss object if enabled, otherwise None.
        varifocal_loss: VarifocalLoss object if enabled, otherwise None.
        use_fixed_layer: Boolean flag to use a fixed layer for auxiliary losses.
        fixed_layer_index: Integer index of the fixed layer for label assignment.
        matcher: HungarianMatcher object for matching predictions to ground truth.
        device: Device where tensors are located.

    Method __init__:
        Initialize attributes based on provided arguments or default values.

    Method compute_classification_loss(predictions, targets, ground_truth_scores, num_ground_truths, suffix):
        - Compute one-hot encoded ground truth labels.
        - Calculate classification loss using focal loss or BCE loss.
        - Return classification loss scaled by loss weight.

    Method compute_bbox_loss(predicted_boxes, ground_truth_boxes, suffix):
        - Compute L1 loss for bounding box regression.
        - Compute GIoU loss for bounding boxes.
        - Return bounding box regression loss and GIoU loss scaled by loss weights.

    Method compute_auxiliary_losses(predicted_boxes, predicted_scores, ground_truth_boxes, ground_truth_classes, ground_truth_groups, match_indices, suffix, masks, ground_truth_masks):
        - If match indices are None and fixed layer is used, assign labels using fixed layer.
        - For each decoder layer:
            - Compute classification, bbox regression, and GIoU losses.
            - Optionally compute mask losses.
        - Return dictionary with auxiliary losses for each layer.

    Method extract_indices(match_indices):
        - Extract batch, source, and destination indices from match indices.
        - Return extracted indices.

    Method calculate_loss(predicted_boxes, predicted_scores, ground_truth_boxes, ground_truth_classes, ground_truth_groups, match_indices, masks, ground_truth_masks, suffix):
        - If match indices are provided:
            - Gather relevant ground truth data.
        - Compute classification loss.
        - Compute bounding box regression loss.
        - Compute GIoU loss.
        - Return dictionary with total loss values.

    Method forward(predicted_boxes, predicted_scores, ground_truth_boxes, ground_truth_classes, ground_truth_groups, match_indices, masks, ground_truth_masks):
        - Compute primary loss using the first decoder layer's predictions.
        - If auxiliary losses are enabled:
            - Compute auxiliary losses for additional decoder layers.
        - Return dictionary with total loss values, including auxiliary losses.

"""
