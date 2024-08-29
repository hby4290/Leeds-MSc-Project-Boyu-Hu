import math
from itertools import product
from typing import Any, Generator, List, Tuple

import numpy as np
import torch


def is_box_near_crop_edge(
    boxes: torch.Tensor, crop_box: List[int], orig_box: List[int], tolerance: float = 20.0
) -> torch.Tensor:
    """Determine if bounding boxes are near the edge of the crop area."""
    crop_tensor = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_tensor = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)
    boxes = adjust_boxes_to_original(boxes, crop_box).float()
    near_crop_edge = torch.isclose(boxes, crop_tensor[None, :], atol=tolerance, rtol=0)
    near_image_edge = torch.isclose(boxes, orig_tensor[None, :], atol=tolerance, rtol=0)
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    return torch.any(near_crop_edge, dim=1)


def data_batch_generator(batch_size: int, *datasets) -> Generator[List[Any], None, None]:
    """Yield data in batches from the provided datasets."""
    assert datasets and all(len(dataset) == len(datasets[0]) for dataset in datasets), "All inputs must have the same length."
    total_batches = len(datasets[0]) // batch_size + int(len(datasets[0]) % batch_size != 0)
    for batch in range(total_batches):
        yield [dataset[batch * batch_size : (batch + 1) * batch_size] for dataset in datasets]


def compute_stability_score(masks: torch.Tensor, mask_threshold: float, threshold_margin: float) -> torch.Tensor:
    """
    Calculates the stability score for a set of masks based on IoU between high and low thresholded masks.
    """
    # Calculate intersections and unions for the mask stability score
    intersections = (masks > (mask_threshold + threshold_margin)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    unions = (masks > (mask_threshold - threshold_margin)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    return intersections / unions


def create_point_grid(points_per_side: int) -> np.ndarray:
    """Generate a grid of points evenly spaced within the range [0,1]x[0,1]."""
    offset = 1 / (2 * points_per_side)
    points = np.linspace(offset, 1 - offset, points_per_side)
    x_coords = np.tile(points[None, :], (points_per_side, 1))
    y_coords = np.tile(points[:, None], (1, points_per_side))
    return np.stack([x_coords, y_coords], axis=-1).reshape(-1, 2)


def create_all_layer_grids(points_per_side: int, num_layers: int, scale_per_layer: int) -> List[np.ndarray]:
    """Generate point grids for each layer of the crop."""
    return [create_point_grid(int(points_per_side / (scale_per_layer**layer))) for layer in range(num_layers + 1)]


def generate_crops(
    image_size: Tuple[int, ...], num_layers: int, overlap_ratio: float
) -> Tuple[List[List[int]], List[int]]:
    """
    Generate crop boxes of varying sizes for different layers.
    Each layer produces a number of crop boxes based on its scale.
    """
    crop_boxes, layer_indices = [], []
    image_height, image_width = image_size
    min_side = min(image_height, image_width)

    # Original image as the first crop box
    crop_boxes.append([0, 0, image_width, image_height])
    layer_indices.append(0)

    def calculate_crop_size(original_size, num_crops, overlap):
        """Calculate the size of each crop box."""
        return int(math.ceil((overlap * (num_crops - 1) + original_size) / num_crops))

    for layer in range(num_layers):
        num_crops_side = 2 ** (layer + 1)
        overlap = int(overlap_ratio * min_side * (2 / num_crops_side))

        crop_width = calculate_crop_size(image_width, num_crops_side, overlap)
        crop_height = calculate_crop_size(image_height, num_crops_side, overlap)

        crop_x_offsets = [int((crop_width - overlap) * i) for i in range(num_crops_side)]
        crop_y_offsets = [int((crop_height - overlap) * i) for i in range(num_crops_side)]

        # Generate crop boxes in XYWH format
        for x0, y0 in product(crop_x_offsets, crop_y_offsets):
            crop_box = [x0, y0, min(x0 + crop_width, image_width), min(y0 + crop_height, image_height)]
            crop_boxes.append(crop_box)
            layer_indices.append(layer + 1)

    return crop_boxes, layer_indices


def adjust_boxes_to_original(boxes: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    """Adjust bounding boxes to match their position in the original image."""
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    return boxes + offset


def adjust_points_to_original(points: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    """Adjust points to match their position in the original image."""
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0]], device=points.device)
    if len(points.shape) == 3:
        offset = offset.unsqueeze(1)
    return points + offset


def pad_masks_to_original(masks: torch.Tensor, crop_box: List[int], original_height: int, original_width: int) -> torch.Tensor:
    """Pad cropped masks to the original image size."""
    x0, y0, x1, y1 = crop_box
    if x0 == 0 and y0 == 0 and x1 == original_width and y1 == original_height:
        return masks
    pad_x, pad_y = original_width - (x1 - x0), original_height - (y1 - y0)
    pad_values = (x0, pad_x - x0, y0, pad_y - y0)
    return torch.nn.functional.pad(masks, pad_values, value=0)


def remove_small_areas(mask: np.ndarray, area_threshold: float, operation_mode: str) -> Tuple[np.ndarray, bool]:
    """Remove small disconnected regions or holes from a mask based on the specified mode."""
    import cv2  # type: ignore

    assert operation_mode in {"holes", "islands"}
    remove_holes = operation_mode == "holes"
    mask_with_opposite = (remove_holes ^ mask).astype(np.uint8)
    num_labels, labeled_regions, stats, _ = cv2.connectedComponentsWithStats(mask_with_opposite, 8)
    region_sizes = stats[:, -1][1:]  # Skip background label
    small_region_indices = [i + 1 for i, size in enumerate(region_sizes) if size < area_threshold]
    if not small_region_indices:
        return mask, False
    fill_labels = [0] + small_region_indices
    if not remove_holes:
        fill_labels = [i for i in range(num_labels) if i not in fill_labels] or [int(np.argmax(region_sizes)) + 1]
    mask = np.isin(labeled_regions, fill_labels)
    return mask, True


def convert_masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Converts masks to bounding boxes in XYXY format.
    Returns a tensor with bounding boxes for each mask.
    """
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Reshape masks to CxHxW if needed
    mask_shape = masks.shape
    height, width = mask_shape[-2:]
    masks = masks.flatten(0, -3) if len(mask_shape) > 2 else masks.unsqueeze(0)

    # Calculate bounding box edges
    height_max, _ = torch.max(masks, dim=-1)
    height_coords = height_max * torch.arange(height, device=height_max.device)[None, :]
    bottom_edges, _ = torch.max(height_coords, dim=-1)
    height_coords += height * (~height_max)
    top_edges, _ = torch.min(height_coords, dim=-1)

    width_max, _ = torch.max(masks, dim=-2)
    width_coords = width_max * torch.arange(width, device=width_max.device)[None, :]
    right_edges, _ = torch.max(width_coords, dim=-1)
    width_coords += width * (~width_max)
    left_edges, _ = torch.min(width_coords, dim=-1)

    # Correct boxes where the mask is empty
    empty_boxes = (right_edges < left_edges) | (bottom_edges < top_edges)
    bounding_boxes = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    bounding_boxes *= (~empty_boxes).unsqueeze(-1)

    return bounding_boxes.reshape(*mask_shape[:-2], 4) if len(mask_shape) > 2 else bounding_boxes[0]


"""
Function is_box_near_crop_edge(boxes, crop_box, orig_box, tolerance):
    Convert crop_box and orig_box to tensors with the same device as boxes
    Uncrop boxes using crop_box
    Check if boxes are close to crop_box using tolerance
    Check if boxes are close to orig_box using tolerance
    Compute the logical AND of the two conditions (near crop edge but not near image edge)
    Return True if any box meets the condition, otherwise False

Function data_batch_generator(batch_size, *datasets):
    Verify all datasets have the same length
    Calculate the number of batches needed
    For each batch index:
        Yield a list of slices from each dataset corresponding to the current batch

Function compute_stability_score(masks, mask_threshold, threshold_margin):
    Calculate the number of pixels above high threshold (intersections)
    Calculate the number of pixels above low threshold (unions)
    Compute and return the IoU (intersections / unions)

Function create_point_grid(points_per_side):
    Calculate offset for points
    Generate linearly spaced points
    Create a grid of points using the calculated offset
    Return the grid as a flattened array of 2D points

Function create_all_layer_grids(points_per_side, num_layers, scale_per_layer):
    For each layer from 0 to num_layers:
        Generate a point grid with reduced points_per_side based on scale_per_layer
    Return a list of point grids for all layers

Function generate_crops(image_size, num_layers, overlap_ratio):
    Initialize crop_boxes and layer_indices
    Calculate minimum dimension of the image
    Add the original image as the first crop box
    For each layer:
        Calculate crop dimensions and overlap based on the layer
        Create crop boxes for the current layer
    Return crop_boxes and layer_indices

Function adjust_boxes_to_original(boxes, crop_box):
    Compute offset from crop_box
    Add offset to boxes to adjust for cropping
    Return the adjusted boxes

Function adjust_points_to_original(points, crop_box):
    Compute offset from crop_box
    Add offset to points to adjust for cropping
    Return the adjusted points

Function pad_masks_to_original(masks, crop_box, original_height, original_width):
    If the crop_box covers the entire original image:
        Return masks as is
    Calculate padding needed to restore original size
    Pad masks with zeros to match original dimensions
    Return the padded masks

Function remove_small_areas(mask, area_threshold, operation_mode):
    Invert mask based on operation_mode
    Perform connected component analysis on the mask
    Identify and remove small regions or holes based on area_threshold
    Return the updated mask and a flag indicating if modifications were made

Function convert_masks_to_boxes(masks):
    If masks are empty:
        Return zero boxes
    Reshape masks if necessary
    Calculate bounding box edges (top, bottom, left, right)
    Handle empty masks by correcting bounding boxes
    Return bounding boxes reshaped to original format

"""
