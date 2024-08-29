import torch

def refine_bounding_boxes_to_edges(boxes, image_size, edge_threshold=20):
    """
    Refine bounding boxes to align with the image edges if they are close to the borders.

    Args:
        boxes (torch.Tensor): Tensor of bounding boxes with shape (n, 4), where n is the number of boxes.
        image_size (tuple): Dimensions of the image as (height, width).
        edge_threshold (int): Distance in pixels to consider a box edge close to the image border.

    Returns:
        torch.Tensor: The refined bounding boxes tensor.
    """

    # Unpack image dimensions
    height, width = image_size

    # Adjust box coordinates near the image borders
    boxes[:, 0] = torch.where(boxes[:, 0] < edge_threshold, torch.zeros_like(boxes[:, 0]), boxes[:, 0])  # x1
    boxes[:, 1] = torch.where(boxes[:, 1] < edge_threshold, torch.zeros_like(boxes[:, 1]), boxes[:, 1])  # y1
    boxes[:, 2] = torch.where(boxes[:, 2] > width - edge_threshold, torch.full_like(boxes[:, 2], width), boxes[:, 2])  # x2
    boxes[:, 3] = torch.where(boxes[:, 3] > height - edge_threshold, torch.full_like(boxes[:, 3], height), boxes[:, 3])  # y2

    return boxes

def calculate_iou(reference_box, comparison_boxes, iou_threshold=0.9, image_size=(640, 640), return_raw=False):
    """
    Calculate the Intersection Over Union (IoU) between a reference bounding box and multiple comparison boxes.

    Args:
        reference_box (torch.Tensor): Tensor of a single bounding box with shape (4,).
        comparison_boxes (torch.Tensor): Tensor of multiple bounding boxes with shape (n, 4).
        iou_threshold (float): The threshold above which IoU is considered significant.
        image_size (tuple): Size of the image as (height, width).
        return_raw (bool): If True, returns the raw IoU values instead of indices.

    Returns:
        torch.Tensor: Indices of boxes with IoU greater than the threshold, or raw IoU values if return_raw is True.
    """
    # Refine the bounding boxes to align with image borders
    comparison_boxes = refine_bounding_boxes_to_edges(comparison_boxes, image_size)

    # Calculate the intersection coordinates
    inter_x1 = torch.max(reference_box[0], comparison_boxes[:, 0])
    inter_y1 = torch.max(reference_box[1], comparison_boxes[:, 1])
    inter_x2 = torch.min(reference_box[2], comparison_boxes[:, 2])
    inter_y2 = torch.min(reference_box[3], comparison_boxes[:, 3])

    # Calculate intersection area, ensuring non-negative dimensions
    intersection_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Calculate areas of the reference and comparison boxes
    ref_box_area = (reference_box[2] - reference_box[0]) * (reference_box[3] - reference_box[1])
    comp_boxes_area = (comparison_boxes[:, 2] - comparison_boxes[:, 0]) * (comparison_boxes[:, 3] - comparison_boxes[:, 1])

    # Calculate union area
    union_area = ref_box_area + comp_boxes_area - intersection_area

    # Compute IoU
    iou_values = intersection_area / union_area

    if return_raw:
        # Return raw IoU values
        return iou_values if iou_values.numel() > 0 else torch.tensor(0.0)

    # Return indices of boxes that meet the IoU threshold
    return torch.nonzero(iou_values > iou_threshold).flatten()
