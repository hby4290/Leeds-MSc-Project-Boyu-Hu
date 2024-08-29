import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh


class HungarianMatcher(nn.Module):
    """
    Implements the HungarianMatcher, a module designed to solve the assignment problem between predicted and ground truth
    bounding boxes using a differentiable approach.

    Attributes:
        cost_gain (dict): Coefficients for various cost components: 'class', 'bbox', 'giou', 'mask', and 'dice'.
        use_fl (bool): Flag indicating whether to use Focal Loss for classification cost.
        with_mask (bool): Flag indicating whether mask predictions are included.
        num_sample_points (int): Number of sample points used for mask cost calculation.
        alpha (float): Focal Loss alpha parameter.
        gamma (float): Focal Loss gamma parameter.

    Methods:
        forward(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None): Computes the optimal
            matching between predictions and ground truth based on various cost metrics.
        _cost_mask(bs, num_gts, masks=None, gt_mask=None): Computes the mask cost and dice cost when masks are available.
    """

    def __init__(self, cost_weight=None, use_fl=True, with_mask=False, num_sample_points=12544, alpha=0.25, gamma=2.0):
        """
        Initializes the HungarianMatcher with the provided cost coefficients, flags for Focal Loss and mask predictions,
        number of sample points for mask calculation, and parameters for Focal Loss.
        """
        super().__init__()
        # Default cost coefficients if none are provided
        if cost_weight is None:
            cost_weight = {"class": 1, "bbox": 5, "giou": 2, "mask": 1, "dice": 1}
        self.cost_weight = cost_gain
        self.use_fl = use_fl
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None):
        """
        Computes the cost matrix and performs the assignment between predicted and ground truth bounding boxes for a batch.

        Args:
            pred_bboxes (Tensor): Predicted bounding boxes of shape [batch_size, num_queries, 4].
            pred_scores (Tensor): Predicted scores of shape [batch_size, num_queries, num_classes].
            gt_cls (Tensor): Ground truth class labels of shape [num_gts, ].
            gt_bboxes (Tensor): Ground truth bounding boxes of shape [num_gts, 4].
            gt_groups (List[int]): List indicating the number of ground truth boxes per image in the batch.
            masks (Tensor, optional): Predicted masks of shape [batch_size, num_queries, height, width].
            gt_mask (List[Tensor], optional): List of ground truth masks, each of shape [num_masks, height, width].

        Returns:
            List[Tuple[Tensor, Tensor]]: A list where each element is a tuple (index_i, index_j) representing the indices of
            matched predictions and ground truth boxes for each batch item.
        """
        bs, nq, nc = pred_scores.shape

        # Return empty tensors if there are no ground truth boxes
        if sum(gt_groups) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]

        # Flatten tensors for batch processing
        pred_scores = pred_scores.detach().view(-1, nc)
        pred_scores = F.sigmoid(pred_scores) if self.use_fl else F.softmax(pred_scores, dim=-1)
        pred_bboxes = pred_bboxes.detach().view(-1, 4)

        # Compute classification cost
        pred_scores = pred_scores[:, gt_cls]
        if self.use_fl:
            neg_cost_class = (1 - self.alpha) * (pred_scores**self.gamma) * (-(1 - pred_scores + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - pred_scores) ** self.gamma) * (-(pred_scores + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -pred_scores

        # Compute L1 cost for bounding boxes
        cost_bbox = (pred_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)  # Shape: (bs*num_queries, num_gt)

        # Compute GIoU cost for bounding boxes
        cost_giou = 1.0 - bbox_iou(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0), xywh=True, GIoU=True).squeeze(-1)

        # Assemble the final cost matrix
        C = (
            self.cost_gain["class"] * cost_class
            + self.cost_gain["bbox"] * cost_bbox
            + self.cost_gain["giou"] * cost_giou
        )
        # Add mask cost if masks are used
        if self.with_mask:
            C += self._cost_mask(bs, gt_groups, masks, gt_mask)

        # Replace invalid values with zero
        C[C.isnan() | C.isinf()] = 0.0

        C = C.view(bs, nq, -1).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(gt_groups, -1))]
        gt_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)  # Compute cumulative sum for group indices
        return [
            (torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long) + gt_groups[k])
            for k, (i, j) in enumerate(indices)
        ]

    def _cost_mask(self, bs, num_gts, masks=None, gt_mask=None):
        """
        Computes the mask and dice costs if mask predictions and ground truth masks are provided.

        Args:
            bs (int): Batch size.
            num_gts (List[int]): List of number of ground truth boxes per image.
            masks (Tensor): Predicted masks of shape [batch_size, num_queries, height, width].
            gt_mask (List[Tensor]): List of ground truth masks.

        Returns:
            Tensor: Computed cost matrix including mask and dice costs.
        """
        assert masks is not None and gt_mask is not None, 'Ensure masks and gt_mask are provided'
        # Generate random sample points for efficient matching
        sample_points = torch.rand([bs, 1, self.num_sample_points, 2]) * 2.0 - 1.0

        # Sample masks and ground truth masks
        out_mask = F.grid_sample(masks.detach(), sample_points, align_corners=False).squeeze(-2)
        out_mask = out_mask.flatten(0, 1)

        tgt_mask = torch.cat(gt_mask).unsqueeze(1)
        sample_points = torch.cat([a.repeat(b, 1, 1, 1) for a, b in zip(sample_points, num_gts) if b > 0])
        tgt_mask = F.grid_sample(tgt_mask, sample_points, align_corners=False).squeeze([1, 2])

        with torch.cuda.amp.autocast(False):
            # Compute binary cross entropy cost
            pos_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.ones_like(out_mask), reduction='none')
            neg_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.zeros_like(out_mask), reduction='none')
            cost_mask = torch.matmul(pos_cost_mask, tgt_mask.T) + torch.matmul(neg_cost_mask, 1 - tgt_mask.T)
            cost_mask /= self.num_sample_points

            # Compute dice cost
            out_mask = F.sigmoid(out_mask)
            numerator = 2 * torch.matmul(out_mask, tgt_mask.T)
            denominator = out_mask.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
            cost_dice = 1 - (numerator + 1) / (denominator + 1)

            C = self.cost_gain['mask'] * cost_mask + self.cost_gain['dice'] * cost_dice
        return C


def get_cdn_group(
    batch, num_classes, num_queries, class_embed, num_dn=100, cls_noise_ratio=0.5, box_noise_scale=1.0, training=False
):
    """
    Creates a contrastive denoising training group with noisy class labels and bounding boxes.

    Args:
        batch (dict): Dictionary containing 'gt_cls', 'gt_bboxes', and 'gt_groups'.
        num_classes (int): Number of classes.
        num_queries (int): Number of queries.
        class_embed (Tensor): Class embeddings used for mapping class labels.
        num_dn (int, optional): Number of denoising samples. Defaults to 100.
        cls_noise_ratio (float, optional): Ratio of noisy class labels. Defaults to 0.5.
        box_noise_scale (float, optional): Scale for bounding box noise. Defaults to 1.0.
        training (bool, optional): Indicates whether the function is used in training mode. Defaults to False.

    Returns:
        Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]:
            - Modified ground truth class tensor.
            - Modified ground truth bounding box tensor.
            - Modified ground truth mask tensor (if available).
            - Dictionary containing the denoising group with noisy samples.
    """
    gt_cls = batch['gt_cls']
    gt_bboxes = batch['gt_bboxes']
    gt_groups = batch['gt_groups']

    if num_dn > 0:
        dn = {}
        if num_dn > 0 and training:
            # Create noisy class labels
            noise_cls = torch.full([len(gt_cls)], num_classes, dtype=torch.long)
            idx = torch.rand(len(gt_cls)) < cls_noise_ratio
            noise_cls[idx] = torch.randint(0, num_classes, (idx.sum(),), dtype=torch.long)
            gt_cls = torch.cat([gt_cls, noise_cls])
            # Create noisy bounding boxes
            noise_bboxes = gt_bboxes.new_zeros((num_dn, 4))
            noise_bboxes[:, 2:] = torch.rand((num_dn, 2)) * box_noise_scale
            gt_bboxes = torch.cat([gt_bboxes, noise_bboxes])
            # Create denoising group dictionary
            dn['gt_cls'] = gt_cls
            dn['gt_bboxes'] = gt_bboxes
            dn['num_dn'] = num_dn
        return gt_cls, gt_bboxes, None, dn
    else:
        return gt_cls, gt_bboxes, None, None

"""
Class HungarianMatcher:
    Initialize(cost_gain, use_fl, with_mask, num_sample_points, alpha, gamma):
        Set cost_gain to default values if not provided
        Set use_fl, with_mask, num_sample_points, alpha, gamma to provided values

    Function forward(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None):
        If gt_groups is all zeros:
            Return empty matching indices for each batch

        Flatten predictions and ground truths
        Compute classification scores:
            If use_fl is true:
                Calculate Focal Loss based classification cost
            Else:
                Use softmax to get classification cost
        Compute L1 cost between predicted and ground truth bounding boxes
        Compute GIoU cost between predicted and ground truth bounding boxes
        Combine classification cost, bbox cost, and GIoU cost into a final cost matrix C
        
        If with_mask is true:
            Add mask cost to C using _cost_mask function

        Replace NaNs and infinities in C with zeros
        Reshape cost matrix and apply Hungarian algorithm to get matching indices
        Adjust indices for each batch and return matching indices

    Function _cost_mask(bs, num_gts, masks=None, gt_mask=None):
        Assert masks and gt_mask are provided
        Generate sample points for masks
        Compute predicted and ground truth mask costs
        Calculate binary cross-entropy and dice cost
        Return combined mask cost

Function get_cdn_group(batch, num_classes, num_queries, class_embed, num_dn=100, cls_noise_ratio=0.5, box_noise_scale=1.0, training=False):
    If not training or num_dn is less than or equal to zero:
        Return None for all elements

    Extract ground truth information from batch
    Calculate number of denoising groups
    Pad ground truth data to match the maximum number of ground truths per batch

    Create denoising samples for classes and bounding boxes
    Add noise to class labels and bounding box coordinates if specified
    Calculate class embeddings and pad them
    Create attention mask to ensure no overlap in reconstruction

    Return padded class embeddings, bounding boxes, attention mask, and metadata

"""
