import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.utils.torch_utils import select_device
from .amg import (
    batch_iterator,
    batched_mask_to_box,
    build_all_layer_point_grids,
    calculate_stability_score,
    generate_crop_boxes,
    is_box_near_crop_edge,
    remove_small_regions,
    uncrop_boxes_xyxy,
    uncrop_masks,
)
from .build import build_sam

class SAMPredictor(BasePredictor):
    """
    SAMPredictor class for the Segment Anything Model (SAM), extending BasePredictor.

    This class provides an interface for performing image segmentation using SAM, capable of handling various prompts
    including bounding boxes, points, and masks. It supports both real-time and prompt-based segmentation tasks.

    Attributes:
        cfg (dict): Configuration settings for the model and task.
        overrides (dict): Configuration overrides.
        _callbacks (dict): Callbacks for custom behavior.
        args (namespace): Command-line arguments or operational variables.
        im (torch.Tensor): Preprocessed input image tensor.
        features (torch.Tensor): Image features used for inference.
        prompts (dict): Prompt settings for the model.
        segment_all (bool): Flag to determine if all objects in the image should be segmented.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the SAMPredictor with configurations, overrides, and callbacks.

        Args:
            cfg (dict): Configuration dictionary.
            overrides (dict, optional): Dictionary for configuration overrides.
            _callbacks (dict, optional): Callback functions for customization.
        """
        if overrides is None:
            overrides = {}
        overrides.update(dict(task="segment", mode="predict", imgsz=1024))
        super().__init__(cfg, overrides, _callbacks)
        self.args.retina_masks = True
        self.im = None
        self.features = None
        self.prompts = {}
        self.segment_all = False

    def preprocess_image(self, im):
        """
        Preprocess the input image to prepare it for model inference.

        Args:
            im (torch.Tensor | List[np.ndarray]): Image in tensor or numpy format.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        if self.im is not None:
            return self.im
        
        if not isinstance(im, torch.Tensor):
            im = np.stack(self.transform_images(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        if not isinstance(im, torch.Tensor):
            im = (im - self.mean) / self.std
        return im

    def transform_images(self, im):
        """
        Apply initial transformations to images for preprocessing.

        Args:
            im (List[np.ndarray]): List of images in HWC format.

        Returns:
            List[np.ndarray]: Transformed images.
        """
        assert len(im) == 1, "Batch processing is not supported"
        letterbox = LetterBox(self.args.imgsz, auto=False, center=False)
        return [letterbox(image=x) for x in im]

    def perform_inference(self, im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False, *args, **kwargs):
        """
        Run segmentation inference based on given prompts or generate new masks.

        Args:
            im (torch.Tensor): Preprocessed image tensor.
            bboxes (np.ndarray | List, optional): Bounding boxes for object location.
            points (np.ndarray | List, optional): Points indicating object location.
            labels (np.ndarray | List, optional): Labels for points.
            masks (np.ndarray, optional): Low-resolution masks from previous results.
            multimask_output (bool, optional): Whether to return multiple masks.

        Returns:
            tuple: Contains masks, scores, and low-resolution logits.
        """
        bboxes = self.prompts.pop("bboxes", bboxes)
        points = self.prompts.pop("points", points)
        masks = self.prompts.pop("masks", masks)

        if all(i is None for i in [bboxes, points, masks]):
            return self.create_masks(im, *args, **kwargs)

        return self.perform_prompt_inference(im, bboxes, points, labels, masks, multimask_output)

    def perform_prompt_inference(self, im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False):
        """
        Conduct inference using specific prompts (e.g., bounding boxes, points).

        Args:
            im (torch.Tensor): Preprocessed image tensor.
            bboxes (np.ndarray | List, optional): Bounding boxes.
            points (np.ndarray | List, optional): Points for object location.
            labels (np.ndarray | List, optional): Labels for points.
            masks (np.ndarray, optional): Low-resolution masks.
            multimask_output (bool, optional): Flag for multiple mask outputs.

        Returns:
            tuple: Masks, scores, and logits.
        """
        features = self.model.image_encoder(im) if self.features is None else self.features

        src_shape, dst_shape = self.batch[1][0].shape[:2], im.shape[2:]
        scale_factor = 1.0 if self.segment_all else min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])
        
        if points is not None:
            points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
            points = points[None] if points.ndim == 1 else points
            if labels is None:
                labels = np.ones(points.shape[0])
            labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
            points *= scale_factor
            points, labels = points[:, None, :], labels[:, None]
        
        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32, device=self.device)
            bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
            bboxes *= scale_factor
        
        if masks is not None:
            masks = torch.as_tensor(masks, dtype=torch.float32, device=self.device).unsqueeze(1)

        points = (points, labels) if points is not None else None
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=points, boxes=bboxes, masks=masks)

        pred_masks, pred_scores = self.model.mask_decoder(
            image_embeddings=features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        return pred_masks.flatten(0, 1), pred_scores.flatten(0, 1)

    def create_masks(
        self,
        im,
        crop_layers=0,
        crop_overlap=512 / 1500,
        downscale_factor=1,
        point_grids=None,
        stride=32,
        batch_size=64,
        confidence_threshold=0.88,
        stability_threshold=0.95,
        stability_offset=0.95,
        nms_threshold=0.7,
    ):
        """
        Segment the image into parts using the SAM model, optionally processing crops.

        Args:
            im (torch.Tensor): Input image tensor.
            crop_layers (int): Number of crop layers.
            crop_overlap (float): Overlap ratio between crops.
            downscale_factor (int): Downscale factor for sampling points.
            point_grids (list[np.ndarray], optional): Custom grids for point sampling.
            stride (int, optional): Sampling stride for points.
            batch_size (int): Batch size for processing points.
            confidence_threshold (float): Threshold for mask confidence.
            stability_threshold (float): Threshold for mask stability.
            stability_offset (float): Offset for stability score calculation.
            nms_threshold (float): IoU threshold for Non-Maximum Suppression.

        Returns:
            tuple: Masks, scores, and bounding boxes.
        """
        self.segment_all = True
        ih, iw = im.shape[2:]
        crop_boxes, layer_indices = generate_crop_boxes((ih, iw), crop_layers, crop_overlap)
        if point_grids is None:
            point_grids = build_all_layer_point_grids(stride, crop_layers, downscale_factor)
        
        all_masks, all_scores, all_bboxes, areas = [], [], [], []
        for box, layer_idx in zip(crop_boxes, layer_indices):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            area = torch.tensor(w * h, device=im.device)
            points_scale = np.array([[w, h]])
            cropped_im = F.interpolate(im[..., y1:y2, x1:x2], size=(self.args.imgsz, self.args.imgsz), mode='bilinear', align_corners=False)
            masks, scores = self.perform_prompt_inference(cropped_im, *args, **kwargs)
            masks, scores = self.process_crops(masks, scores, box, points_scale, area, confidence_threshold, stability_threshold, stability_offset, nms_threshold)
            all_masks.append(masks)
            all_scores.append(scores)
            all_bboxes.append(box)
            areas.append(area)
        
        # Combine and filter masks
        all_masks = torch.cat(all_masks, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_bboxes = torch.tensor(all_bboxes, dtype=torch.float32, device=im.device)
        areas = torch.cat(areas, dim=0)
        all_bboxes = uncrop_boxes_xyxy(all_bboxes, (ih, iw))
        all_masks = uncrop_masks(all_masks, all_bboxes, (ih, iw), crop_layers)

        # Non-Maximum Suppression
        keep = ops.non_max_suppression(all_bboxes, all_scores, iou_thres=nms_threshold)
        return all_masks[keep], all_scores[keep], all_bboxes[keep]

    def process_crops(self, masks, scores, box, points_scale, area, confidence_threshold, stability_threshold, stability_offset, nms_threshold):
        """
        Process and filter masks from cropped images.

        Args:
            masks (torch.Tensor): Mask predictions from the model.
            scores (torch.Tensor): Mask scores.
            box (tuple): Crop bounding box coordinates.
            points_scale (np.ndarray): Scaling factors for points.
            area (torch.Tensor): Area of the crop.
            confidence_threshold (float): Confidence threshold.
            stability_threshold (float): Stability threshold.
            stability_offset (float): Offset for stability score calculation.
            nms_threshold (float): IoU threshold for Non-Maximum Suppression.

        Returns:
            tuple: Processed masks and scores.
        """
        scores, masks = self.filter_by_confidence(scores, masks, confidence_threshold)
        scores, masks = self.filter_by_stability(scores, masks, area, stability_threshold, stability_offset)
        scores, masks = self.apply_nms(scores, masks, box, nms_threshold)
        return masks, scores

    def filter_by_confidence(self, scores, masks, threshold):
        """
        Filter masks by confidence score.

        Args:
            scores (torch.Tensor): Mask scores.
            masks (torch.Tensor): Mask predictions.
            threshold (float): Confidence threshold.

        Returns:
            tuple: Filtered scores and masks.
        """
        keep = scores >= threshold
        return scores[keep], masks[keep]

    def filter_by_stability(self, scores, masks, area, threshold, offset):
        """
        Filter masks by stability score.

        Args:
            scores (torch.Tensor): Mask scores.
            masks (torch.Tensor): Mask predictions.
            area (torch.Tensor): Area of the crop.
            threshold (float): Stability threshold.
            offset (float): Offset for stability score calculation.

        Returns:
            tuple: Filtered scores and masks.
        """
        stability_scores = calculate_stability_score(masks, area, offset)
        keep = stability_scores >= threshold
        return scores[keep], masks[keep]

    def apply_nms(self, scores, masks, box, threshold):
        """
        Apply Non-Maximum Suppression to filter overlapping masks.

        Args:
            scores (torch.Tensor): Mask scores.
            masks (torch.Tensor): Mask predictions.
            box (tuple): Bounding box coordinates.
            threshold (float): IoU threshold for NMS.

        Returns:
            tuple: Masks and scores after NMS.
        """
        bboxes = batched_mask_to_box(masks, box)
        keep = ops.non_max_suppression(bboxes, scores, iou_thres=threshold)
        return masks[keep], scores[keep]

# Example usage
if __name__ == "__main__":
    # Initialize the model and predictor
    device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = SAMPredictor(cfg=DEFAULT_CFG, overrides={'imgsz': 1024})
    predictor.model.to(device)

    # Load and preprocess the image
    image_path = 'path/to/image.jpg'
    image = torchvision.io.read_image(image_path).unsqueeze(0).float() / 255.0
    image = image.to(device)

    # Run segmentation
    masks, scores, bboxes = predictor.perform_inference(image)

    # Process results
    print(f"Detected {len(masks)} masks with scores: {scores}")
    for mask in masks:
        # Convert mask to PIL image or other formats for visualization
        pass

"""
Class SAMPredictor inherits from BasePredictor
    Method initialize(cfg, overrides=None, _callbacks=None)
        Call parent class initialize
        Set special attributes for the model
        Initialize image, features, and prompts

    Method preprocess_image(im)
        If im is already preprocessed
            Return im
        If im is not a tensor
            Convert im to tensor (e.g., Numpy array to tensor)
        Convert image to appropriate format and device
        Return preprocessed image

    Method transform_images(im)
        Apply initial transformations to the input image (e.g., resizing)
        Return list of transformed images

    Method perform_inference(im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False, *args, **kwargs)
        If no prompts (e.g., bounding boxes, points, or masks) are provided
            Create masks
        Otherwise, perform prompt-based inference
        Return masks, scores, and low-resolution logits

    Method perform_prompt_inference(im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False)
        If feature maps are not available
            Obtain features from the model
        Perform inference based on prompts (points, bounding boxes, masks)
        Return masks and scores

    Method create_masks(im, crop_layers=0, crop_overlap=0.34, downscale_factor=1, point_grids=None, stride=32, batch_size=64, confidence_threshold=0.88, stability_threshold=0.95, stability_offset=0.95, nms_threshold=0.7)
        Generate cropping boxes for the image
        For each cropping box
            Crop and infer from the image
        Process and combine results from each cropped region
        Return combined masks

"""
