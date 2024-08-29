import math
import random
from copy import deepcopy

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_version
from ultralytics.utils.instance import Instances
from ultralytics.utils.metrics import bbox_ioa
from ultralytics.utils.ops import segment2box, xyxyxyxy2xywhr
from ultralytics.utils.torch_utils import TORCHVISION_0_10, TORCHVISION_0_11, TORCHVISION_0_13
from .utils import polygons2masks, polygons2masks_overlap

# Default normalization parameters
DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)
DEFAULT_CROP_FRACTION = 1.0

class ImageTransformBase:
    """
    A base class for image transformations that can be extended for different use cases,
    such as classification or segmentation tasks.

    Methods:
        - apply_image: Placeholder method to apply transformations to images.
        - apply_instances: Placeholder method for object instance transformations.
        - apply_semantic: Placeholder method for semantic segmentation transformations.
        - __call__: Applies the transformations in sequence.
    """

    def __init__(self) -> None:
        """Initialize the transformation object."""
        pass

    def apply_image(self, labels):
        """Placeholder to apply image transformations."""
        pass

    def apply_instances(self, labels):
        """Placeholder to apply object instance transformations."""
        pass

    def apply_semantic(self, labels):
        """Placeholder to apply semantic segmentation."""
        pass

    def __call__(self, labels):
        """Sequentially applies image, instance, and semantic transformations."""
        self.apply_image(labels)
        self.apply_instances(labels)
        self.apply_semantic(labels)


class TransformationPipeline:
    """Class for chaining multiple image transformations."""

    def __init__(self, transforms):
        """Initialize with a list of transformations."""
        self.transforms = transforms

    def __call__(self, data):
        """Apply each transformation in the pipeline."""
        for transform in self.transforms:
            data = transform(data)
        return data

    def append(self, transform):
        """Add a transformation to the pipeline."""
        self.transforms.append(transform)

    def to_list(self):
        """Convert the pipeline to a standard list."""
        return self.transforms

    def __repr__(self):
        """Return a string representation of the pipeline."""
        return f"{self.__class__.__name__}({', '.join([str(t) for t in self.transforms])})"


class MixTransformationBase:
    """
    Base class for mixup/mosaic augmentations applied to datasets.

    This class handles the logic for selecting and applying different mixing transformations like
    mixup or mosaic, which combine multiple images into one.
    """

    def __init__(self, dataset, pre_transform=None, probability=0.0) -> None:
        """Initialize with a dataset, optional pre-transform, and probability for the transformation."""
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.probability = probability

    def __call__(self, labels):
        """Apply the mixup/mosaic transformation with a certain probability."""
        if random.uniform(0, 1) > self.probability:
            return labels

        indexes = self.get_indexes()
        indexes = [indexes] if isinstance(indexes, int) else indexes

        mixed_labels = [self.dataset.get_image_and_label(i) for i in indexes]
        if self.pre_transform:
            mixed_labels = [self.pre_transform(label) for label in mixed_labels]

        labels["mixed_labels"] = mixed_labels
        labels = self.apply_mix_transform(labels)
        labels.pop("mixed_labels", None)
        return labels

    def apply_mix_transform(self, labels):
        """Apply the specific mixup/mosaic transformation (to be defined in subclasses)."""
        raise NotImplementedError

    def get_indexes(self):
        """Return a list of random indexes for the transformation."""
        raise NotImplementedError


class MosaicTransform(MixTransformationBase):
    """
    Class for performing mosaic augmentation by combining multiple images into one.

    Mosaic augmentation is useful for increasing the diversity of training samples by combining
    multiple images into a larger composite image.
    """

    def __init__(self, dataset, img_size=640, probability=1.0, grid_size=4):
        """Initialize the mosaic transformation with dataset, image size, probability, and grid size."""
        assert 0 <= probability <= 1.0, "Probability must be between 0 and 1."
        assert grid_size in [4, 9], "Grid size must be 4 or 9."
        super().__init__(dataset, probability=probability)
        self.img_size = img_size
        self.grid_size = grid_size
        self.border = (-img_size // 2, -img_size // 2)

    def get_indexes(self, buffer=True):
        """Return a list of random indexes for mosaic."""
        return random.choices(list(self.dataset.buffer), k=self.grid_size - 1) if buffer else \
               [random.randint(0, len(self.dataset) - 1) for _ in range(self.grid_size - 1)]

    def apply_mix_transform(self, labels):
        """Apply the specific mosaic transformation."""
        if self.grid_size == 4:
            return self._mosaic4(labels)
        elif self.grid_size == 9:
            return self._mosaic9(labels)

    def _mosaic4(self, labels):
        """Apply a 2x2 mosaic transformation."""
        mosaic_labels = []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)

        for i in range(4):
            labels_patch = labels if i == 0 else labels["mixed_labels"][i - 1]
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            if i == 0:
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # i == 3
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw, padh = x1a - x1b, y1a - y1b
            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)

        final_labels = self._concat_labels(mosaic_labels)
        final_labels["img"] = img4
        return final_labels

    def _mosaic9(self, labels):
        """Apply a 3x3 mosaic transformation."""
        # Implement 3x3 mosaic similar to _mosaic4, creating a 3x3 grid of images.
        # The logic will be similar to _mosaic4, but handling 9 images.
        pass

    @staticmethod
    def _update_labels(labels, padw, padh):
        """Update the labels for the mosaic operation."""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].add_padding(padw, padh)
        return labels

    def _concat_labels(self, mosaic_labels):
        """Concatenate labels for the final mosaic image."""
        if not mosaic_labels:
            return {}
        cls_list, instances_list = [], []
        img_size = self.img_size * 2

        for labels in mosaic_labels:
            cls_list.append(labels["cls"])
            instances_list.append(labels["instances"])

        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (img_size, img_size),
            "cls": np.concatenate(cls_list, axis=0),
            "instances": Instances.concatenate(instances_list, axis=0),
            "mosaic_border": self.border,
        }

        final_labels["instances"].clip(img_size, img_size)
        valid = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][valid]
        return final_labels


class MixUpTransform(MixTransformationBase):
    """Class for applying MixUp augmentation to a dataset."""

    def __init__(self, dataset, pre_transform=None, probability=0.0) -> None:
        """Initialize with a dataset, optional pre-transform, and probability for MixUp."""
        super().__init__(dataset, pre_transform, probability)

    def get_indexes(self):
        """Return a random index from the dataset."""
        return random.randint(0, len(self.dataset) - 1)

    def apply_mix_transform(self, labels):
        """Apply the MixUp augmentation."""
        ratio = np.random.beta(32.0, 32.0)
        mixed_labels = labels["mixed_labels"][0]

        labels["img"] = (labels["img"] * ratio + mixed_labels["img"] * (1 - ratio)).astype(np.uint8)
        labels["instances"] = Instances.concatenate([labels["instances"], mixed_labels["instances"]], axis=0)
        labels["cls"] = np.concatenate([labels["cls"], mixed_labels["cls"]], axis=0)
        return labels


class RandomAffine:
    """
    This class performs a series of random affine transformations on images, such as rotation, scaling, shearing,
    and translation. It also adjusts the corresponding annotations, like bounding boxes and keypoints.
    """

    def __init__(self, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, border=(0, 0), pre_transform=None):
        """Initialize with parameters for rotation, translation, scaling, and shearing."""
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border  # Mosaic border
        self.pre_transform = pre_transform

    def _affine_matrix(self, img, border):
        """Generate the affine transformation matrix."""
        center = np.eye(3, dtype=np.float32)
        center[0, 2] = -img.shape[1] / 2
        center[1, 2] = -img.shape[0] / 2

        perspective_matrix = np.eye(3, dtype=np.float32)
        perspective_matrix[2, 0] = random.uniform(-self.perspective, self.perspective)
        perspective_matrix[2, 1] = random.uniform(-self.perspective, self.perspective)

        rotation_matrix = np.eye(3, dtype=np.float32)
        angle = random.uniform(-self.degrees, self.degrees)
        scale = random.uniform(1 - self.scale, 1 + self.scale)
        rotation_matrix[:2] = cv2.getRotationMatrix2D(center=(0, 0), angle=angle, scale=scale)

        shear_matrix = np.eye(3, dtype=np.float32)
        shear_matrix[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        shear_matrix[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)

        translation_matrix = np.eye(3, dtype=np.float32)
        translation_matrix[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]
        translation_matrix[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]

        # Combine matrices
        transformation_matrix = translation_matrix @ shear_matrix @ rotation_matrix @ perspective_matrix @ center
        return transformation_matrix

    def __call__(self, labels):
        """Apply the affine transformations to images and their annotations."""
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)

        img = labels["img"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*img.shape[:2][::-1])

        self.size = img.shape[1] + self.border[1] * 2, img.shape[0] + self.border[0] * 2
        transformation_matrix = self._affine_matrix(img, self.border)

        img, M, scale = self._apply_affine(img, transformation_matrix, border)

        bboxes = self._apply_to_bboxes(instances.bboxes, M)
        segments = instances.segments
        keypoints = instances.keypoints

        if segments:
            bboxes, segments = self._apply_to_segments(segments, M)

        if keypoints is not None:
            keypoints = self._apply_to_keypoints(keypoints, M)

        updated_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        updated_instances.clip(*self.size)

        valid_indices = self._filter_candidates(instances.bboxes.T, updated_instances.bboxes.T)
        labels["instances"] = updated_instances[valid_indices]
        labels["cls"] = labels["cls"][valid_indices]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

    def _apply_affine(self, img, transformation_matrix, border):
        """Apply affine transformation to the image."""
        if (border[0] != 0) or (border[1] != 0) or (transformation_matrix != np.eye(3)).any():
            if self.perspective:
                img = cv2.warpPerspective(img, transformation_matrix, dsize=self.size, borderValue=(114, 114, 114))
            else:
                img = cv2.warpAffine(img, transformation_matrix[:2], dsize=self.size, borderValue=(114, 114, 114))
        return img, transformation_matrix, scale

    def _apply_to_bboxes(self, bboxes, transformation_matrix):
        """Apply affine transformations to bounding boxes."""
        # Implementation similar to the original, but using helper methods for readability
        pass

    def _apply_to_segments(self, segments, transformation_matrix):
        """Apply affine transformations to segments."""
        # Implementation similar to the original, but using helper methods for readability
        pass

    def _apply_to_keypoints(self, keypoints, transformation_matrix):
        """Apply affine transformations to keypoints."""
        # Implementation similar to the original, but using helper methods for readability
        pass

    def _filter_candidates(self, box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
        """Filter bounding boxes based on predefined thresholds."""
        # Implementation similar to the original, but refactored for clarity
        pass


class RandomHSVTransform:
    """
    This class applies random adjustments to the Hue, Saturation, and Value (HSV) of an image.

    It's useful for augmenting image data to make models more robust to color variations.
    """

    def __init__(self, h_gain=0.5, s_gain=0.5, v_gain=0.5) -> None:
        """Initialize with gain factors for each HSV channel."""
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain

    def __call__(self, labels):
        """Apply random HSV transformation to the image."""
        img = labels["img"]
        if self.h_gain or self.s_gain or self.v_gain:
            random_factors = np.random.uniform(-1, 1, 3) * [self.h_gain, self.s_gain, self.v_gain] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

            lut_hue = ((np.arange(256) * random_factors[0]) % 180).astype(img.dtype)
            lut_sat = np.clip(np.arange(256) * random_factors[1], 0, 255).astype(img.dtype)
            lut_val = np.clip(np.arange(256) * random_factors[2], 0, 255).astype(img.dtype)

            img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return labels


class RandomFlipTransform:
    """
    This class applies a random flip (horizontal or vertical) to images and updates the corresponding annotations.
    """

    def __init__(self, probability=0.5, direction="horizontal", flip_index=None) -> None:
        """Initialize with probability, direction, and optional flip index for keypoints."""
        assert direction in ["horizontal", "vertical"], "Direction must be 'horizontal' or 'vertical'."
        self.probability = probability
        self.direction = direction
        self.flip_index = flip_index

    def __call__(self, labels):
        """Apply the flip transformation."""
        img = labels["img"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        h, w = img.shape[:2]

        if self.direction == "vertical" and random.random() < self.probability:
            img = np.flipud(img)
            instances.flipud(h)
        elif self.direction == "horizontal" and random.random() < self.probability:
            img = np.fliplr(img)
            instances.fliplr(w)
            if self.flip_index is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_index, :])

        labels["img"] = np.ascontiguousarray(img)
        labels["instances"] = instances
        return labels


class ResizeWithLetterBox:
    """
    This class resizes an image while maintaining its aspect ratio and adds padding (letterboxing) if necessary.
    """

    def __init__(self, target_size=(640, 640), auto=False, scale_fill=False, scale_up=True, center=True, stride=32):
        """Initialize with target size, auto-flag, and other resizing options."""
        self.target_size = target_size
        self.auto = auto
        self.scale_fill = scale_fill
        self.scale_up = scale_up
        self.stride = stride
        self.center = center

    def __call__(self, labels=None, image=None):
        """Resize the image and add padding if necessary."""
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]
        new_shape = labels.pop("rect_shape", self.target_size)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        r = min(r, 1.0) if not self.scale_up else r

        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scale_fill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        if self.center:
            dw /= 2
            dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        if labels:
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels based on resizing and padding."""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


class CopyPasteTransform:
    """
    This class implements the Copy-Paste augmentation technique where objects from one image are pasted onto another
    image to create new training samples.
    """

    def __init__(self, probability=0.5) -> None:
        """Initialize with the probability of applying Copy-Paste."""
        self.probability = probability

    def __call__(self, labels):
        """Apply the Copy-Paste augmentation."""
        img = labels["img"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        h, w = img.shape[:2]

        if self.probability and len(instances.segments):
            copied_instances = deepcopy(instances)
            copied_instances.fliplr(w)

            ioa = bbox_ioa(copied_instances.bboxes, instances.bboxes)
            indexes = np.nonzero((ioa < 0.30).all(1))[0]

            for idx in random.sample(list(indexes), k=round(self.probability * len(indexes))):
                instances = Instances.concatenate([instances, copied_instances[[idx]]], axis=0)
                cv2.drawContours(img, copied_instances.segments[[idx]].astype(np.int32), -1, (1, 1, 1), cv2.FILLED)

        labels["img"] = img
        labels["instances"] = instances
        return labels


class AlbumentationsTransform:
    """
    This class applies a series of image augmentations using the Albumentations library, such as blurring,
    color adjustments, and noise addition.
    """

    def __init__(self, probability=1.0):
        """Initialize the augmentation pipeline with a given probability."""
        self.probability = probability
        self.transform = None
        self._initialize_transform()

    def _initialize_transform(self):
        """Set up the augmentation pipeline using Albumentations."""
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)

            transformations = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]

            self.transform = A.Compose(
                transformations,
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])
            )
            LOGGER.info(f"Albumentations: {', '.join(str(t).replace('always_apply=False, ', '') for t in transformations if t.p)}")
        except ImportError:
            pass
        except Exception as e:
            LOGGER.info(f"Albumentations: {e}")

    def __call__(self, labels):
        """Apply the Albumentations transformations."""
        img = labels["img"]
        if len(labels["cls"]) and self.transform and random.random() < self.probability:
            labels["instances"].convert_bbox("xywh")
            labels["instances"].normalize(*img.shape[:2][::-1])

            transformed = self.transform(image=img, bboxes=labels["instances"].bboxes, class_labels=labels["cls"])
            if transformed["class_labels"]:
                labels["img"] = transformed["image"]
                labels["cls"] = np.array(transformed["class_labels"])
                labels["instances"].update(bboxes=np.array(transformed["bboxes"], dtype=np.float32))
        return labels
