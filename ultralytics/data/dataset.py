import os
import cv2
import numpy as np
from pathlib import Path
from multiprocessing.pool import ThreadPool
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

# Constants for cache versioning
CACHE_VERSION = "1.1.0"

class YOLODataset(Dataset):

    def __init__(self, data_config=None, task_type="detection", **kwargs):
        """
        Initializes the YOLODataset with configurations for detection, segmentation, or keypoint tasks.

        Args:
            data_config (dict, optional): Configuration dictionary containing dataset paths and parameters.
            task_type (str): Defines the type of task - 'detection', 'segmentation', or 'keypoints'.
        """
        self.task_type = task_type
        self.data_config = data_config
        self.use_keypoints = task_type == "keypoints"
        self.use_segmentation = task_type == "segmentation"
        super().__init__(**kwargs)

    def cache_data(self, cache_path="./data_cache.cache"):
        """
        Caches dataset labels and image metadata to improve loading efficiency.

        Args:
            cache_path (str): Path where the cache file should be saved.

        Returns:
            dict: Cached dataset information including labels and image shapes.
        """
        cache = {"labels": []}
        nm, nf, ne, nc = 0, 0, 0, 0  # counters for missing, found, empty, corrupt
        image_files = self.data_config["image_files"]
        total_images = len(image_files)

        with ThreadPool(os.cpu_count()) as pool:
            results = pool.imap(self._verify_image_and_label, image_files)
            for result in results:
                im_file, label_data, status = result
                if status == "found":
                    nf += 1
                    cache["labels"].append(label_data)
                elif status == "missing":
                    nm += 1
                elif status == "empty":
                    ne += 1
                elif status == "corrupt":
                    nc += 1

        cache["hash"] = self._generate_hash(image_files)
        cache["summary"] = (nf, nm, ne, nc, total_images)
        self._save_cache(cache_path, cache)
        return cache

    def _verify_image_and_label(self, im_file):
        """
        Verifies the existence and integrity of an image and its corresponding label.

        Args:
            im_file (str): Path to the image file.

        Returns:
            tuple: Image file path, label data, and status of verification.
        """
        label_file = self._get_label_path(im_file)
        if not os.path.exists(im_file) or not os.path.exists(label_file):
            return im_file, None, "missing"

        label_data = self._load_label(label_file)
        if label_data is None:
            return im_file, None, "empty"

        return im_file, label_data, "found"

    def _get_label_path(self, im_file):
        """
        Generates the corresponding label file path for a given image file.

        Args:
            im_file (str): Path to the image file.

        Returns:
            str: Path to the label file.
        """
        label_dir = self.data_config["label_dir"]
        label_file = os.path.join(label_dir, os.path.splitext(os.path.basename(im_file))[0] + ".txt")
        return label_file

    def _load_label(self, label_file):
        """
        Loads and processes label data from a file.

        Args:
            label_file (str): Path to the label file.

        Returns:
            dict: Processed label data.
        """
        try:
            with open(label_file, 'r') as file:
                labels = np.loadtxt(file, delimiter=' ')
            return {
                "cls": labels[:, 0:1],
                "bboxes": labels[:, 1:],
                "bbox_format": "xywh",
                "normalized": True
            }
        except Exception as e:
            print(f"Failed to load label: {e}")
            return None

    def _generate_hash(self, files):
        """
        Generates a unique hash based on the list of files.

        Args:
            files (list): List of file paths.

        Returns:
            str: Unique hash value.
        """
        return str(hash(tuple(sorted(files))))

    def _save_cache(self, cache_path, cache_data):
        """
        Saves cache data to a specified path.

        Args:
            cache_path (str): Path where the cache data should be saved.
            cache_data (dict): Data to be cached.
        """
        if os.access(os.path.dirname(cache_path), os.W_OK):
            np.save(cache_path, cache_data)
        else:
            print(f"Cache directory {os.path.dirname(cache_path)} is not writable. Cache not saved.")

    def get_data(self):
        """
        Retrieves and returns dataset labels from cache or by loading them.

        Returns:
            list: List of labels for the dataset.
        """
        cache_path = os.path.join(self.data_config["label_dir"], "data_cache.cache")
        if os.path.exists(cache_path):
            cache = np.load(cache_path, allow_pickle=True).item()
            if cache["hash"] == self._generate_hash(self.data_config["image_files"]):
                return cache["labels"]
            else:
                return self.cache_data(cache_path)["labels"]
        else:
            return self.cache_data(cache_path)["labels"]

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data_config["image_files"])

    def __getitem__(self, index):
        """Returns a single sample from the dataset at the specified index."""
        image_path = self.data_config["image_files"][index]
        label_data = self.get_data()[index]

        # Load image and apply transformations
        image = Image.open(image_path).convert("RGB")
        transform = self._get_transform()
        image = transform(image)

        # Return image and corresponding label
        return {"img": image, "cls": label_data["cls"], "bboxes": label_data["bboxes"]}

    def _get_transform(self):
        """
        Returns a transformation pipeline for preprocessing images.

        Returns:
            Callable: A function that applies transformations to an image.
        """
        if self.task_type == "detection":
            return T.Compose([
                T.Resize((self.data_config["img_size"], self.data_config["img_size"])),
                T.ToTensor(),
            ])
        else:
            return T.Compose([T.ToTensor()])


class ClassificationDataset(Dataset):
    """
    Custom dataset for image classification tasks.
    """

    def __init__(self, root_dir, image_size=224, augment=False):
        """
        Initializes the ClassificationDataset with root directory and optional transformations.

        Args:
            root_dir (str): Directory containing image files.
            image_size (int): Target image size for resizing.
            augment (bool): Whether to apply data augmentation.
        """
        super().__init__()
        self.root_dir = root_dir
        self.image_paths = list(Path(root_dir).glob('**/*.jpg'))
        self.image_size = image_size
        self.augment = augment
        self.transform = self._build_transform()

    def _build_transform(self):
        """
        Builds a transformation pipeline for image preprocessing.

        Returns:
            Callable: Transformation function.
        """
        transform_list = [T.Resize((self.image_size, self.image_size)), T.ToTensor()]
        if self.augment:
            transform_list.insert(1, T.RandomHorizontalFlip())
        return T.Compose(transform_list)

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Retrieves and returns an image and its corresponding label."""
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Assuming directory structure as root/class_name/image.jpg for labels
        label = image_path.parent.name
        return {"img": image, "cls": label}

"""
### YOLODataset Class

Initialize YOLODataset
   - Accept dataset configuration and task type (detection, segmentation, keypoints).
   - Set flags for keypoints and segmentation based on task type.

cache_data(cache_path)
   - Initialize an empty cache dictionary.
   - Iterate over image files in the dataset.
     - For each image:
       - Verify if both image and label files exist.
       - If valid, add label data to the cache dictionary.
   - Generate a hash for the dataset and store it in the cache.
   - Save the cache data to a file at `cache_path`.
   - Return the cache data.

_verify_image_and_label(im_file)
   - Generate the corresponding label file path for the given image file.
   - Check if the image and label files exist.
   - If both files are valid, load and return the label data.
   - If not, return an appropriate status indicating missing, empty, or corrupt files.

_get_label_path(im_file)
   - Construct the path to the label file based on the image file name.

_load_label(label_file)
   - Open and read the label file.
   - Process and return the label data (classes, bounding boxes, etc.).
   - If there is an error, return `None`.

_generate_hash(files)
   - Create and return a unique hash value based on the list of files.

_save_cache(cache_path, cache_data)
   - Check if the directory where the cache is to be saved is writable.
   - If writable, save the cache data to the specified path.
   - If not writable, output a warning message.

get_data()
   - Attempt to load cache data from the specified cache path.
   - If cache exists and is valid, return the labels from the cache.
   - If the cache is missing or invalid, regenerate the cache and return the labels.

__len__()
   - Return the number of images in the dataset.

__getitem__(index)
    - Retrieve the image and label data for the given index.
    - Apply the necessary transformations to the image.
    - Return the transformed image and corresponding label data.

_get_transform()
    - Build and return the transformation pipeline based on task type (detection, classification).
    - Apply resizing, normalization, and other augmentations if necessary.

### ClassificationDataset Class

Initialize ClassificationDataset
   - Set the root directory, image size, and whether to apply augmentations.
   - Create a list of all image file paths in the root directory.
   - Build the transformation pipeline.

_build_transform()
   - Create a list of transformations (resizing, normalization).
   - If augmentations are enabled, add random horizontal flip to the list.
   - Return the transformation pipeline.

__len__()
   - Return the total number of images in the dataset.

__getitem__(index)
   - Retrieve the image file path at the specified index.
   - Open the image and apply the transformations.
   - Determine the class label from the directory structure (assume folder names are class names).
   - Return the transformed image and its class label.

"""
