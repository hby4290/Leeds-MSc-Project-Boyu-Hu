# Rewritten YOLO Loader Classes

import os
import cv2
import time
import torch
import numpy as np
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
from PIL import Image
from dataclasses import dataclass
from collections import deque

from ultralytics.utils import LOGGER, check_requirements
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS


@dataclass
class InputSource:
    """Data class to represent various input source types."""
    webcam: bool = False
    screenshot: bool = False
    image_file: bool = False
    tensor_input: bool = False


class StreamLoader:
    """
    Handles loading and streaming of video sources for real-time processing.

    This class manages the input video streams and ensures that frames are efficiently read
    and made available for processing. It supports various video protocols including RTSP, RTMP, and HTTP.

    Attributes:
        sources (list): List of source URLs or file paths for video streams.
        image_size (int): The target image size for processing.
        frame_stride (int): The stride for reading frames from the video streams.
        buffer_mode (bool): Whether to buffer frames in memory.
        active (bool): Flag to indicate if the loader is actively streaming.
        video_captures (list): List of OpenCV video capture objects.
        frame_queues (list): Lists to hold frames for each video stream.
        shapes (list): List to store the shape of each video stream.
        threads (list): List of threads handling video frame reading.
        frame_rates (list): List of frame rates for each video stream.

    Methods:
        __init__: Initialize the StreamLoader with sources and configurations.
        start_stream: Starts a thread to read frames from each stream.
        fetch_frames: Reads and buffers frames from the video streams.
        stop_streams: Stops all active video streams and releases resources.
        __iter__: Returns an iterator object for processing the frames.
        __next__: Returns the next batch of frames from the streams.
        __len__: Returns the number of sources being streamed.
    """

    def __init__(self, sources="streams.txt", image_size=640, frame_stride=1, buffer_mode=False):
        """Initialize the stream loader with given sources and configurations."""
        self.sources = self._load_sources(sources)
        self.image_size = image_size
        self.frame_stride = frame_stride
        self.buffer_mode = buffer_mode
        self.active = True

        self.video_captures = []
        self.frame_queues = []
        self.shapes = []
        self.threads = []
        self.frame_rates = []

        self._initialize_streams()

    def _load_sources(self, sources):
        """Load video sources from a file or string."""
        if os.path.isfile(sources):
            return Path(sources).read_text().splitlines()
        return [sources]

    def _initialize_streams(self):
        """Initialize video capture objects and start streaming threads."""
        for source in self.sources:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video source: {source}")
            self.video_captures.append(cap)
            self.frame_queues.append(deque(maxlen=30 if self.buffer_mode else 1))
            self.shapes.append(self._get_frame_shape(cap))
            self.frame_rates.append(self._get_frame_rate(cap))
            thread = Thread(target=self._fetch_frames, args=(cap, len(self.threads)), daemon=True)
            self.threads.append(thread)
            thread.start()

    def _get_frame_shape(self, cap):
        """Retrieve the shape of frames from a video capture object."""
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (height, width, 3)

    def _get_frame_rate(self, cap):
        """Retrieve the frame rate of the video capture."""
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30  # Default to 30 FPS if not available

    def _fetch_frames(self, cap, index):
        """Fetch frames from the video capture and store in the queue."""
        while self.active and cap.isOpened():
            if len(self.frame_queues[index]) < self.frame_queues[index].maxlen:
                cap.grab()
                if cap.get(cv2.CAP_PROP_POS_FRAMES) % self.frame_stride == 0:
                    success, frame = cap.retrieve()
                    if success:
                        self.frame_queues[index].append(frame)
                    else:
                        LOGGER.warning(f"Failed to retrieve frame for stream {index}.")
                        self.frame_queues[index].append(np.zeros(self.shapes[index], dtype=np.uint8))
            else:
                time.sleep(0.01)

    def stop_streams(self):
        """Stop all streaming threads and release resources."""
        self.active = False
        for thread in self.threads:
            if thread.is_alive():
                thread.join()
        for cap in self.video_captures:
            cap.release()
        cv2.destroyAllWindows()

    def __iter__(self):
        """Return an iterator object for the streams."""
        self.frame_index = -1
        return self

    def __next__(self):
        """Retrieve the next set of frames from the streams."""
        self.frame_index += 1
        frames = []
        for queue in self.frame_queues:
            if not queue:
                raise StopIteration
            frames.append(queue.popleft() if self.buffer_mode else queue[0])
        return self.sources, frames

    def __len__(self):
        """Return the number of video sources."""
        return len(self.sources)


class ScreenshotLoader:
    """
    Manages the loading of screenshots for processing.

    This class handles the real-time capture of screenshots from a specified screen region
    and prepares them for processing. It uses the `mss` library for efficient screen capture.

    Attributes:
        monitor (dict): Monitor configuration for capturing a specific screen area.
        screen_number (int): The screen index to capture.
        image_size (int): The target image size for processing.
        frame_counter (int): Counter to track the number of frames captured.
        capture_tool (mss.mss): The screen capture tool instance.

    Methods:
        __init__: Initializes the screenshot loader with screen configuration.
        capture_screen: Captures a screenshot and returns it as a numpy array.
        __iter__: Returns an iterator object for capturing screenshots.
        __next__: Returns the next screenshot frame.
    """

    def __init__(self, monitor_config="0 0 0 1920 1080", image_size=640):
        """Initialize the screenshot loader with monitor configuration."""
        check_requirements("mss")
        import mss

        self.screen_number, self.left, self.top, self.width, self.height = map(int, monitor_config.split())
        self.image_size = image_size
        self.frame_counter = 0
        self.capture_tool = mss.mss()
        self.monitor = self._setup_monitor()

    def _setup_monitor(self):
        """Configure the monitor settings for capturing."""
        screen = self.capture_tool.monitors[self.screen_number]
        return {
            "left": screen["left"] + self.left,
            "top": screen["top"] + self.top,
            "width": self.width,
            "height": self.height
        }

    def capture_screen(self):
        """Capture a screenshot and return it as a numpy array."""
        screenshot = self.capture_tool.grab(self.monitor)
        return np.array(screenshot)[:, :, :3]  # Convert BGRA to BGR

    def __iter__(self):
        """Return an iterator object for capturing screenshots."""
        return self

    def __next__(self):
        """Capture and return the next screenshot."""
        self.frame_counter += 1
        return self.capture_screen()


class ImageLoader:
    """
    Handles loading of image and video files for batch processing.

    This class manages the loading of images and videos, processes them, and returns
    the data in a format suitable for further analysis.

    Attributes:
        files (list): List of image and video file paths.
        image_size (int): The target image size for processing.
        is_video (list): Flags to indicate whether the corresponding file is a video.
        video_capture (cv2.VideoCapture): Video capture object for handling video files.

    Methods:
        __init__: Initializes the loader with file paths and configurations.
        load_files: Loads and processes image and video files.
        __iter__: Returns an iterator object for the files.
        __next__: Returns the next image or video frame.
    """

    def __init__(self, file_path, image_size=640, frame_stride=1):
        """Initialize the image loader with file paths."""
        self.file_path = file_path
        self.image_size = image_size
        self.frame_stride = frame_stride
        self.files, self.is_video = self._load_files(file_path)
        self.video_capture = None
        self._prepare_video_loader()

    def _load_files(self, file_path):
        """Load and categorize files as images or videos."""
        if os.path.isdir(file_path):
            files = [str(p) for p in Path(file_path).glob("*.*")]
        elif os.path.isfile(file_path):
            files = [file_path]
        else:
            raise FileNotFoundError(f"File or directory not found: {file_path}")

        image_files = [f for f in files if f.split(".")[-1].lower() in IMG_FORMATS]
        video_files = [f for f in files if f.split(".")[-1].lower() in VID_FORMATS]
        is_video = [False] * len(image_files) + [True] * len(video_files)

        return image_files + video_files, is_video

    def _prepare_video_loader(self):
        """Prepare video loader if there are video files."""
        if any(self.is_video):
            self.video_capture = cv2.VideoCapture(self.files[self.is_video.index(True)])
            if not self.video_capture.isOpened():
                raise ValueError(f"Cannot open video file: {self.files[self.is_video.index(True)]}")

    def __iter__(self):
        """Return an iterator object for the files."""
        self.index = 0
        return self

    def __next__(self):
        """Return the next image or video frame."""
        if self.index >= len(self.files):
            raise StopIteration

        if self.is_video[self.index]:
            success, frame = self._get_video_frame()
            if not success:
                self.index += 1
                self.video_capture.release()
                if self.index < len(self.files):
                    self._prepare_video_loader()
                return self.__next__()
            return self.files[self.index], frame
        else:
            frame = cv2.imread(self.files[self.index])
            if frame is None:
                raise ValueError(f"Cannot read image file: {self.files[self.index]}")
            self.index += 1
            return self.files[self.index - 1], frame

    def _get_video_frame(self):
        """Retrieve a frame from the video file."""
        self.video_capture.grab()
        for _ in range(self.frame_stride - 1):
            self.video_capture.grab()
        return self.video_capture.retrieve()


class PilNumpyLoader:
    """
    Handles loading of images from PIL objects and Numpy arrays.

    This class is designed to load and process images provided as PIL Image objects or Numpy arrays,
    converting them to a standard format for further processing.

    Attributes:
        images (list): List of PIL Image objects or Numpy arrays.
        image_size (int): The target image size for processing.
        file_names (list): List of image file names for logging.

    Methods:
        __init__: Initializes the loader with images and configurations.
        process_image: Processes and converts a PIL Image or Numpy array to a standard format.
        __iter__: Returns an iterator object for the images.
        __next__: Returns the next processed image.
    """

    def __init__(self, images, image_size=640):
        """Initialize with PIL images or Numpy arrays."""
        self.images = images if isinstance(images, list) else [images]
        self.image_size = image_size
        self.file_names = [self._get_filename(i) for i in range(len(self.images))]

    def _get_filename(self, index):
        """Generate a filename for an image."""
        return getattr(self.images[index], "filename", f"image_{index}.jpg")

    def process_image(self, image):
        """Convert and prepare the image for processing."""
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError("Unsupported image format.")
        return image

    def __iter__(self):
        """Return an iterator object for the images."""
        self.index = 0
        return self

    def __next__(self):
        """Return the next processed image."""
        if self.index >= len(self.images):
            raise StopIteration

        processed_image = self.process_image(self.images[self.index])
        self.index += 1
        return self.file_names[self.index - 1], processed_image


class TensorLoader:
    """
    Handles loading of images from PyTorch tensor objects.

    This class manages loading and pre-processing of images provided as PyTorch tensors,
    ensuring they are properly formatted for further processing.

    Attributes:
        tensor (torch.Tensor): Input tensor containing the image data.
        file_names (list): List of generated file names for the tensors.

    Methods:
        __init__: Initializes the loader with a tensor.
        check_tensor: Validates and prepares the tensor for processing.
        __iter__: Returns an iterator object for the tensor data.
        __next__: Returns the next batch of processed tensor data.
    """

    def __init__(self, tensor):
        """Initialize with a PyTorch tensor."""
        self.tensor = self._check_tensor(tensor)
        self.file_names = [f"tensor_image_{i}.jpg" for i in range(tensor.shape[0])]

    def _check_tensor(self, tensor):
        """Validate and prepare the tensor for processing."""
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        if len(tensor.shape) != 4:
            raise ValueError("Tensor must have shape (B, C, H, W)")
        if tensor.max() > 1.0:
            tensor = tensor.float() / 255.0
        return tensor

    def __iter__(self):
        """Return an iterator object for the tensor data."""
        self.index = 0
        return self

    def __next__(self):
        """Return the next batch of processed tensor data."""
        if self.index >= len(self.file_names):
            raise StopIteration

        data = self.tensor[self.index]
        self.index += 1
        return self.file_names[self.index - 1], data


def merge_sources(sources):
    """Merge multiple sources into a list of images for processing."""
    images = []
    for source in sources:
        if isinstance(source, (str, Path)):
            images.append(Image.open(source))
        elif isinstance(source, (Image.Image, np.ndarray)):
            images.append(source)
        else:
            raise TypeError("Unsupported source type.")
    return images


LOADER_CLASSES = (StreamLoader, PilNumpyLoader, ImageLoader, ScreenshotLoader, TensorLoader)


def extract_youtube_url(youtube_url, prefer_pafy=True):
    """
    Extract the best video stream URL from a YouTube link.

    This function supports both pafy and yt_dlp libraries to extract video information
    from YouTube and select the highest quality video stream available.

    Args:
        youtube_url (str): URL of the YouTube video.
        prefer_pafy (bool): Whether to prefer the pafy library (True) or yt_dlp (False).

    Returns:
        str: URL of the best quality video stream.
    """
    if prefer_pafy:
        check_requirements(("pafy", "youtube_dl"))
        import pafy
        return pafy.new(youtube_url).getbestvideo(preftype="mp4").url
    else:
        check_requirements("yt-dlp")
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
        for fmt in reversed(info_dict.get("formats", [])):
            if fmt["vcodec"] != "none" and fmt["acodec"] == "none" and fmt["ext"] == "mp4":
                if fmt.get("width", 0) >= 1920 or fmt.get("height", 0) >= 1080:
                    return fmt.get("url")
