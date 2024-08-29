import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from ultralytics.utils import TQDM


class QuickSegPrompt:
    """
    QuickSegPrompt is a utility class designed for image annotation, segmentation, and visualization tasks.

    Attributes:
        device (str): Specifies the computing device ('cuda' or 'cpu').
        results: Stores the detection or segmentation results.
        source: Source of the image (can be an image object or a path to the image).
        clip: The CLIP model used for text-image similarity processing.
    """

    def __init__(self, source, results, device="cuda") -> None:
        """
        Initializes the QuickSegPrompt with the provided image source, results, and computing device.
        Sets up the CLIP model for text-image similarity calculations.
        
        Args:
            source: The source image or path to the image.
            results: The results from the object detection or segmentation process.
            device: The device to run computations on ('cuda' or 'cpu').
        """
        self.device = device
        self.results = results
        self.source = source

        # Import CLIP model for text-image similarity tasks
        try:
            import clip
        except ImportError:
            from ultralytics.utils.checks import check_requirements
            check_requirements("git+https://github.com/openai/CLIP.git")
            import clip
        self.clip = clip

    @staticmethod
    def _segment_image(image, bbox):
        """
        Segments a region of the image based on the provided bounding box coordinates.

        Args:
            image: The input image to segment.
            bbox: The bounding box coordinates [x1, y1, x2, y2] to define the region of interest.

        Returns:
            Image: A segmented image where only the area within the bounding box is visible.
        """
        image_array = np.array(image)
        segmented_image_array = np.zeros_like(image_array)
        x1, y1, x2, y2 = bbox
        segmented_image_array[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]
        segmented_image = Image.fromarray(segmented_image_array)
        background_image = Image.new("RGB", image.size, (255, 255, 255))
        
        # Create a transparency mask for the segmented area
        transparency_mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
        transparency_mask[y1:y2, x1:x2] = 255
        transparency_mask_image = Image.fromarray(transparency_mask, mode="L")
        
        # Paste the segmented image onto the background image using the transparency mask
        background_image.paste(segmented_image, mask=transparency_mask_image)
        return background_image

    @staticmethod
    def _format_results(result, filter_threshold=0):
        """
        Formats detection results into a list of annotations.

        Args:
            result: The detection result object containing masks and bounding boxes.
            filter_threshold: Minimum area required for an annotation to be included.

        Returns:
            list: A list of dictionaries, each containing annotation details like ID, segmentation mask, bounding box, score, and area.
        """
        annotations = []
        n = len(result.masks.data) if result.masks is not None else 0
        for i in range(n):
            mask = result.masks.data[i] == 1.0
            if torch.sum(mask) >= filter_threshold:
                annotation = {
                    "id": i,
                    "segmentation": mask.cpu().numpy(),
                    "bbox": result.boxes.data[i],
                    "score": result.boxes.conf[i],
                }
                annotation["area"] = annotation["segmentation"].sum()
                annotations.append(annotation)
        return annotations

    @staticmethod
    def _get_bbox_from_mask(mask):
        """
        Extracts the bounding box coordinates from a segmentation mask.

        Args:
            mask: The segmentation mask from which to extract the bounding box.

        Returns:
            list: Bounding box coordinates [x1, y1, x2, y2].
        """
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x1, y1, w, h = cv2.boundingRect(contours[0])
        x2, y2 = x1 + w, y1 + h
        
        # Adjust bounding box if multiple contours are detected
        if len(contours) > 1:
            for contour in contours:
                x_temp, y_temp, w_temp, h_temp = cv2.boundingRect(contour)
                x1 = min(x1, x_temp)
                y1 = min(y1, y_temp)
                x2 = max(x2, x_temp + w_temp)
                y2 = max(y2, y_temp + h_temp)
        
        return [x1, y1, x2, y2]

    def plot(self, annotations, output_path, bbox=None, points=None, point_labels=None, random_mask_color=True, enhance_quality=True, retina_mode=False, draw_contours=True):
        """
        Plots the annotations, bounding boxes, and points on images and saves the output.

        Args:
            annotations (list): List of annotations to plot.
            output_path (str or Path): Directory to save the plotted images.
            bbox (list, optional): Coordinates of the bounding box to plot. Defaults to None.
            points (list, optional): Points to plot on the image. Defaults to None.
            point_labels (list, optional): Labels for the points to plot. Defaults to None.
            random_mask_color (bool, optional): If True, uses random colors for the masks. Defaults to True.
            enhance_quality (bool, optional): If True, applies morphological transformations for better mask quality. Defaults to True.
            retina_mode (bool, optional): If True, uses retina mode for mask display. Defaults to False.
            draw_contours (bool, optional): If True, draws contours around masks. Defaults to True.
        """
        pbar = TQDM(annotations, total=len(annotations))
        for annotation in pbar:
            result_name = os.path.basename(annotation.path)
            image = annotation.orig_img[..., ::-1]  # Convert BGR to RGB
            original_height, original_width = annotation.orig_shape

            # Set up the plot
            plt.figure(figsize=(original_width / 100, original_height / 100))
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.imshow(image)

            if annotation.masks is not None:
                masks = annotation.masks.data
                if enhance_quality:
                    masks = np.array(masks.cpu())
                    for i, mask in enumerate(masks):
                        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                        masks[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))

                self.display_mask(
                    masks,
                    plt.gca(),
                    random_color=random_mask_color,
                    bbox=bbox,
                    points=points,
                    point_labels=point_labels,
                    retina_mode=retina_mode,
                    target_height=original_height,
                    target_width=original_width,
                )

                if draw_contours:
                    contour_list = []
                    temp_mask = np.zeros((original_height, original_width, 1))
                    for mask in masks:
                        mask = mask.astype(np.uint8)
                        if not retina_mode:
                            mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        contour_list.extend(iter(contours))
                    cv2.drawContours(temp_mask, contour_list, -1, (255, 255, 255), 2)
                    color = np.array([0 / 255, 0 / 255, 1.0, 0.8])
                    contour_mask = temp_mask / 255 * color.reshape(1, 1, -1)
                    plt.imshow(contour_mask)

            # Save the plot
            save_path = Path(output_path) / result_name
            save_path.parent.mkdir(exist_ok=True, parents=True)
            plt.axis("off")
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0, transparent=True)
            plt.close()
            pbar.set_description(f"Saving {result_name} to {save_path}")

    @staticmethod
    def display_mask(mask_data, ax, random_color=False, bbox=None, points=None, point_labels=None, retina_mode=True, target_height=960, target_width=960):
        """
        Displays mask annotations on a matplotlib axis.

        Args:
            mask_data (array-like): The mask data to be displayed.
            ax (matplotlib.axes.Axes): The matplotlib axis to plot on.
            random_color (bool, optional): If True, random colors are used for the masks. Defaults to False.
            bbox (list, optional): Bounding box coordinates to display. Defaults to None.
            points (list, optional): Points to plot on the image. Defaults to None.
            point_labels (list, optional): Labels for the points. Defaults to None.
            retina_mode (bool, optional): If True, applies retina mode for mask display. Defaults to True.
            target_height (int, optional): Target height for resizing. Defaults to 960.
            target_width (int, optional): Target width for resizing. Defaults to 960.
        """
        num_masks, mask_height, mask_width = mask_data.shape

        # Sort masks by area
        areas = np.sum(mask_data, axis=(1, 2))
        mask_data = mask_data[np.argsort(areas)]

        # Set up mask visualization
        index_mask = (mask_data != 0).argmax(axis=0)
        color = np.random.random((num_masks, 1, 1, 3)) if random_color else np.ones((num_masks, 1, 1, 3)) * np.array([30 / 255, 144 / 255, 1.0])
        transparency = np.ones((num_masks, 1, 1, 1)) * 0.6
        visual_mask = np.concatenate([color, transparency], axis=-1)
        combined_mask_image = np.expand_dims(mask_data, -1) * visual_mask

        display_image = np.zeros((mask_height, mask_width, 4))
        height_indices, width_indices = np.meshgrid(np.arange(mask_height), np.arange(mask_width), indexing="ij")
        indices = (index_mask[height_indices, width_indices], height_indices, width_indices, slice(None))

        display_image[height_indices, width_indices, :] = combined_mask_image[indices]

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="b", linewidth=1))

        if points is not None:
            plt.scatter(
                [point[0] for i, point in enumerate(points) if point_labels[i] == 1],
                [point[1] for i, point in enumerate(points) if point_labels[i] == 1],
                s=20,
                c="y",
            )
            plt.scatter(
                [point[0] for i, point in enumerate(points) if point_labels[i] == 0],
                [point[1] for i, point in enumerate(points) if point_labels[i] == 0],
                s=20,
                c="m",
            )

        if not retina_mode:
            display_image = cv2.resize(display_image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        ax.imshow(display_image)

    @torch.no_grad()
    def retrieve_similarities(self, model, preprocess, elements, query_text, device):
        """
        Processes images and text with the CLIP model, calculates similarity scores, and returns them.

        Args:
            model: The CLIP model to use.
            preprocess: Preprocessing function for the images.
            elements: List of images to process.
            query_text (str): The text query to compare against.
            device: The device to run the computations on.

        Returns:
            torch.Tensor: Similarity scores between the images and the text.
        """
        preprocessed_images = [preprocess(image).to(device) for image in elements]
        tokenized_text = self.clip.tokenize([query_text]).to(device)
        stacked_images = torch.stack(preprocessed_images)
        image_features = model.encode_image(stacked_images)
        text_features = model.encode_text(tokenized_text)
        
        # Normalize the features and calculate similarity scores
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity_scores = 100.0 * image_features @ text_features.T
        return similarity_scores[:, 0].softmax(dim=0)

    def _crop_image(self, formatted_results):
        """
        Crops the image based on the provided annotations.

        Args:
            formatted_results: The formatted results containing segmentation masks and bounding boxes.

        Returns:
            tuple: Cropped images, their bounding boxes, non-cropped images, filtered IDs, and the annotations.
        """
        if os.path.isdir(self.source):
            raise ValueError(f"'{self.source}' is a directory, not a valid source for cropping.")
        
        image = Image.fromarray(cv2.cvtColor(self.results[0].orig_img, cv2.COLOR_BGR2RGB))
        original_width, original_height = image.size
        annotations = formatted_results
        mask_height, mask_width = annotations[0]["segmentation"].shape
        
        if original_width != mask_width or original_height != mask_height:
            image = image.resize((mask_width, mask_height))

        cropped_boxes = []
        cropped_images = []
        filtered_ids = []
        non_cropped = []

        for idx, mask in enumerate(annotations):
            if np.sum(mask["segmentation"]) <= 100:
                filtered_ids.append(idx)
                continue
            bbox = self._get_bbox_from_mask(mask["segmentation"])
            cropped_boxes.append(self._segment_image(image, bbox))
            cropped_images.append(bbox)

        return cropped_boxes, cropped_images, non_cropped, filtered_ids, annotations

    def apply_box_prompt(self, bbox):
        """
        Adjusts bounding boxes based on detected masks and computes IoU.

        Args:
            bbox (list): The bounding box coordinates to be applied.

        Returns:
            Updated segmentation results after applying the bounding box prompt.
        """
        if self.results[0].masks is not None:
            assert bbox[2] != 0 and bbox[3] != 0
            if os.path.isdir(self.source):
                raise ValueError(f"'{self.source}' is a directory, not a valid source for this function.")
            
            masks = self.results[0].masks.data
            target_height, target_width = self.results[0].orig_shape
            mask_height, mask_width = masks.shape[1], masks.shape[2]

            if mask_height != target_height or mask_width != target_width:
                bbox = [
                    int(bbox[0] * mask_width / target_width),
                    int(bbox[1] * mask_height / target_height),
                    int(bbox[2] * mask_width / target_width),
                    int(bbox[3] * mask_height / target_height),
                ]

            bbox[0] = max(round(bbox[0]), 0)
            bbox[1] = max(round(bbox[1]), 0)
            bbox[2] = min(round(bbox[2]), mask_width)
            bbox[3] = min(round(bbox[3]), mask_height)

            bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
            masks_area = torch.sum(masks[:, bbox[1]:bbox[3], bbox[0]:bbox[2]], dim=(1, 2))
            original_masks_area = torch.sum(masks, dim=(1, 2))

            union_area = bbox_area + original_masks_area - masks_area
            iou = masks_area / union_area
            max_iou_index = torch.argmax(iou)

            self.results[0].masks.data = torch.tensor(np.array([masks[max_iou_index].cpu().numpy()]))
        return self.results

    def apply_point_prompt(self, points, point_labels):
        """
        Adjusts the segmentation masks based on user-provided points and returns the updated results.

        Args:
            points (list): List of points to apply the prompt to.
            point_labels (list): Labels corresponding to the points.

        Returns:
            Updated segmentation results after applying the point prompt.
        """
        if self.results[0].masks is not None:
            if os.path.isdir(self.source):
                raise ValueError(f"'{self.source}' is a directory, not a valid source for this function.")
            
            masks = self._format_results(self.results[0], 0)
            target_height, target_width = self.results[0].orig_shape
            mask_height, mask_width = masks[0]["segmentation"].shape

            if mask_height != target_height or mask_width != target_width:
                points = [[int(point[0] * mask_width / target_width), int(point[1] * mask_height / target_height)] for point in points]
            
            combined_mask = np.zeros((mask_height, mask_width))

            for annotation in masks:
                mask = annotation["segmentation"] if isinstance(annotation, dict) else annotation
                for i, point in enumerate(points):
                    if mask[point[1], point[0]] == 1 and point_labels[i] == 1:
                        combined_mask += mask
                    if mask[point[1], point[0]] == 1 and point_labels[i] == 0:
                        combined_mask -= mask

            combined_mask = combined_mask >= 1
            self.results[0].masks.data = torch.tensor(np.array([combined_mask]))
        return self.results

    def apply_text_prompt(self, text):
        """
        Processes a text prompt to select the most relevant segmentation mask and updates the results.

        Args:
            text (str): The text prompt used for mask selection.

        Returns:
            Updated segmentation results after applying the text prompt.
        """
        if self.results[0].masks is not None:
            formatted_results = self._format_results(self.results[0], 0)
            cropped_boxes, cropped_images, non_cropped, filter_ids, annotations = self._crop_image(formatted_results)

            clip_model, preprocess = self.clip.load("ViT-B/32", device=self.device)
            scores = self.retrieve_similarities(clip_model, preprocess, cropped_boxes, text, device=self.device)
            max_index = scores.argsort()[-1]
            max_index += sum(np.array(filter_ids) <= int(max_index))

            self.results[0].masks.data = torch.tensor(np.array([annotations[max_index]["segmentation"]]))
        return self.results

    def retrieve_all_results(self):
        """
        Returns the processed results without any further modifications.

        Returns:
            The current segmentation results.
        """
        return self.results

"""
IMPORT necessary libraries and modules

DECLARE CLASS QuickSegPrompt:
    """
    A utility class for image annotation, segmentation, and visualization tasks.
    """

    METHOD __init__(self, source, results, device="cuda"):
        """
        Initialize the QuickSegPrompt with the image source, results, and computing device.

        Args:
            source: The image source (image object or path).
            results: The detection or segmentation results.
            device: The computing device to use ('cuda' or 'cpu').
        """
        SET self.device to device
        SET self.results to results
        SET self.source to source

        TRY to import CLIP model for text-image processing
        IF ImportError:
            INSTALL required CLIP package
            IMPORT CLIP model
        SET self.clip to CLIP model

    STATIC METHOD _segment_image(image, bbox):
        """
        Segment a region of the image based on bounding box coordinates.

        Args:
            image: The input image to segment.
            bbox: The bounding box coordinates [x1, y1, x2, y2].

        Returns:
            Segmented image with only the region inside the bounding box visible.
        """
        CONVERT image to numpy array
        CREATE empty array for segmented image
        COPY region within bbox to segmented image
        CREATE transparency mask for the segmented region
        COMBINE segmented image and transparency mask
        RETURN the combined image

    STATIC METHOD _format_results(result, filter_threshold=0):
        """
        Format detection results into a list of annotations.

        Args:
            result: The detection result containing masks and bounding boxes.
            filter_threshold: Minimum area required to include an annotation.

        Returns:
            List of annotations with ID, segmentation mask, bounding box, score, and area.
        """
        INITIALIZE empty list for annotations
        FOR each mask in result:
            IF mask area >= filter_threshold:
                CREATE annotation dictionary with ID, segmentation, bbox, score, and area
                ADD annotation to the list
        RETURN annotations list

    STATIC METHOD _get_bbox_from_mask(mask):
        """
        Extract bounding box coordinates from a segmentation mask.

        Args:
            mask: The segmentation mask to process.

        Returns:
            Bounding box coordinates [x1, y1, x2, y2].
        """
        FIND contours in the mask
        COMPUTE bounding box around the largest contour
        ADJUST bounding box if there are multiple contours
        RETURN bounding box coordinates

    METHOD plot(annotations, output_path, bbox=None, points=None, point_labels=None, random_mask_color=True, enhance_quality=True, retina_mode=False, draw_contours=True):
        """
        Plot annotations, bounding boxes, and points on images and save the output.

        Args:
            annotations: List of annotations to plot.
            output_path: Directory to save the plots.
            bbox: Bounding box coordinates to plot (optional).
            points: Points to plot on the image (optional).
            point_labels: Labels for the points (optional).
            random_mask_color: Use random colors for masks (default: True).
            enhance_quality: Apply morphological transformations to improve mask quality (default: True).
            retina_mode: Use retina mode for displaying masks (default: False).
            draw_contours: Draw contours around masks (default: True).
        """
        FOR each annotation in annotations:
            SET UP the plot with the image dimensions
            DISPLAY the image on the plot
            IF masks are present:
                PROCESS and display the masks on the plot
                IF contours should be drawn:
                    FIND and draw contours on the masks
            SAVE the plot to the output directory

    STATIC METHOD display_mask(mask_data, ax, random_color=False, bbox=None, points=None, point_labels=None, retina_mode=True, target_height=960, target_width=960):
        """
        Display mask annotations on a matplotlib axis.

        Args:
            mask_data: Mask data to display.
            ax: Matplotlib axis to plot on.
            random_color: Use random colors for the masks (default: False).
            bbox: Bounding box coordinates to display (optional).
            points: Points to plot on the image (optional).
            point_labels: Labels for the points (optional).
            retina_mode: Use retina mode for mask display (default: True).
            target_height: Target height for resizing (default: 960).
            target_width: Target width for resizing (default: 960).
        """
        SORT masks by area
        SET UP visualization with color and transparency
        DISPLAY the combined masks on the plot
        IF bounding box is provided:
            DRAW the bounding box on the plot
        IF points are provided:
            PLOT the points on the image
        IF retina mode is disabled:
            RESIZE the mask display to target dimensions
        SHOW the final image with masks on the axis

    METHOD retrieve_similarities(model, preprocess, elements, query_text, device):
        """
        Calculate similarity scores between images and a text query using the CLIP model.

        Args:
            model: The CLIP model for processing.
            preprocess: Preprocessing function for images.
            elements: List of images to process.
            query_text: The text query for comparison.
            device: The device to run computations on.

        Returns:
            Similarity scores between the images and the text query.
        """
        PREPROCESS images and encode them with the CLIP model
        TOKENIZE the text query and encode it with the CLIP model
        CALCULATE similarity scores between image and text features
        RETURN softmax-normalized similarity scores

    METHOD _crop_image(formatted_results):
        """
        Crop the image based on the provided annotations.

        Args:
            formatted_results: Annotations with segmentation masks and bounding boxes.

        Returns:
            Cropped images, their bounding boxes, non-cropped images, filtered IDs, and the annotations.
        """
        IF source is a directory:
            RAISE an error
        LOAD the image and resize if necessary
        INITIALIZE lists for cropped boxes, images, and other data
        FOR each annotation:
            IF mask area is too small, filter it out
            EXTRACT bounding box from mask
            SEGMENT the image using the bounding box and store the result
        RETURN the lists of cropped boxes, images, and other data

    METHOD apply_box_prompt(bbox):
        """
        Adjust bounding boxes based on detected masks and compute IoU.

        Args:
            bbox: Bounding box coordinates to apply.

        Returns:
            Updated segmentation results.
        """
        IF masks are present:
            ADJUST bounding box dimensions to match mask size
            COMPUTE IoU between bounding box and masks
            SELECT the mask with the highest IoU
            UPDATE the results with the selected mask
        RETURN the updated results

    METHOD apply_point_prompt(points, point_labels):
        """
        Adjust segmentation masks based on user-provided points.

        Args:
            points: List of points to apply the prompt to.
            point_labels: Labels corresponding to the points.

        Returns:
            Updated segmentation results.
        """
        IF masks are present:
            ADJUST point coordinates to match mask size
            CREATE a combined mask based on the points and labels
            UPDATE the results with the combined mask
        RETURN the updated results

    METHOD apply_text_prompt(text):
        """
        Process a text prompt and select the most relevant segmentation mask.

        Args:
            text: The text prompt to use.

        Returns:
            Updated segmentation results.
        """
        IF masks are present:
            FORMAT the results and crop the images
            LOAD the CLIP model and preprocess the images
            CALCULATE similarity scores between cropped images and text
            SELECT the mask with the highest similarity score
            UPDATE the results with the selected mask
        RETURN the updated results

    METHOD retrieve_all_results():
        """
        Retrieve and return the processed segmentation results.

        Returns:
            The current segmentation results.
        """
        RETURN the results

"""
