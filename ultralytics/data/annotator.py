from pathlib import Path
from ultralytics import SAM, YOLO

def automatic_image_annotation(image_dir, detection_model="yolov8x.pt", segmentation_model="sam_b.pt", device="", save_directory=None):
    """
    Automatically annotates images using a YOLO detection model and a SAM segmentation model.

    Args:
        image_dir (str): Directory containing images to be annotated.
        detection_model (str, optional): Path to the pre-trained YOLO detection model. Defaults to 'yolov8x.pt'.
        segmentation_model (str, optional): Path to the pre-trained SAM segmentation model. Defaults to 'sam_b.pt'.
        device (str, optional): Device to run the models on (e.g., 'cpu' or 'cuda'). Defaults to an empty string, which selects the best available option.
        save_directory (str, optional): Directory to save the annotation files. Defaults to a folder named 'auto_annotations' in the same directory as 'image_dir'.

    Example:
        ```python
        from ultralytics.data.annotator import automatic_image_annotation

        automatic_image_annotation(image_dir='ultralytics/assets', detection_model='yolov8n.pt', segmentation_model='mobile_sam.pt')
        ```
    """

    # Load the YOLO detection model and SAM segmentation model
    detection_model = YOLO(detection_model)
    segmentation_model = SAM(segmentation_model)

    # Convert image directory to Path object and set default save directory if not provided
    image_dir = Path(image_dir)
    if not save_directory:
        save_directory = image_dir.parent / f"{image_dir.stem}_auto_annotations"
    
    # Create the output directory if it doesn't exist
    Path(save_directory).mkdir(parents=True, exist_ok=True)

    # Run detection model on the images
    detection_results = detection_model(image_dir, stream=True, device=device)

    # Process each detection result
    for detection in detection_results:
        class_ids = detection.boxes.cls.int().tolist()  # List of class IDs from detection
        if class_ids:
            # Get bounding box coordinates from the detection result
            bounding_boxes = detection.boxes.xyxy
            
            # Run segmentation model using the original image and the detected bounding boxes
            segmentation_results = segmentation_model(detection.orig_img, bboxes=bounding_boxes, verbose=False, save=False, device=device)
            segmented_masks = segmentation_results[0].masks.xyn  # Normalized segmentation masks

            # Save the annotation results to a text file
            annotation_file_path = Path(save_directory) / f"{Path(detection.path).stem}.txt"
            with annotation_file_path.open("w") as annotation_file:
                for i, mask in enumerate(segmented_masks):
                    if not mask:
                        continue
                    flattened_mask = " ".join(map(str, mask.reshape(-1).tolist()))
                    annotation_file.write(f"{class_ids[i]} {flattened_mask}\n")

"""
IMPORT Path from pathlib
IMPORT SAM and YOLO from ultralytics

DEFINE FUNCTION automatic_image_annotation(image_dir, detection_model="yolov8x.pt", segmentation_model="sam_b.pt", device="", save_directory=None):
    """
    Automatically annotates images using a YOLO detection model and a SAM segmentation model.

    Args:
        image_dir: Directory containing images to be annotated.
        detection_model: Path to the pre-trained YOLO detection model (default: 'yolov8x.pt').
        segmentation_model: Path to the pre-trained SAM segmentation model (default: 'sam_b.pt').
        device: Device to run the models on ('cpu' or 'cuda', default: auto-select).
        save_directory: Directory to save the annotation files (default: 'auto_annotations' folder in the same directory as 'image_dir').

    Example:
        automatic_image_annotation(image_dir='ultralytics/assets', detection_model='yolov8n.pt', segmentation_model='mobile_sam.pt')
    """

    # LOAD the YOLO detection model
    detection_model = LOAD YOLO model with detection_model path

    # LOAD the SAM segmentation model
    segmentation_model = LOAD SAM model with segmentation_model path

    # CONVERT image_dir to a Path object
    image_dir = CONVERT image_dir to Path object

    # SET save_directory to a default value if not provided
    IF save_directory is None:
        SET save_directory to a new folder named 'auto_annotations' in the parent directory of image_dir

    # CREATE the save_directory if it doesn't exist
    CREATE save_directory if it doesn't exist

    # RUN detection model on the images in image_dir
    detection_results = RUN detection_model on image_dir with stream=True and device

    # PROCESS each detection result
    FOR each detection in detection_results:
        # EXTRACT class IDs from the detection result
        class_ids = EXTRACT class IDs from detection

        # IF there are class IDs:
        IF class_ids is not empty:
            # GET bounding box coordinates from the detection result
            bounding_boxes = GET bounding box coordinates from detection

            # RUN segmentation model using the original image and bounding boxes
            segmentation_results = RUN segmentation_model with original image and bounding boxes

            # EXTRACT normalized segmentation masks
            segmented_masks = EXTRACT normalized masks from segmentation_results

            # CREATE the annotation file path
            annotation_file_path = CREATE file path in save_directory with the same name as the image

            # OPEN the annotation file for writing
            OPEN annotation_file_path for writing

            # WRITE class ID and mask coordinates to the annotation file
            FOR each mask in segmented_masks:
                IF mask is not empty:
                    CONVERT mask coordinates to a string and write to the file with the corresponding class ID

            # CLOSE the annotation file
            CLOSE the annotation file

"""
