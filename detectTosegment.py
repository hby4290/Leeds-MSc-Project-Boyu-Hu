import os
import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from pycocotools import mask as maskUtils

def create_yolo_instance_labels(image_dir, mask_dir, output_dir, img_shape):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for mask_filename in os.listdir(mask_dir):
        if mask_filename.endswith('.png'):
            mask_path = os.path.join(mask_dir, mask_filename)
            label_path = os.path.join(output_dir, os.path.splitext(mask_filename)[0] + '.txt')
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            with open(label_path, 'w') as f:
                for class_id in np.unique(mask_img):
                    if class_id == 0:  # Skip background
                        continue
                    
                    # Create binary mask for the current class
                    binary_mask = (mask_img == class_id).astype(np.uint8)
                    
                    # Find contours
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        x_center = (x + w / 2) / img_shape[1]
                        y_center = (y + h / 2) / img_shape[0]
                        width = w / img_shape[1]
                        height = h / img_shape[0]
                        
                        # Get polygon
                        polygon = contour.flatten().tolist()
                        polygon = [p / img_shape[i % 2] for i, p in enumerate(polygon)]
                        
                        # Write label
                        f.write(f"{class_id} {x_center} {y_center} {width} {height} {' '.join(map(str, polygon))}\n")

# 定义路径
image_dir = '/root/yolov8/ultralytics/SegmentDatasets/images/train'
mask_dir = '/root/yolov8/ultralytics/SegmentDatasets/labels/train'
output_dir = '/root/yolov8/ultralytics/SegmentDatasets/labels/train_yolo'
img_shape = (640, 640)  # Replace with the actual image shape

# 处理训练标签
create_yolo_instance_labels(image_dir, mask_dir, output_dir, img_shape)

# 处理验证标签
mask_dir = '/root/yolov8/ultralytics/SegmentDatasets/labels/val'
output_dir = '/root/yolov8/ultralytics/SegmentDatasets/labels/val_yolo'
create_yolo_instance_labels(image_dir, mask_dir, output_dir, img_shape)

