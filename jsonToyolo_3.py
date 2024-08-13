import os
import glob
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

def make_dirs():
    """创建所需的目录"""
    train_labels_path = '/root/yolov8/ultralytics/datasets/CASIA/labels/train'
    val_labels_path = '/root/yolov8/ultralytics/datasets/CASIA/labels/val'
    os.makedirs(train_labels_path, exist_ok=True)
    os.makedirs(val_labels_path, exist_ok=True)
    return train_labels_path, val_labels_path

def exif_size(img):
    """获取图片的宽度和高度"""
    return img.size

def convert_polygon_to_yolo(points, img_width, img_height):
    """将多边形转换为YOLO格式的边界框"""
    points = np.array(points)
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])

    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    return x_center, y_center, width, height

def convert_infolks_json(json_files, labels_dir, images_dir):
    """将Infolks JSON格式转换为YOLO格式"""
    for json_file in tqdm(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img_path = os.path.join(images_dir, data['imagePath'])
        img = Image.open(img_path)
        img_width, img_height = exif_size(img)
        
        yolo_annotations = []
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']
            x_center, y_center, width, height = convert_polygon_to_yolo(points, img_width, img_height)
            
            # 使用标签字典将标签名称转换为类别ID，这里假设标签为'0'表示'ship'，'1'表示'submarine'
            class_id = 0 if label == 'ship' else 1
            yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
        
        # 保存为YOLO格式的标签文件
        yolo_file_path = os.path.join(labels_dir, Path(json_file).stem + '.txt')
        with open(yolo_file_path, 'w') as f:
            f.write("\n".join(yolo_annotations))

def main():
    train_labels_path, val_labels_path = make_dirs()
    
    train_json_files = glob.glob('/root/yolov8/ultralytics/datasets/CASIA/train/*.json')
    val_json_files = glob.glob('/root/yolov8/ultralytics/datasets/CASIA/val/*.json')
    
    convert_infolks_json(train_json_files, train_labels_path, '/root/yolov8/ultralytics/datasets/CASIA/train/image')
    convert_infolks_json(val_json_files, val_labels_path, '/root/yolov8/ultralytics/datasets/CASIA/val/image')

if __name__ == "__main__":
    main()

