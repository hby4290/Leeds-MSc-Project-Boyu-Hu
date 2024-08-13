import os
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append((name, xmin, ymin, xmax, ymax))
    return objects, int(root.find('size/width').text), int(root.find('size/height').text)

def collect_classes(xml_files):
    class_set = set()
    for xml_file in tqdm(xml_files, desc="Collecting classes"):
        objects, _, _ = parse_xml(xml_file)
        for obj in objects:
            class_set.add(obj[0])
    return sorted(class_set)

def convert_to_yolo_format(objects, img_width, img_height, class_map):
    yolo_annotations = []
    for obj in objects:
        label = obj[0]
        xmin, ymin, xmax, ymax = obj[1], obj[2], obj[3], obj[4]
        
        # Normalizing coordinates by image width and height
        points = [
            xmin / img_width, ymin / img_height,
            xmax / img_width, ymin / img_height,
            xmax / img_width, ymax / img_height,
            xmin / img_width, ymax / img_height
        ]
        
        class_id = class_map[label]
        yolo_annotations.append(f"{class_id} " + " ".join(map(str, points)))
    return yolo_annotations

def convert_voc_to_yolo(xml_files, labels_dir, class_map):
    for xml_file in tqdm(xml_files, desc="Converting to YOLO"):
        objects, img_width, img_height = parse_xml(xml_file)
        yolo_annotations = convert_to_yolo_format(objects, img_width, img_height, class_map)
        
        yolo_file_path = os.path.join(labels_dir, Path(xml_file).stem + '.txt')
        with open(yolo_file_path, 'w') as f:
            f.write("\n".join(yolo_annotations))

def main():
    dataset_paths = {
        "train": {
            "img": "/root/yolov8/ultralytics/datasets/CASIA_NEW/train/img",
            "mask": "/root/yolov8/ultralytics/datasets/CASIA_NEW/train/mask"
        },
        "val": {
            "img": "/root/yolov8/ultralytics/datasets/CASIA_NEW/val/img",
            "mask": "/root/yolov8/ultralytics/datasets/CASIA_NEW/val/mask"
        }
    }
    
    all_xml_files = []
    for split, paths in dataset_paths.items():
        xml_files = list(Path(paths["mask"]).rglob('*.xml'))
        all_xml_files.extend(xml_files)
    
    # Step 1: Collect all classes
    class_list = collect_classes(all_xml_files)
    class_map = {cls: idx for idx, cls in enumerate(class_list)}
    
    print(f"Classes found: {class_map}")
    
    for split, paths in dataset_paths.items():
        labels_dir = os.path.join(os.path.dirname(paths["img"]), 'label')
        os.makedirs(labels_dir, exist_ok=True)
        
        xml_files = list(Path(paths["mask"]).rglob('*.xml'))
        convert_voc_to_yolo(xml_files, labels_dir, class_map)

if __name__ == "__main__":
    main()

