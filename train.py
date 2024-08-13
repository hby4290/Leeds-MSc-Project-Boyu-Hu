import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('YOLOv8.2/ultralytics/yolov8-MLLA.yaml')
    # model.load('yolov8n.pt') 
    model.train(data='root/DIOR.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=False, 
                batch=16,
                close_mosaic=0,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                # resume='runs/train/exp21/weights/last.pt',
                amp=True, 
                project='runs/train',
                name='exp',
                )
