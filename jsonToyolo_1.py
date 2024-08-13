import os
import glob
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

def make_dirs():
    """创建所需的目录"""
    path = '/root/yolov8/ultralytics/datasets/CASIA/labels'
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def exif_size(img):
    """获取图片的宽度和高度"""
    return img.size  # (width, height)

def convert_infolks_json(name, files, img_path):
    """将INFOLKS JSON注释转换为YOLO格式标签"""
    path = make_dirs()

    # 导入json
    data = []
    for file in glob.glob(files):
        with open(file) as f:
            jdata = json.load(f)
            jdata["json_file"] = file
            data.append(jdata)

    # 写入图片和形状
    name = path + os.sep + name
    file_name, wh, cat = [], [], []
    for x in tqdm(data, desc="处理文件和形状"):
        json_stem = Path(x["json_file"]).stem
        img_file = glob.glob(img_path + json_stem + ".png")

        if not img_file:
            print(f"找不到与 {json_stem} 对应的图片文件")
            continue

        f = img_file[0]
        file_name.append(f)
        wh.append(exif_size(Image.open(f)))  # (width, height)
        cat.extend(a["classTitle"].lower() for a in x["output"]["objects"])  # categories

        # 文件名
        with open(name + ".txt", "a") as file:
            file.write("%s\n" % f)

    # 写入 *.names 文件
    names = sorted(np.unique(cat))
    with open(name + ".names", "a") as file:
        [file.write("%s\n" % a) for a in names]

    # 写入标签文件
    for i, x in enumerate(tqdm(data, desc="处理注释")):
        if i >= len(file_name):
            continue

        label_name = Path(file_name[i]).stem + ".txt"

        with open(path + "/labels/" + label_name, "a") as file:
            for a in x["output"]["objects"]:
                category_id = names.index(a["classTitle"].lower())

                # INFOLKS 的边界框格式是 [x-min, y-min, x-max, y-max]
                box = np.array(a["points"]["exterior"], dtype=np.float32).ravel()
                box[[0, 2]] /= wh[i][0]  # 归一化 x
                box[[1, 3]] /= wh[i][1]  # 归一化 y
                box = [box[[0, 2]].mean(), box[[1, 3]].mean(), box[2] - box[0], box[3] - box[1]]  # 转换为 xywh
                if (box[2] > 0.0) and (box[3] > 0.0):  # 如果 w > 0 和 h > 0
                    file.write("%g %.6f %.6f %.6f %.6f\n" % (category_id, *box))

    # 分割数据为训练集、测试集和验证集
    split_files(name, file_name)
    write_data_data(name + ".data", nc=len(names))
    print(f"完成。输出保存到 {os.getcwd() + os.sep + path}")

def split_files(name, file_name):
    """将文件分割为训练集、测试集和验证集"""
    # 示例代码，根据需求调整
    with open(name + "_train.txt", "w") as train_file, \
         open(name + "_val.txt", "w") as val_file:
        for i, f in enumerate(file_name):
            if i % 5 == 0:
                val_file.write("%s\n" % f)
            else:
                train_file.write("%s\n" % f)

def write_data_data(data_file, nc):
    """写入.data文件"""
    with open(data_file, "w") as f:
        f.write(f"classes={nc}\n")
        f.write(f"train={os.path.splitext(data_file)[0]}_train.txt\n")
        f.write(f"val={os.path.splitext(data_file)[0]}_val.txt\n")
        f.write(f"names={os.path.splitext(data_file)[0]}.names\n")

if __name__ == "__main__":
    convert_infolks_json(
        name="casia",
        files="/root/yolov8/ultralytics/datasets/CASIA/train/mask/*.json",
        img_path="/root/yolov8/ultralytics/datasets/CASIA/train/image/"
    )

