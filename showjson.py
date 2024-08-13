# -*- coding: utf-8 -*-
# -----------------------------------------------------
# Time :  2023/2/27 11:28
# Auth :  Written by zuofengyuan
# File :  统计coco信息.py
# Copyright (c) Shenyang Pedlin Technolofy Co., Ltd.
# -----------------------------------------------------
"""
 Description: TODO
"""
from pycocotools.coco import COCO

# 文件路径
dataDir = r'l/：'
dataType = 'train2017' #val2017
annFile = '{}/instances_{}.json'.format(dataDir, dataType)

# initialize COCO api for instance annotations
coco_train = COCO(annFile)

# display COCO categories and supercategories
# 显示所有类别
cats = coco_train.loadCats(coco_train.getCatIds())
cat_nms = [cat['name'] for cat in cats]
print('COCO categories:\n{}'.format('\n'.join(cat_nms)) + '\n')
# 统计单个类别的图片数量与标注数量
for cat_name in cat_nms:
    catId = coco_train.getCatIds(catNms=cat_name)
    if cat_name == "person":
        print(catId)
        imgId = coco_train.getImgIds(catIds=catId)
        annId = coco_train.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=False)
        print("{:<15} {:<6d}     {:<10d}\n".format(cat_name, len(imgId), len(annId)))
    if cat_name == "motorcycle":
        print(catId)
        imgId = coco_train.getImgIds(catIds=catId)
        annId = coco_train.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=False)
        print("{:<15} {:<6d}     {:<10d}\n".format(cat_name, len(imgId), len(annId)))
# 统计全部的类别及全部的图片数量和标注数量
print("NUM_categories： " + str(len(coco_train.dataset['categories'])))
print("NUM_images: " + str(len(coco_train.dataset['images'])))
print("NUM_annotations: " + str(len(coco_train.dataset['annotations'])))

