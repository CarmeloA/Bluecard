import torch

import cv2 as cv
import os
import shutil
#
# path = '/home/sysman/zlf/yolov5-master/new-images-bak/'
# new_path = '/home/sysman/zlf/yolov5-master/new-images/'
#
# for name in os.listdir(path):
#     img = path + name
#     print(img)
#     img = cv.imread(img)
#     img = cv.resize(img,(608,342))
#     cv.imwrite(new_path+name,img)

# f = open('/home/sysman/gate_Sample/train.txt','r')
# lab = '/home/sysman/gate_Sample/VOCdevkit/VOC2017/labels/'
# f1 = open('new_train.txt','r')
# lines = f1.readlines()
# for i,line in enumerate(lines):
#     line = line.strip()
#     name = line.split('/')[-1].replace('.jpg','.txt')
#     # img = cv.imread(line)
#     # img = cv.resize(img, (608, 608))
#     # cv.imwrite(new_path+name,img)
#     shutil.copy(lab+name,'/home/sysman/zlf/yolov5-master/labels/'+name)
#
# f = open('train_add.txt','w')
# path = '/home/sysman/gate_Sample/VOCdevkit/VOC2017/JPEGImages/'
# for img in os.listdir(path):
#     img_path = path + img
#     f.write(img_path+'\n')
# f.close()
from models.experimental import attempt_load
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# model = torch.load('model_350_750_0.08016.pt')
model = attempt_load('/home/data/yolov5_pt/runs/exp25/weights/ckpt_model_599_800_0.07873.pt')
model = model.state_dict()
torch.save(model,'20201229_exp25_599_800.pth',_use_new_zipfile_serialization=False)



