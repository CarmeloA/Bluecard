import os

img_path = '/home/sysman/gate_Sample/VOCdevkit/VOC2017/JPEGImages/'

f = open('/home/sysman/gate_Sample/VOCdevkit/VOC2017/ImageSets/train_5th_add.txt','w')
for jpg in os.listdir(img_path):
    path = img_path+jpg
    f.write(path+'\n')
f.close()