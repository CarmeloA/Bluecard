import os
import cv2
path = '/home/sysman/gate_Sample/VOCdevkit/VOC2017/JPEGImages/'
img = cv2.imread('/home/sysman/zlf/3rd_add_016299.jpg')
f = open('/home/sysman/gate_Sample/VOCdevkit/VOC2017/labels/582381.txt','r')
lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    c = float(line[0])
    x = float(line[1])
    y = float(line[2])
    w = float(line[3])
    h = float(line[4])
    x1 = int((x-w/2)*1920)
    y1 = int((y-h/2)*1080)
    x2 = int((x+w/2)*1920)
    y2 = int((y+h/2)*1080)
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
cv2.imshow('win',img)
cv2.waitKey(0)
# shape = img.shape
#
# for i in os.listdir(path):
#     img1 = path+i
#     img1 = cv2.imread(img1)
#     shape1 = img1.shape
#     if shape==shape1:
#         c=(img==img1)
#         d=c.all()
#         if d:
#             print(i)

