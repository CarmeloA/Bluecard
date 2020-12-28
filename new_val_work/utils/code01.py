import cv2
import os
import shutil

imgs_path = '/home/data/TestSampleLib/新能源车牌/'
for f in os.listdir(imgs_path):
    if f.endswith('.jpg'):
        shutil.move(imgs_path+f,imgs_path+'JPEGImgaes/'+f)


# img = cv2.imread('/home/sysman/zlf/055000-055885/JPEGImages/055812.jpg')
# height = img.shape[0]
# width = img.shape[1]
# f = open('/home/sysman/zlf/055000-055885/labels/055812.txt','r')
# lines = f.readlines()
# for line in lines:
#     line = line.split(' ')
#     x = float(line[1])
#     y = float(line[2])
#     w = float(line[3])
#     h = float(line[4])
#
#     x1 = int(float((x-w/2))*width)
#     y1 = int(float((y-h/2))*height)
#     x2 = int(float((x+w/2))*width)
#     y2 = int(float((y+h/2))*height)
#
#     cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
# img = cv2.resize(img,(width//2,height//2))
# cv2.imshow('win',img)
# cv2.waitKey(0)



