import cv2
import os
from skimage import io
l = []
path = '/home/sysman/gate_Sample/VOCdevkit/VOC2017/test_samples/JPEGImages/'
for i,name in enumerate(os.listdir(path)):
    try:
        print('%s|%s'%(i,len(os.listdir(path))-1))
        io.imread(path+name)
    except Exception as e:
        l.append(name)
        print(path+name)
        print(e)

print(l)

# for i in range(585475,599632):
#     print(i)
#     os.remove('/home/sysman/gate_Sample/VOCdevkit/VOC2017/JPEGImages/%s.jpg'%i)
#     os.remove('/home/sysman/gate_Sample/VOCdevkit/VOC2017/labels/%s.txt'%i)
