import cv2
import os
import shutil

def check_img(start,end,path,save_path,label_path,label_save_path):
    # img1 = path+start+'.jpg'
    # img2 = path+end+'.jpg'
    # img1 = cv2.imread(img1)
    # img2 = cv2.imread(img2)
    # # cv2.imshow('win1',img1)
    # cv2.imshow('win2',img2)
    # cv2.waitKey(0)
    # for img in os.listdir(path):
    #     pass
    for num in range(start,end+1):
        name = '0'+str(num)+'.jpg'
        name1 = '0'+str(num)+'.txt'
        shutil.copy(path+name,save_path+name)
        shutil.copy(label_path+name1,label_save_path+name1)

if __name__ == '__main__':
    check_img(55000,55885,'/home/sysman/gate_Sample/VOCdevkit/VOC2017/JPEGImages/',
              '/home/sysman/zlf/055000-055885/JPEGImages/',
              '/home/sysman/gate_Sample/VOCdevkit/VOC2017/labels/',
              '/home/sysman/zlf/055000-055885/labels/')