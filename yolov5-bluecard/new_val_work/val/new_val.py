from new_val_work.val.new_test1 import build_target,compared,count_target,\
    draw_bbox,check_compare_res_and_save,check_pred

from utils.general import non_max_suppression,scale_coords
import cv2
import torch
import time
import numpy as np
import os
from torchvision import transforms as tf
from new_val_work.val.init_model import ModelV


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def val(model,imgs_path,issave=True,save_path='',result_txt=''):
    # print('=============start val=============')
    t0 = time_synchronized()
    l = []
    res_txt = open(result_txt,'w')
    f = open(imgs_path,'r')
    lines = f.readlines()
    for ind,img_path in enumerate(lines):
        img_path = img_path.strip()
        boxes = build_target(img_path)
        img = cv2.imread(img_path)
        image = img.copy()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(320,180))
        img = cv2.copyMakeBorder(img, 6, 6, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = img / (255., 255., 255.)
        img = img.transpose((2,0,1))
        img = img.astype(np.float32)
        # 计算图片亮度,调整conf
        m = img.mean()*255.
        # print('mean:',m)


        pred = model.detect(img)

        if m < 20:
            conf_thres = 0.3
        else:
            conf_thres = 0.5
        pred = non_max_suppression(pred,conf_thres=conf_thres,iou_thres=0.35)[0]
        # 分析预测结果,将重叠度高的预测框按置信度进行筛选,保证每个位置只有一个框
        pred = check_pred(pred,0.7)
        if pred is not None and len(pred):
            # 将预测值去padding并转化为1920x1080比例下的坐标
            pred[:, :4] = scale_coords(img.shape[1:], pred[:, :4], image.shape).round()
        image,img_c = draw_bbox(image,pred,boxes)

        res_txt.write('%s/%s: '%(img_path.split('/')[-2],img_path.split('/')[-1]))
        d,img = compared(pred,boxes,0.4,image,res_txt)
        res_txt.write('\n')
        if issave:
            #TODO:整理单张图比对结果
            name = img_path.split('/')[-1]
            check_compare_res_and_save(d,save_path,name,img)
            cv2.imwrite(save_path+name,img_c)
        l.append(d)
        # print('%s|%s %s 检测完毕(%.3fs,%.3fs)'% (ind,len(lines)-1,img_path,(t1-t0),(t2-t1)))
    t1 = time_synchronized()
    # print('=============val is over=============')
    # print('total time:%s'%(t1-t0))
    return l



if __name__ == '__main__':
    path = '/home/sysman/zlf/DL_project/new_val_project/20201130_new_anchors/'
    if not os.path.exists(path):
        os.makedirs(path)
    # model = Model('../count_10w_yolov5_350_quant.mnn')
    model = ModelV('/home/data/yolov5_pt/runs/exp16_new_anchors_4000_kmeans/weights/last.pt')
    l = val(model,
            '../val.txt',
            issave=True,
            save_path=path,
            result_txt=path+'result.txt')



