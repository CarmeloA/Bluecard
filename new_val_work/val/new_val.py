import sys
sys.path.append('/home/sysman/zlf/new_val_work')
from val.new_test1 import build_target,compared,count_target,\
    draw_bbox,check_compare_res_and_save,check_pred,write_label

from utils.general import non_max_suppression,scale_coords,non_max_suppression_for_logo,letterbox
from utils.to_labels import build_target1
import cv2
import torch
import time
import os
from torchvision import transforms as tf
from val.init_model import Model
from tqdm import tqdm
import gc


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def val(model,imgs_path,issave=True,save_path='',result_txt='',logo=None,only_detect=False,flag=0):
    # res_txt = open(result_txt,'a')
    # original_result_txt = result_txt.replace('result','original_result')
    # original_res_txt = open(original_result_txt,'a')
    if imgs_path.endswith('.txt'):
        lines = open(imgs_path,'r',encoding='gbk')
        # lines = f.readlines()
    else:
        lines = (imgs_path+name for name in os.listdir(imgs_path) if name.endswith('.jpg'))
        # total = list(lines)
        lines1 = [imgs_path+name for name in os.listdir(imgs_path) if name.endswith('.jpg')]
        print(sys.getsizeof(lines))
    for ind,line in enumerate(lines):

        res_txt = open(result_txt, 'a')
        original_result_txt = result_txt.replace('result', 'original_result')
        original_res_txt = open(original_result_txt, 'a')

        if only_detect:
            img_path = line.strip()
            name = img_path.split('/')[-1]
            boxes = torch.tensor([])
        else:
            if flag == 0:
                img_path = line.strip()
                name = img_path.split('/')[-1]
                boxes = build_target(img_path)
            else:
                boxes = build_target1(line)
                img_path = '/home/data/TestSampleLib/马泉营out/'+line[:10]
                name = line[:10]
        img = cv2.imread(img_path)
        image = img.copy()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img,(320,180))
        # img = cv2.copyMakeBorder(img, 6, 6, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = letterbox(img, new_shape=(320, 320))[0]
        img = img / (255., 255., 255.)
        img = img.transpose((2,0,1))
        # 计算图片亮度,调整conf
        m = img.mean()*255.
        # print('mean:',m)

        t0 = time.time()
        pred = model.detect(img)
        t1 = time.time()
        if m < 20:
            conf_thres = 0.3
        else:
            conf_thres = 0.5
        if logo:
            # 检测出所有logo的nms,先降低conf检出所有物体,再提高conf重新筛选非logo类
            pred = non_max_suppression_for_logo(pred,conf_thres=0.01,iou_thres=0.35,logo_conf=conf_thres)[0]
        else:
            pred = non_max_suppression(pred,conf_thres=conf_thres,iou_thres=0.3)[0]
            # pred_for_low_conf = non_max_suppression(pred,conf_thres=0.1,iou_thres=0.2)[0]

        # 分析预测结果,将重叠度高的预测框按置信度进行筛选,保证每个位置只有一个框
        pred_after_check = check_pred(pred,0.7)


        if pred_after_check is not None and len(pred_after_check):
            # 将预测值去padding并转化为1920x1080比例下的坐标
            pred_after_check[:, :4] = scale_coords(img.shape[1:], pred_after_check[:, :4], image.shape).round()
            pred[:, :4] = scale_coords(img.shape[1:], pred[:, :4], image.shape).round()

        image,img_c = draw_bbox(image,pred_after_check,boxes)
        if only_detect:
            # write label for 'only detect' img
            if not os.path.exists(save_path+'labels/'):
                os.mkdir(save_path+'labels/')
            txt_name = name.replace('.jpg','.txt')
            label_txt = open(save_path+'labels/'+txt_name,'w')
            write_label(pred_after_check,img_c,label_txt)
            if not os.path.exists(save_path+'only_detect/'):
                os.mkdir(save_path+'only_detect/')
            cv2.imwrite(save_path+'only_detect/'+ name, img_c)
            print('%s is saved'%name)
            continue
        # if flag:
        #     # write label for 'flag = 1'
        #     if not os.path.exists(save_path+'labels/'):
        #         os.mkdir(save_path+'labels/')
        #     txt_name = name.replace('.jpg','.txt')
        #     label_txt = open(save_path+'labels/'+txt_name,'w')
        #     write_label(pred_after_check,img_c,label_txt)


        res_txt.write('%s/%s: '%(img_path.split('/')[-2],img_path.split('/')[-1]))
        original_res_txt.write('%s/%s: '%(img_path.split('/')[-2],img_path.split('/')[-1]))
        # 原始结果
        d_,img_ = compared(pred,boxes,0.4,None,original_res_txt,img_path=img_path,check_error=None)
        # 过滤了一些小目标
        d,img = compared(pred_after_check,boxes,0.4,image,res_txt,img_path=img_path,check_error=1)
        res_txt.write('\n')
        original_res_txt.write('\n')

        res_txt.close()
        original_res_txt.close()

        t2 = time.time()
        if issave:
            # name = img_path.split('/')[-1]
            check_compare_res_and_save(d,save_path,name,img)
            if not os.path.exists(save_path+'result/'):
                os.mkdir(save_path+'result/')
            cv2.imwrite(save_path+'result/'+name,img_c)
        # l.append(d)
        print('%s|%s %s 检测完毕(%.3fs,%.3fs)'% (ind,len(lines1)-1,img_path,(t1-t0),(t2-t1)))
        if ind % 100 == 0 and ind != 1:
            gc.collect()
            print('==========clear==========')

    print('==========val is over==========')




if __name__ == '__main__':
    path = '/home/data/inference/20201222_ADMM/ADMM_quan_mnn_阿里/'
    if not os.path.exists(path):
        os.makedirs(path)
    # model = Model('../count_10w_yolov5_350_quant.mnn')
    model = Model('../weights/20201214best_quan_20w_ADMM.mnn',mnn_quan=False)
    val(model,
            '/home/data/TestSampleLib/阿里/',
            issave=True,
            save_path=path,
            result_txt=path+'result.txt',logo=0,only_detect=False,flag=0)


