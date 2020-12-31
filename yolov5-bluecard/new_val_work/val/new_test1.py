'''
add by zlf on 20201104
'''
import os
import torch
import cv2
import time
import numpy as np
from numpy import random
from utils.general import plot_one_box
names = ['CAR', 'CARPLATE', 'BICYCLE', 'TRICYCLE', 'PEOPLE', 'MOTORCYCLE', 'LOGO_AUDI', 'LOGO_BENZE', 'LOGO_BENZC', 'LOGO_BMW', 'LOGO_BUICK', 'LOGO_CHEVROLET', 'LOGO_CITROEN', 'LOGO_FORD', 'LOGO_HONDA', 'LOGO_HYUNDAI', 'LOGO_KIA', 'LOGO_MAZDA', 'LOGO_NISSAN', 'LOGO_PEUGEOT', 'LOGO_SKODA', 'LOGO_SUZUKI', 'LOGO_TOYOTA', 'LOGO_VOLVO', 'LOGO_VW', 'LOGO_ZHONGHUA', 'LOGO_SUBARU', 'LOGO_LEXUS', 'LOGO_CADILLAC', 'LOGO_LANDROVER', 'LOGO_JEEP', 'LOGO_BYD', 'LOGO_BYDYUAN', 'LOGO_BYDTANG', 'LOGO_CHERY', 'LOGO_CARRY', 'LOGO_HAVAL', 'LOGO_GREATWALL', 'LOGO_GREATWALLOLD', 'LOGO_ROEWE', 'LOGO_JAC', 'LOGO_HAFEI', 'LOGO_SGMW', 'LOGO_CASY', 'LOGO_CHANAJNX', 'LOGO_CHANGAN', 'LOGO_CHANA', 'LOGO_CHANGANCS', 'LOGO_XIALI', 'LOGO_FAW', 'LOGO_YQBT', 'LOGO_REDFLAG', 'LOGO_GEELY', 'LOGO_EMGRAND', 'LOGO_GLEAGLE', 'LOGO_ENGLON', 'LOGO_BAOJUN', 'LOGO_DF', 'LOGO_JINBEI', 'LOGO_BAIC', 'LOGO_WEIWANG', 'LOGO_HUANSU', 'LOGO_FOTON', 'LOGO_HAIMA', 'LOGO_ZOTYEAUTO', 'LOGO_MITSUBISHI', 'LOGO_RENAULT', 'LOGO_MG', 'LOGO_DODGE', 'LOGO_FIAT', 'LOGO_INFINITI', 'LOGO_MINI', 'LOGO_TESLA', 'LOGO_SMART', 'LOGO_BORGWARD', 'LOGO_JAGUAR', 'LOGO_HUMMER', 'LOGO_PORSCHE', 'LOGO_LAMBORGHINI', 'LOGO_DS', 'LOGO_CROWN', 'LOGO_LUXGEN', 'LOGO_ACURA', 'LOGO_LINCOLN', 'LOGO_SOUEAST', 'LOGO_VENUCIA', 'LOGO_TRUMPCHI', 'LOGO_LEOPAARD', 'LOGO_ZXAUTO', 'LOGO_LIFAN', 'LOGO_HUANGHAI', 'LOGO_HAWTAI', 'LOGO_REIZ', 'LOGO_CHANGHE', 'LOGO_GOLDENDRAGON', 'LOGO_YUTONG', 'LOGO_HUIZHONG', 'LOGO_JMC', 'LOGO_JMCYUSHENG', 'LOGO_LANDWIND', 'LOGO_NAVECO', 'LOGO_QOROS', 'LOGO_OPEL', 'LOGO_YUEJING']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

# 将labels中的值构造成tensor
def build_target(img_path):
    boxes = []
    lable_path = img_path.replace('JPEGImages','labels').replace('.jpg','.txt')
    try:
        f = open(lable_path,'r')
    except Exception as e:
        print(e)
    lines = f.readlines()
    if lines:
        for line in lines:
            box = []
            items = line.strip().split(' ')
            for i,item in enumerate(items):
                item = float(item)
                if (i == 1 or i == 3):
                    item *= 1920
                elif (i == 2 or i == 4):
                    item *= 1080
                box.append(item)
            boxes.append(box)
    boxes = torch.tensor(boxes)
    return boxes

# 统计验证集中的目标数
def check_val_samples(txt):
    d = {}
    f = open(txt,'r')
    lines = f.readlines()
    for line in lines:
        lable_path = line.replace('JPEGImages', 'labels').replace('.jpg', '.txt').strip()
        label_file = open(lable_path,'r')
        targets = label_file.readlines()
        for target in targets:
            cls = int(target.split(' ')[0])
            cls_name = names[cls]
            if cls_name not in d:
                d[cls_name] = 0
            d[cls_name] += 1
        label_file.close()
    f.close()
    return d

# def check_pred(pred,iou_thresh):
#     if pred is None or len(pred) == 1:
#         return pred
#     else:
#         c_pred = pred.clone()
#         for i in range(0,len(pred)-1):
#             conf1 = pred[i][4]
#             for j in range(i+1,len(pred)):
#                 conf2 = pred[j][4]
#                 iou = pbox_iou(pred[i],pred[j])
#                 if iou > iou_thresh:
#                     if conf1 > conf2:
#                         c_pred = c_pred[torch.arange(c_pred.size(0)) != j]
#                     else:
#                         c_pred = c_pred[torch.arange(c_pred.size(0)) != i]
#     return c_pred

def check_pred(pred,iou_thresh):
    if pred is None or len(pred) == 1:
        return pred
    else:
        l = []
        c_pred = pred.clone()
        for i in range(len(pred)):
            is_save = True
            conf1 = pred[i][4]
            for j in range(len(pred)):
                if i == j:
                    continue
                else:
                    conf2 = pred[j][4]
                    iou = pbox_iou(pred[i],pred[j])
                    if iou > iou_thresh:
                        if conf1 > conf2:
                            is_save = True
                        else:
                            is_save = False
                            break
                    else:
                        is_save = True
            if is_save:
                l.append(i)
        c_pred = c_pred[l]
    return c_pred

# 比较
#TODO:根据level修改比对结果
def compared(pred,boxes,iou_thres=0.4,img=None,res_txt=None):
    d = {}
    # 真值为空
    if boxes.shape == torch.Size([0]):
        # 预测值为空
        if pred is None:
            return d,img
        # 预测值不为空,都算多检
        else:
            for p in pred:
                cls = int(p[-3])
                d['多检'] = {}
                if names[cls] not in d['多检']:
                    d['多检'][names[cls]] = 0
                d['多检'][names[cls]] += 1
                draw_error_bbox_and_write_res(img,p,xywh=False,error='多检',res_txt=res_txt)

    # 真值不为空
    else:
        # 预测值为空,都算漏检
        if pred is None:
            for box in boxes:
                t_cls = int(box[0])
                # 漏检
                # print('漏检:',box)
                d['漏检'] = {}
                if names[t_cls] not in d['漏检']:
                    d['漏检'][names[t_cls]] = 0
                d['漏检'][names[t_cls]] += 1
                draw_error_bbox_and_write_res(img, box, xywh=True, error='漏检',res_txt=res_txt)
        # 预测值不为空
        else:
            # 建立一个容器存放没有被匹配的真值
            undetected_b = []
            # 建立一个容器存放没有使用过的预测值
            nouse_p = []

            # 真值->预测值,判断出误检,漏检
            for box in boxes:
                # 真值是否被检测
                b_isdetected = False
                t_cls = int(box[0])
                for p in pred:
                    cls = int(p[-3])
                    # 计算iou
                    iou = box_iou1(p,box[1:5])
                    # 预测结果正确
                    if iou > iou_thres and (t_cls == cls):
                        b_isdetected = True
                        break

                    # 没有预测结果与真值匹配
                    elif iou < iou_thres:
                        b_isdetected = False

                    # 预测结果错误
                    elif iou > iou_thres and (t_cls != cls):
                        # d['误检'] = {}
                        # if names[cls] not in d['误检']:
                        #     d['误检'][names[cls]] = 0
                        # d['误检'][names[cls]] += 1
                        # draw_error_bbox(img, p, xywh=True, error='误检')
                        b_isdetected = True
                        break

                if b_isdetected == False and (box.numpy().tolist() not in undetected_b):
                    undetected_b.append(box.numpy().tolist())

            # 预测值->真值,判断出多检
            for p in pred:
                p_isused = False
                p_writed = False
                cls = int(p[-3])
                for box in boxes:
                    t_cls = int(box[0])
                    iou = box_iou1(p, box[1:5])
                    # 预测结果正确
                    if iou > iou_thres and (t_cls == cls):
                        p_isused = True
                        break

                    # 没有预测结果与真值匹配
                    elif iou < iou_thres:
                        p_isused = False

                    # 预测结果错误
                    elif iou > iou_thres and (t_cls != cls):
                        d['误检'] = {}
                        if names[cls] not in d['误检']:
                            d['误检'][names[cls]] = 0
                        d['误检'][names[cls]] += 1
                        draw_error_bbox_and_write_res(img, p, xywh=False, error='误检',res_txt=res_txt,t_cls=t_cls)
                        # b_isdetected = True
                        p_isused = True
                        p_writed = True
                        break
                p = p.cpu().detach()
                if p_isused == False and (p.numpy().tolist() not in nouse_p):
                    nouse_p.append(p.numpy().tolist())
                # 将错误外的检测结果也写到txt中
                else:
                    if not p_writed:
                        draw_error_bbox_and_write_res(img,p,xywh=False,res_txt=res_txt)

            # 没有被匹配的(不论结果)真值都是漏检
            d['漏检'] = {}
            for box in undetected_b:
                t_cls = int(box[0])
                if names[t_cls] not in d['漏检']:
                    d['漏检'][names[t_cls]] = 0
                d['漏检'][names[t_cls]] += 1
                draw_error_bbox_and_write_res(img, box, xywh=True, error='漏检',res_txt=res_txt)
            # 没有用到的预测值都是多检
            d['多检'] = {}
            for p in nouse_p:
                cls = int(p[-3])
                if names[cls] not in d['多检']:
                    d['多检'][names[cls]] = 0
                # new_cls = define_target_level(p,cls,xywh=False)
                # if new_cls:
                #     if new_cls not in d['多检']:
                #         d['多检'][new_cls] = 0
                #     d['多检'][new_cls] += 1
                d['多检'][names[cls]] += 1
                draw_error_bbox_and_write_res(img, p, xywh=False, error='多检',res_txt=res_txt)

    return d,img

#TODO:整理单张图比对结果并保存图片到相应位置
def check_compare_res_and_save(d,save_path,name,img):
    if d:
        for k,v in d.items():
            save_dir = save_path+k+'/'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            if d[k]:
                cv2.imwrite(save_dir+name,img)
            else:
                continue
    else:
        return

# 比对用iou计算函数
def box_iou1(p,box):
    p_x1 = p[0].item()
    p_y1 = p[1].item()
    p_x2 = p[2].item()
    p_y2 = p[3].item()

    box_x1 = (box[0] - box[2] / 2).item()
    box_y1 = (box[1] - box[3] / 2).item()
    box_x2 = (box[0] + box[2] / 2).item()
    box_y2 = (box[1] + box[3] / 2).item()

    union = (min(p_x2,box_x2) - max(p_x1,box_x1))*\
            (min(p_y2,box_y2) - max(p_y1,box_y1))
    iou = union/((p[2]-p[0]) * (p[3]-p[1]) + box[2] * box[3] - union)
    return iou

def pbox_iou(p1,p2):
    p1_x1 = p1[0].item()
    p1_y1 = p1[1].item()
    p1_x2 = p1[2].item()
    p1_y2 = p1[3].item()

    p2_x1 = p2[0].item()
    p2_y1 = p2[1].item()
    p2_x2 = p2[2].item()
    p2_y2 = p2[3].item()

    union = (min(p1_x2, p2_x2) - max(p1_x1, p1_x1)) * \
            (min(p1_y2, p2_y2) - max(p1_y1, p2_y1))
    iou = union / ((p1[2] - p1[0]) * (p1[3] - p1[1]) + ((p2[2] - p2[0]) * (p2[3] - p2[1]) - union))
    return iou

# 计算目标框面积
def area(box,xywh=True):
    if xywh:
        a = box[2] * box[3]
    else:
        w = box[2] - box[0]
        h = box[3] - box[1]
        a = w * h
    return a

# 根据物体类别,目标框面积或长宽比,推断物体属于大中小哪个级别
def define_target_level(box,cls,xywh=True):
    cls = int(cls)
    cls_ind = cls
    if xywh:
        w = box[2]
        h = box[3]
        r = w/h
    else:
        w = box[2] - box[0]
        h = box[3] - box[1]
        r = w/h
    a = area(box,xywh)
    cls = names[cls]
    # 车
    if cls_ind == 0:
        pass
    # 车牌
    elif cls_ind == 1:
        if w > 50:
            return '大' + cls
        else:
            return '小' + cls
    elif cls_ind == 2:
        pass
    elif cls_ind == 3:
        pass
    elif cls_ind == 4:
        pass
    elif cls_ind == 5:
        pass
    else:
        pass
    return 0

# 将真值和检测结果画到图上
def draw_bbox(img,pred,boxes):
    img_c = img.copy()
    for box in boxes:
        x1 = int((box[1] - box[3] / 2))
        y1 = int((box[2] - box[4] / 2))
        x2 = int((box[1] + box[3] / 2))
        y2 = int((box[2] + box[4] / 2))
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
    if pred != None:
        for box in pred:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            conf = box[4]
            cls = box[5]
            obj_conf = box[6]
            cls_conf = box[7]
            text = '%s|%.2f'%(names[int(cls)], conf)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(img, text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
            label = '%s %.2f %.2f %.2f' % (names[int(cls)], conf, obj_conf, cls_conf)
            plot_one_box(box, img_c, label=label, color=colors[int(cls)], line_thickness=3)
    return img,img_c

# 将错误的目标框画在图上并把所有检测结果记录在txt中
def draw_error_bbox_and_write_res(img,box,xywh=False,error=None,res_txt=None,t_cls=None):

    if xywh:
        cls_num = int(box[0])
        x1 = int((box[1]-box[3]/2))
        y1 = int((box[2]-box[4]/2))
        x2 = int((box[1]+box[3]/2))
        y2 = int((box[2]+box[4]/2))
        if error:
            if error == '漏检':
                color = (0, 255, 0)
            elif error == '误检':
                color = (0, 255, 255)
            elif error == '多检':
                color = (255, 255, 255)
            text = '%s' % (names[cls_num])
            cv2.rectangle(img, (x1, y1), (x2, y2), color,2)
            # cv2.rectangle(img, (x1, y1), (x1 + 200, y1 + 25), (0, 0, 0), -1)
            cv2.putText(img, text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            # 如果是误检,需要把真实类别记录下来
            if error == '误检':
                line = "['%s',%d,%d,%d,%d,'%s','%s']" % (names[cls_num], x1, y1, x2, y2,names[t_cls], error)
            else:
                line = "['%s',%d,%d,%d,%d,'%s']" % (names[cls_num], x1, y1, x2, y2,error)
        else:
            line = "['%s',%d,%d,%d,%d]" % (names[cls_num], x1, y1, x2, y2)
    else:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        conf = box[4]
        if type(conf) == torch.Tensor:
            conf = conf.item()
        cls_num = int(box[5])
        if error:
            if error == '漏检':
                color = (0, 255, 0)
            elif error == '误检':
                color = (0, 255, 255)
            elif error == '多检':
                color = (255, 255, 255)
            text = '%s|%.3f'%(names[cls_num],conf)
            cv2.rectangle(img, (x1, y1), (x2, y2), color,2)
            # cv2.rectangle(img, (x1, y1), (x1 + 200, y1 + 30), (0, 0, 0), -1)
            cv2.putText(img, text, (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            if error == '误检':
                line = "['%s',%d,%d,%d,%d,'%s','%s']" % (names[cls_num], x1, y1, x2, y2,names[t_cls], error)
            else:
                line = "['%s',%.2f,%d,%d,%d,%d,'%s']" % (names[cls_num], conf, x1, y1, x2, y2,error)
        else:
            line = "['%s',%.2f,%d,%d,%d,%d]" % (names[cls_num], conf, x1, y1, x2, y2)
    res_txt.write(line)

# 统计验证集比对结果
def count_target(l):
    d = {'漏检': {}, '误检': {}, '多检': {}}
    for res in l:
        if res:
            for k,v in res.items():
                for k1,v1 in v.items():
                    if k1 not in d[k]:
                        d[k][k1] = 0
                    d[k][k1] += v1
    return d


# 评分
def scored(d):
    pass




