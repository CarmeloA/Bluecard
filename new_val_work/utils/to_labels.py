# -*- coding=utf-8 -*- #
'''
识别结果过滤程序(筛选掉小目标) v1.0
2020-8-7
'''
import time
from argparse import ArgumentParser
import torch
import cv2

NAMES = ['CAR', 'CARPLATE', 'BICYCLE', 'TRICYCLE', 'PEOPLE', 'MOTORCYCLE', 'LOGO_AUDI', 'LOGO_BENZE', 'LOGO_BENZC', 'LOGO_BMW', 'LOGO_BUICK', 'LOGO_CHEVROLET', 'LOGO_CITROEN', 'LOGO_FORD', 'LOGO_HONDA', 'LOGO_HYUNDAI', 'LOGO_KIA', 'LOGO_MAZDA', 'LOGO_NISSAN', 'LOGO_PEUGEOT', 'LOGO_SKODA', 'LOGO_SUZUKI', 'LOGO_TOYOTA', 'LOGO_VOLVO', 'LOGO_VW', 'LOGO_ZHONGHUA', 'LOGO_SUBARU', 'LOGO_LEXUS', 'LOGO_CADILLAC', 'LOGO_LANDROVER', 'LOGO_JEEP', 'LOGO_BYD', 'LOGO_BYDYUAN', 'LOGO_BYDTANG', 'LOGO_CHERY', 'LOGO_CARRY', 'LOGO_HAVAL', 'LOGO_GREATWALL', 'LOGO_GREATWALLOLD', 'LOGO_ROEWE', 'LOGO_JAC', 'LOGO_HAFEI', 'LOGO_SGMW', 'LOGO_CASY', 'LOGO_CHANAJNX', 'LOGO_CHANGAN', 'LOGO_CHANA', 'LOGO_CHANGANCS', 'LOGO_XIALI', 'LOGO_FAW', 'LOGO_YQBT', 'LOGO_REDFLAG', 'LOGO_GEELY', 'LOGO_EMGRAND', 'LOGO_GLEAGLE', 'LOGO_ENGLON', 'LOGO_BAOJUN', 'LOGO_DF', 'LOGO_JINBEI', 'LOGO_BAIC', 'LOGO_WEIWANG', 'LOGO_HUANSU', 'LOGO_FOTON', 'LOGO_HAIMA', 'LOGO_ZOTYEAUTO', 'LOGO_MITSUBISHI', 'LOGO_RENAULT', 'LOGO_MG', 'LOGO_DODGE', 'LOGO_FIAT', 'LOGO_INFINITI', 'LOGO_MINI', 'LOGO_TESLA', 'LOGO_SMART', 'LOGO_BORGWARD', 'LOGO_JAGUAR', 'LOGO_HUMMER', 'LOGO_PORSCHE', 'LOGO_LAMBORGHINI', 'LOGO_DS', 'LOGO_CROWN', 'LOGO_LUXGEN', 'LOGO_ACURA', 'LOGO_LINCOLN', 'LOGO_SOUEAST', 'LOGO_VENUCIA', 'LOGO_TRUMPCHI', 'LOGO_LEOPAARD', 'LOGO_ZXAUTO', 'LOGO_LIFAN', 'LOGO_HUANGHAI', 'LOGO_HAWTAI', 'LOGO_REIZ', 'LOGO_CHANGHE', 'LOGO_GOLDENDRAGON', 'LOGO_YUTONG', 'LOGO_HUIZHONG', 'LOGO_JMC', 'LOGO_JMCYUSHENG', 'LOGO_LANDWIND', 'LOGO_NAVECO', 'LOGO_QOROS', 'LOGO_OPEL', 'LOGO_YUEJING']
# 将每一行改写为“[[],[],[],...]”
def modify_target(line):
    l = line.split(' ')
    img_name = l[0][:-1]
    target_info = l[1]
    help_list = target_info.strip().split('][')
    for i in range(len(help_list)):
        if len(help_list) == 1:
            return help_list,img_name
        else:
            if i == 0:
                help_list[i] = help_list[i] + ']'
            elif i == len(help_list) - 1:
                help_list[i] = '[' + help_list[i]
            else:
                help_list[i] = '[' + help_list[i] + ']'
    return help_list,img_name

def insert_marks(target,num,index):
    s = list(target)
    i = s.index(',')
    s.insert(1, '"')
    s.insert(i + 1, '"')
    if num == 5:
        s = ''.join(s)
        return s
    elif num == 6:
        s.insert(-1, '"')
        s.insert(-index, '"')
        s = ''.join(s)
        return s


def count_comma(target):
    num = 0
    index = 0
    for i,s in enumerate(target):
        if s == ',':
            num += 1
            index = i
    index = len(target)-index
    return num,index

def build_target1(line,label_path=None,flag=False,total=None,current=None):
    boxes = []
    help_list, img_name = modify_target(line)
    if flag:
        label_txt = label_path+img_name.replace('.jpg','.txt')
        print('%s|%s:%s'%(str(current),str(total),label_txt))
        f = open(label_txt,'a')
        img = cv2.imread('/home/data/TestSampleLib/马泉营out/'+img_name)
        height = img.shape[0]
        width = img.shape[1]
    for target in help_list:
        box = []
        if target == '[]':
            boxes = torch.tensor([])
            return boxes
        else:
            num,index = count_comma(target)
            s = insert_marks(target,num,index)
            s = eval(s)
            s = s[:6]
            # print(s)
            x1 = int(s[2])
            y1 = int(s[3])
            x2 = int(s[4])
            y2 = int(s[5])
            w = x2 - x1
            h = y2 - y1
            x = int(x1 + w/2)
            y = int(y1 + h/2)
            if flag:
                cls = s[0]
                if cls not in NAMES:
                    cls = 'LOGO_' + cls
                cls_num = int(NAMES.index(cls))
                x = x/width
                y = y/height
                w = w/width
                h = h/height
                label = '%d %.7f %.7f %.7f %.7f'%(cls_num,x,y,w,h)
                f.write(label+'\n')
            new_s = [s[0],x,y,w,h]
            for i, item in enumerate(new_s):
                if i == 0:
                    if item not in NAMES:
                        item = 'LOGO_'+item
                    item = NAMES.index(item)
                item = int(item)
                box.append(item)
        boxes.append(box)
    if flag:
        f.close()
    boxes = torch.tensor(boxes)
    return boxes

if __name__ == '__main__':
    f = open('/home/sysman/zlf/0728_cuda_马泉营out.txt','r',encoding='gbk')
    lines = f.readlines()
    for ind,line in enumerate(lines):
        boxes = build_target1(line,label_path='/home/data/labels/马泉营out/',flag=True,
                              total=len(lines),current=ind)



