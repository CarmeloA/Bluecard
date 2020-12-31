# -*- coding=utf-8 -*- #
'''
????????(??????) v1.0
2020-8-7
'''
import time
from argparse import ArgumentParser
import torch


# ????????[[],[],[],...]?
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

def insert_marks(target):
    s = list(target)
    i = s.index(',')
    s.insert(1, '"')
    s.insert(i + 1, '"')
    s = ''.join(s)
    return s

def build_target1(line):
    boxes = []
    help_list, img_name = modify_target(line)
    for target in help_list:
        box = []
        s = insert_marks(target)
        s = eval(s)
        s = s[:6]
        print(s)
        for i, item in enumerate(s):
            if i == 0:
                continue
            item = float(item)
            if (i == 1 or i == 3):
                item *= 1920
            elif (i == 2 or i == 4):
                item *= 1080
            box.append(item)
            boxes.append(box)
    boxes = torch.tensor(boxes)
    return boxes

if __name__ == '__main__':
    f = open('0728_cuda_?????.txt','r',encoding='gbk')
    lines = f.readlines()
    for line in lines:
        boxes = []
        help_list, img_name = modify_target(line)
        for target in help_list:
            box = []
            s = insert_marks(target)
            s = eval(s)
            s = s[:6]
            print(s)
            for i,item in enumerate(s):
                item = float(item)
                if (i == 1 or i == 3):
                    item *= 1920
                elif (i == 2 or i == 4):
                    item *= 1080
                box.append(item)



