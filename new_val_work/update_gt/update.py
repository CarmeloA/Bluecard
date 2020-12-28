import os
import cv2
from val.result_op import modify_target
from val.new_test1 import names
'''
1. load error images
2. load detect result
3. load ground truth
4. compared
5. update ground truth
'''

def main_update(path):
    # load error images path
    # load detect result
    d = load_error_images_and_detect_result(path,flag=False)
    print(d)
    update_ground_truth(d,'/home/data/test/')

def load_error_images_and_detect_result(path,flag=True):
    # path: inference result for images
    # create by new_val.py
    d = {}
    f = open(path + 'result.txt', 'r')
    lines = f.readlines()
    f.close()
    for item in os.listdir(path):
        if item.endswith('.txt'):
            continue
        else:
            if flag:
                # all error
                if item in ['漏检','误检','多检']:
                    if item not in d:
                        d[item] = None
                    l = load_error_images(path+item)
                    l = load_detect_result(l,lines)
                    d[item] = l
                # else:
                #     print('useless dir:%s'%item)
            else:
                if item in ['miss','wrong','extra']:
                    if item not in d:
                        d[item] = None
                    l = load_error_images(path+item)
                    l = load_detect_result(l, lines)
                    d[item] = l
                # else:
                #     print('no dir!')
    return d


def load_error_images(path):
    l = []
    for img in os.listdir(path):
        l.append([path+'/'+img])
    return l

def load_detect_result(l,lines):
    for path_list in l:
        path = path_list[0]
        name = path.split('/')[-1]
        for line in lines:
            name_in_txt = line.split(' ')[0].split('/')[-1][:-1]
            if name == name_in_txt:
                path_list.append(line)
                break
    return l

def update_ground_truth(d,path):
    for k,v in d.items():

        compared(k,v,path)

def compared(k,v,path):
    print('===========total error:%s==========='%k)
    for ind,item in enumerate(v):
        error_img_path = item[0]
        error_info = item[1]
        help_list, img_name = modify_target(error_info)
        img_path = path+img_name
        # label_path = img_path.replace('JPEGImages','labels').replace('.jpg','.txt')
        label_path = '/home/data/labels/港澳车牌/'+img_name.split('/')[-1].replace('.jpg','.txt')
        print('%s|%s:%s'%(ind,len(v)-1,label_path))
        label_txt = open(label_path,'r')
        lines = label_txt.readlines()
        is_update = False
        label_txt.close()
        error_image = cv2.imread(error_img_path)
        height = error_image.shape[0]
        width = error_image.shape[1]
        for t in help_list:
            error_image_c = error_image.copy()
            t = eval(t)
            label, label_xyxy,error = analyse_target(t, height, width)
            # print(error)
            if error:
                print('error:%s'%error)
                cv2.rectangle(error_image_c,(label_xyxy[0],label_xyxy[1]),(label_xyxy[2],label_xyxy[3]),(255,255,0),3)
                error_image_c = cv2.resize(error_image_c,(width//2,height//2))
                # cv2.imshow(error_img_path,error_image_c)
                cv2.imshow('win',error_image_c)
                key = cv2.waitKey(0)
                if key == 13:
                    # print('press p')
                    lines,inter_list = match_label(lines, label, 0.5, error,width,height)

                    # print('inter:',inter_list)
                    # for inter in inter_list:
                    #     if inter:
                    #         x1 = int(inter[0]*width//1)
                    #         y1 = int(inter[1]*height//1)
                    #         x2 = int(inter[2]*width//1)
                    #         y2 = int(inter[3]*height//1)
                    #         cv2.rectangle(error_image_c, (x1, y1),(x2, y2),(144, 32, 208),-1)
                    # cv2.imshow('win', error_image_c)
                    # cv2.waitKey(0)
                    is_update = True
                    # print(is_update)
            else:
                continue
        # cv2.destroyAllWindows()
        # print(is_update)
        if is_update:
            rewrite_label_txt(label_path, lines)
            print('%s is updated'%label_path)
            print('\n')


def analyse_target(t,height,width):
    detect_cls = t[0]
    if '漏检' in t:
        # 只统计车和牌的漏检情况
        error = '漏检'
        x1 = t[1]
        y1 = t[2]
        x2 = t[3]
        y2 = t[4]
    elif '误检' in t:
        # print('误检:',t)
        error = '误检'
        x1 = t[4]
        y1 = t[5]
        x2 = t[6]
        y2 = t[7]
    elif '多检' in t:
        error = '多检'
        x1 = t[4]
        y1 = t[5]
        x2 = t[6]
        y2 = t[7]
    else:
        error = None
        x1 = t[4]
        y1 = t[5]
        x2 = t[6]
        y2 = t[7]
    w = x2 - x1
    h = y2 - y1
    x = round((x1 + w/2)/width,7)
    y = round((y1 + h/2)/height,7)
    w = round(w/width,7)
    h = round(h/height,7)
    detect_cls = names.index(detect_cls)
    return [str(detect_cls),str(x),str(y),str(w),str(h)],[x1,y1,x2,y2],error


def match_label(lines,label,iou_thresh,error,width,height):
    if error == '多检':
        label_str = ' '.join(label)
        lines.append(label_str)
        return lines,0
    inter_list = []
    for row,line in enumerate(lines):
        # print('line:',line)
        line = line.strip()
        line = line.split(' ')
        iou,inter = box_iou(line, label,width,height)
        inter_list.append(inter)
        # print(iou)
        # print(inter)
        # print(type(inter))
        if iou > iou_thresh:
            if error == '误检':
                label_str = ' '.join(label)
                lines[row] = label_str
            elif error == '漏检':
                lines[row] = None
    lines = list(filter(None,lines))
    return lines,inter_list

def rewrite_label_txt(label_path,lines):
    f = open(label_path,'w')
    new_lines = [line.strip()+'\n' for line in lines]
    for line in new_lines:
        f.write(line)
    f.close()

def box_iou(line,label,width,height):

    gt_x = float(line[1])
    gt_y = float(line[2])
    gt_w = float(line[3])
    gt_h = float(line[4])

    x = float(label[1])
    y = float(label[2])
    w = float(label[3])
    h = float(label[4])

    gt_x1 = gt_x-gt_w/2
    gt_y1 = gt_y-gt_h/2
    gt_x2 = gt_x+gt_w/2
    gt_y2 = gt_y+gt_h/2

    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2

    union_x = min(gt_x2, x2) - max(gt_x1, x1)
    union_y = min(gt_y2, y2) - max(gt_y1, y1)
    if union_x < 0 or union_y < 0:
        return 0,()
    union = union_x * union_y
    iou = union / ((gt_w * gt_h) + (w * h) - union)
    return iou,(min(gt_x2, x2),min(gt_y2, y2),max(gt_x1, x1),max(gt_y1, y1),iou)

# import time
# from multiprocessing.dummy import Pool as ThreadPool
# def process(item):
#    print('????for??')
#    print(item)
#    time.sleep(5)
# items = ['apple', 'bananan', 'cake', 'dumpling']
# pool = ThreadPool()
# pool.map(process, items)
# pool.close()
# pool.join()

if __name__ == '__main__':
    path = '/home/data/inference/20201219_ADMM/ADMM_quan_mnn_港澳车牌/'
    main_update(path)