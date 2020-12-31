import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from new_val_work.val.new_val import val
from new_val_work.val.new_test1 import build_target
from new_val_work.val.init_model import ModelV
from new_val_work.val.result_op import check_result

def val_and_count(weights,count_f,imgs_path,epoch,iteration,
                  miss_detect_l, wrong_detect_l, extra_detect_l, total_error_l):
    model = ModelV(weights)
    l = val(model, imgs_path, issave=False, result_txt='result.txt')
    d1, d2, miss_detect, wrong_detect, extra_detect, total_error = check_result('result.txt')
    miss_detect_l.append(miss_detect)
    wrong_detect_l.append(wrong_detect)
    extra_detect_l.append(extra_detect)
    total_error_l.append(total_error)
    line1 = 'model_epoch%s_%s: ' % (epoch, iteration)
    line2 = '{"total":%d,"miss":%d,"wrong":%d,"extra":%d}' % (
        total_error, miss_detect, wrong_detect, extra_detect)
    line = line1 + line2 + '\n' + str(d2)
    count_f.write(line + '\n')
    return miss_detect, wrong_detect, extra_detect, total_error

def val_local_yolov5_pt(path,imgs_path,count_txt,show_img=True):
    count_f = open(count_txt,'w')
    miss_detect_l = []
    wrong_detect_l = []
    extra_detect_l = []
    total_error_l = []
    apix_x = []
    for pt in os.listdir(path):
        pt_info = pt.split('_')
        if len(pt_info) == 5:
            style = pt_info[0]
            epoch = int(pt_info[2])
            iteration = int(pt_info[3])
            if 'ckpt' in style:
                weights = path+pt
                if epoch > 100 and epoch < 200:
                    if iteration == 500:
                        print('weights:',weights)
                        val_and_count(weights,count_f,imgs_path,epoch,iteration,
                                      miss_detect_l,wrong_detect_l,extra_detect_l,total_error_l)
                        apix_x.append('%s_%s'%(epoch,iteration))
                elif epoch >= 200 and epoch < 300:
                    if iteration == 500 or iteration == 750:
                        print('weights:', weights)
                        val_and_count(weights,count_f,imgs_path,epoch,iteration,
                                      miss_detect_l, wrong_detect_l, extra_detect_l, total_error_l)
                        apix_x.append('%s_%s' % (epoch, iteration))
                elif epoch >= 300:
                    if iteration == 500 or iteration == 750:
                        print('weights:', weights)
                        val_and_count(weights, count_f, imgs_path, epoch, iteration,
                                      miss_detect_l, wrong_detect_l, extra_detect_l, total_error_l)
                        apix_x.append('%s_%s' % (epoch, iteration))
    count_f.close()
    if show_img:
        plt.plot(apix_x,total_error_l)
        plt.savefig('exp3_total_error.jpg')
        # plt.show()


def check_count_result(txt):
    total = []
    detail = []
    f = open(txt,'r')
    lines = f.readlines()
    for i in range(0,len(lines)-1,2):
        model_total = lines[i].strip()
        model_detail = lines[i+1].strip()
        model_name = model_total.split(' ')[0][:-1]
        total_count = eval(model_total.split(' ')[1])
        detail_count = eval(model_detail)
        total_count['name'] = model_name
        detail_count['name'] = model_name
        total.append(total_count)
        detail.append(detail_count)
    f.close()
    return total,detail

class TotalCountHandle():
    def __init__(self,total):
        self.total = total

    def find_error_count_min(self,error,error1):
        min = self.total[0][error]
        l = []
        for model_total in self.total:
            error_count = model_total[error]
            if error_count < min:
                l = []
                min = error_count
                l.append(model_total)
            elif error_count == min:
                l.append(model_total)
        self.sort(l,error1)
        return l

    @staticmethod
    def sort(l,error):
        l.sort(key=lambda x:x[error])

class DetailCountHandle():
    def __init__(self,detail):
        self.detail = detail

    def find_error_count_min(self,error,cls):
        min = self.detail[0][error][cls]
        l = []
        for model_detail in self.detail:
            error_count = model_detail[error][cls]
            if error_count < min:
                l = []
                min = error_count
                l.append(model_detail)
            elif error_count == min:
                l.append(model_detail)
        return l







if __name__ == '__main__':
    # val_local_yolov5_pt('/home/data/yolov5_pt/runs/exp3/weights/',
    #                     '/home/sysman/zlf/yolov5-master/new_val_work/val.txt',
    #                     '/home/sysman/zlf/yolov5-master/new_val_work/exp3_result_count1.txt')
    total, detail = check_count_result('../exp3_result_count1.txt')
    tch = TotalCountHandle(total)
    l = tch.find_error_count_min('miss','wrong')
    print(l)
    print(len(l))
    dch = DetailCountHandle(detail)
    l1 = dch.find_error_count_min('漏检','CAR')
    print(l1)
    print(len(l1))

