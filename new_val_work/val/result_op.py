'''
add by zlf on 20201111
'''
#后处理result.txt
import shutil
import os
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
import cv2



error_dict = {'漏检':'miss','多检':'extra','误检':'wrong'}

def check_result(result):
    save_path = result[:-10]
    # 整体预测结果统计
    d1 = {}
    # 错误统计
    d2 = {}
    # 置信度分布汇总
    l = []
    f = open(result,'r')
    lines = f.readlines()
    for line in lines:
        help_list,img_name = modify_target(line)
        name = img_name.strip().split('/')[-1]
        # print(img_name)
        if help_list:
            #TODO:find max area
            all_error_in_img = []
            all_cls_in_img = []
            for t in help_list:
                t = eval(t)
                error,cls,t_error_count,new_cls,f_error_count = analyse_target(t)
                # 统计检测结果,每个目标记录一次
                if cls not in d1:
                    d1[cls] = 0
                d1[cls] += 1
                # 统计检测错误结果
                if error:
                    if new_cls in ['CAR', 'CARPLATE']:
                        all_error_in_img.append(error)
                        all_cls_in_img.append(new_cls)
                    if error not in d2:
                        d2[error] = {}
                    if cls not in d2[error]:
                        d2[error][cls] = 0
                    if new_cls not in d2[error]:
                        d2[error][new_cls] = 0
                    d2[error][cls] += t_error_count
                    d2[error][new_cls] += f_error_count

                    if not os.path.exists(save_path + '%s/' % (error_dict[error])):
                        os.mkdir(save_path + '%s/' % (error_dict[error]))

                    # 再次筛选图片,只保留车,车牌识别错的
                    # if 'LOGO' not in cls and 'SMALL_CAR' not in cls and 'PEOPLE' not in cls and 'SMALL_CARPLATE' not in cls:
                    #     save_path = result[:-10]
                    #     if not os.path.exists(save_path + '%s/' % (error_dict[error])):
                    #         os.mkdir(save_path + '%s/' % (error_dict[error]))
                    #     shutil.copy(save_path + '%s/%s' % (error, name),
                    #                 save_path + '%s/%s' % (error_dict[error], name))
            if len(all_error_in_img):
                if len(all_cls_in_img):
                    for e in all_error_in_img:
                        shutil.copy(save_path + '%s/%s' % (e, name),
                        save_path + '%s/%s' % (error_dict[e], name))

                # 统计某一类别conf分布
                # count_conf_distribution(t, 'CARPLATE', l, name)
        # 画分布图
        # l = np.array(l)
        # plot_hist(l)
    # print(d2)
    return d1,d2



# 将每一行改写为“[[],[],[],...]”
def modify_target(line):
    l = line.split(' ')
    img_name = l[0][:-1]
    target_info = l[1]
    help_list = target_info.strip().split('][')
    for i in range(len(help_list)):
        if len(help_list) == 1:
            if help_list[0] == '':
                help_list = []
            return help_list,img_name
        else:
            if i == 0:
                help_list[i] = help_list[i] + ']'
            elif i == len(help_list) - 1:
                help_list[i] = '[' + help_list[i]
            else:
                help_list[i] = '[' + help_list[i] + ']'
    # print('%s:%s'%(img_name,help_list))
    return help_list,img_name

# 分析每一个结果
def analyse_target(t):
    cls = t[0]
    if '漏检' in t:
        # 只统计车和牌的漏检情况
        error = '漏检'
        x1 = t[1]
        y1 = t[2]
        x2 = t[3]
        y2 = t[4]
        # 过滤条件1:面积
        area = (x2-x1)*(y2-y1)
        # 过滤条件2:宽高
        width = x2 - x1
        height = y2 - y1
        # 过滤条件3: 宽高比
        r = width / height

        if cls == 'CAR':
            if area < 216 * 384 or width < 700:
                # 小车漏检可以忽略
                new_cls = 'SMALL_' + cls
                return error,cls,0,new_cls,1
            else:
                return error,cls,1,cls,0
        elif cls == 'CARPLATE':
            if area < 50 * 20 or width<100 or x1 == 0 or x2 == 1920:
                # 小牌可以忽略
                new_cls = 'SMALL_' + cls
                return error,cls,0,new_cls,1
            else:
                return error,cls,1,cls,0
        else:
            # 其他类型均可忽略
            return error,cls,0,cls,0

    elif '误检' in t:
        # print('误检:',t)
        error = '误检'
        # 只讨论识别错车,车牌的情况
        if 'CAR' in t or 'CARPLATE' in t:
            return error,cls,1,cls,0
        else:
            return error,cls,0,cls,0

    elif '多检' in t:
        error = '多检'
        x1 = t[4]
        y1 = t[5]
        x2 = t[6]
        y2 = t[7]
        area = (x2 - x1) * (y2 - y1)
        width = x2 - x1
        height = y2 - y1
        if cls == 'CAR':
            if area < 216 * 384 or width < 700:
                # 小车可以忽略
                new_cls = 'SMALL_' + cls
                return error,cls,0,new_cls,1
            else:
                return error,cls,1,cls,0
        elif cls == 'CARPLATE':
            if area < 50 * 20 or width<100 or x1 == 0 or x2 == 1920:
                # 小牌可以忽略
                new_cls = 'SMALL_' + cls
                return error,cls,0,new_cls,1
            else:
                return error,cls,1,cls,0
        else:
            return error,cls,0,cls,0

    else:
        return 0,cls,0,cls,0

def del_no_count_target(d):
    d1 = {'漏检':{},'误检':{},'多检':{}}
    for k,v in d.items():
        for k1,v1 in v.items():
            if v1 == 0:
                continue
            else:
                d1[k][k1] = v1

    miss_detect = 0
    wrong_detect = 0
    extra_detect = 0

    for k, v in d1.items():
        for k1, v1 in v.items():
            if k == '漏检':
                miss_detect += v1
            elif k == '误检':
                wrong_detect += v1
            elif k == '多检':
                extra_detect += v1
    total_error = miss_detect + wrong_detect + extra_detect
    return d1,miss_detect,wrong_detect,extra_detect,total_error

# 绘制物体置信度统计直方图
# 统计分布
def count_conf_distribution(t,c,l,name):
    path = '/home/sysman/inference/20201126_val_finetune_P0.6_R0.87/'+name
    if '漏检' in t:
        pass
    else:
        cls = t[0]
        conf = t[1]
        if cls == c:
            l.append(conf)
            if conf < 0.8:
                shutil.copy(path,'/home/sysman/val_under0.7/'+name)

# 绘图
def plot_hist(l):
    print(l)
    plt.hist(l, bins=3, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel("区间")
    # 显示纵轴标签
    plt.ylabel("频数/频率")
    # 显示图标题
    plt.title("频数/频率分布直方图")
    print('start save')
    plt.savefig('distribution.jpg')
    print('save over')
    plt.show()




if __name__ == '__main__':
    d1,d2 = check_result('/home/data/inference/20201219_ADMM/ADMM_quan_mnn_港澳车牌/result.txt')
    print('整体检测结果统计:',d1)
    d2,miss_detect,wrong_detect,extra_detect,total_error = del_no_count_target(d2)
    print('错误检测结果统计:',d2)
    print('总错误数:', total_error)
    print('漏检数:', miss_detect)
    print('误检数:', wrong_detect)
    print('多检数:', extra_detect)