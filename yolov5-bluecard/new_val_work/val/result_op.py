'''
add by zlf on 20201111
'''
#后处理result.txt
def check_result(result):
    # 整体预测结果统计
    d1 = {}
    # 错误统计
    d2 = {}
    f = open(result,'r')
    lines = f.readlines()
    for line in lines:
        help_list,img_name = modify_target(line)
        # print(img_name)
        if help_list:
            for t in help_list:
                t = eval(t)
                error,cls,t_error_count,new_cls,f_error_count = analyse_target(t)
                # 统计检测结果,每个目标记录一次
                if cls not in d1:
                    d1[cls] = 0
                d1[cls] += 1
                # 统计检测错误结果
                if error:
                    if error not in d2:
                        d2[error] = {}
                    if cls not in d2[error]:
                        d2[error][cls] = 0
                    if new_cls not in d2[error]:
                        d2[error][new_cls] = 0
                    d2[error][cls] += t_error_count
                    d2[error][new_cls] += f_error_count
    d2, miss_detect, wrong_detect, extra_detect, total_error = del_no_count_target(d2)
    # print(d2)
    return d1,d2, miss_detect, wrong_detect, extra_detect, total_error



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
        if cls == 'CAR':
            if area < 216 * 384:
                # 小车漏检可以忽略
                new_cls = 'SMALL_' + cls
                return error,cls,0,new_cls,1
            else:
                return error,cls,1,cls,0
        elif cls == 'CARPLATE':
            if area < 50 * 20:
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
        x1 = t[2]
        y1 = t[3]
        x2 = t[4]
        y2 = t[5]
        area = (x2 - x1) * (y2 - y1)
        if cls == 'CAR':
            if area < 216 * 384:
                # 小车可以忽略
                new_cls = 'SMALL_' + cls
                return error,cls,0,new_cls,1
            else:
                return error,cls,1,cls,0
        elif cls == 'CARPLATE':
            if area < 50 * 20:
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

    for k,v in d1.items():
        for k1,v1 in v.items():
            if k == '漏检':
                miss_detect += v1
            elif k == '误检':
                wrong_detect += v1
            elif k == '多检':
                extra_detect += v1
    total_error = miss_detect+wrong_detect+extra_detect
    return d1,miss_detect,wrong_detect,extra_detect,total_error

if __name__ == '__main__':
    d1,d2, miss_detect, wrong_detect, extra_detect, total_error = check_result('/home/sysman/zlf/DL_project/new_val_project/yolov51/result.txt')
    print('整体检测结果统计:',d1)
    print('错误检测结果统计:',d2)