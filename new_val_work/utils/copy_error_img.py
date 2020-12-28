import os
import shutil

def quick_copy(path,save_path,copy_path):
    if not os.path.exists(copy_path):
        os.mkdir(copy_path)
    for dir in os.listdir(save_path):
        if dir.endswith('.jpg') or dir.endswith('.txt'):
            continue
        else:
            if dir in ['miss','wrong','extra']:
                if not os.path.exists(copy_path+dir):
                    os.mkdir(copy_path+dir)
                for img in os.listdir(save_path+dir):
                    shutil.copy(path+img,copy_path+'/'+dir+'/'+img)

def quick_copy1(path,save_path,copy_path):
    if not os.path.exists(copy_path):
        os.mkdir(copy_path)
    for img in os.listdir(save_path):
        if img.endswith('.jpg'):
            for p in path:
                if os.path.exists(p + img):
                    shutil.copy(p + img, copy_path+ img)

def quick_copy2(inference_path,img_path,img_copy_path,label_path,label_copy_path):
    for dir in os.listdir(inference_path):
        if dir.endswith('.jpg') or dir.endswith('.txt'):
            continue
        else:
            if dir in ['miss','wrong','extra']:
                for img in os.listdir(inference_path+dir):
                    label = img.replace('.jpg','.txt')
                    shutil.copy(img_path+img,img_copy_path+img)
                    shutil.copy(label_path+label,label_copy_path+label)


if __name__ == '__main__':
    # quick_copy('/home/data/TestSampleLib/港澳车牌/',
    #            '/home/data/inference/20201208_0728_cuda_港澳车牌/',
    #            '/home/data/inference/港澳车牌_error/')
    # quick_copy1(['/home/data/TestSampleLib/港澳车牌/','/home/data/TestSampleLib/新能源车牌/'],
    #             '/home/data/tmp','/home/data/tmp1/')
    quick_copy2('/home/data/inference/20201214_阿里/',
                '/home/data/TestSampleLib/阿里/JPEGImages/',
                '/home/data/samples_20201215/阿里/JPEGImages/',
                '/home/data/labels/阿里/',
                '/home/data/samples_20201215/阿里/labels/')