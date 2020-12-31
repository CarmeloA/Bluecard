import os
import shutil
import random
dir = '马泉营out'
test = open('test.txt','a')
img_paths = '/home/data/TestSampleLib/%s/'%dir
label_paths = '/home/data/labels/%s/'%dir

images = os.listdir(img_paths)
test_image = random.sample(images,len(images)//2)
print(len(test_image))
for i,name in enumerate(test_image):
    print('%s|%s'%(i,len(test_image)-1))
    test_img_path = '/home/sysman/gate_Sample/VOCdevkit/VOC2017/test_samples/JPEGImages/'+name
    test.write(test_img_path+'\n')
    test_label_path = test_img_path.replace('JPEGImages','labels').replace('.jpg','.txt')
    shutil.copy(img_paths+name,test_img_path)
    shutil.copy(label_paths+name.replace('.jpg','.txt'),test_label_path)