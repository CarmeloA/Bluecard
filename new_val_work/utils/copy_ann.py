import os
import shutil

path = '/home/sysman/difficult/JPEGImages/'
path1 = '/home/sysman/Samples/HeGeZhuang/labels/'
path2 = '/home/sysman/difficult/labels/'

# for dir in os.listdir(path):
#     if dir == 'all':
#         continue
#     for jpg in os.listdir(path+dir):
#         ann = jpg.replace('.jpg','.xml')
#         shutil.copy(path1+dir+'/'+ann,path2+ann)

for jpg in os.listdir(path):
    name = jpg.replace('.jpg','.txt')
    if os.path.exists(path1+name):
        shutil.copy(path1+name,path2+name)
