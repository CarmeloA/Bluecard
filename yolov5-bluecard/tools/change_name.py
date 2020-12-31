import os
start = 599629
path = '/home/sysman/zlf/all_after_copy/JPEGImages/'
xml_path = path.replace('JPEGImages','Annotations')
label_path = path.replace('JPEGImages','labels')
for jpg in os.listdir(path):
    xml = jpg.replace('.jpg','.xml')
    label = jpg.replace('.jpg','.txt')
    os.rename(path+jpg,path+str(start)+'.jpg')
    os.rename(xml_path+xml,xml_path+str(start)+'.xml')
    os.rename(label_path+label,label_path+str(start)+'.txt')
    start += 1