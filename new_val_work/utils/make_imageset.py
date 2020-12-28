import os

f = open('/home/sysman/zlf/055000-055885/train.txt','w')
path = '/home/sysman/zlf/055000-055885/JPEGImages/'
for img in os.listdir(path):
    img_path = path+img
    f.write(img_path+'\n')
f.close()