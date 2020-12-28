import cv2
import os
from PIL import Image,ImageStat
import torch
from torchvision import transforms

def brightness1( im_file ):
    im = Image.open(im_file).convert('L')
    stat = ImageStat.Stat(im)
    return stat.mean[0]



path = '/home/sysman/zlf/DL_project/new_val_project/mnn-error/night/'
for name in os.listdir(path):
    if name.endswith('.jpg'):
        img = path + name
        im = Image.open(img)
        m = transforms.ToTensor()(im)
        m = m.mean() * 255
        print(m)
