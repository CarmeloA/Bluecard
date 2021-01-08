import cv2

f=open('/home/sysman/gate_Sample/VOCdevkit/VOC2017/labels/000614.txt','r')
img = cv2.imread('/home/sysman/gate_Sample/VOCdevkit/VOC2017/JPEGImages/000614.jpg')
height = img.shape[0]
width = img.shape[1]

lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    x = float(line[1])
    y = float(line[2])
    w = float(line[3])
    h = float(line[4])

    x1 = int((x-w/2)*width)
    y1 = int((y-h/2)*height)
    x2 = int((x+w/2)*width)
    y2 = int((y+h/2)*height)

    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
img = cv2.resize(img,(width//2,height//2))
cv2.imshow('win',img)
cv2.waitKey(0)